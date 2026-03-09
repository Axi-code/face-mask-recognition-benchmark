import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from fewshot.episode_dataset import EpisodeDataset
from fewshot.proto_net import ProtoNet, prototypical_loss
from utils.dataset import DEFAULT_MEAN, DEFAULT_STD, build_inference_transform, build_train_transform
from utils.metrics import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a prototypical network for few-shot mask recognition.")
    parser.add_argument("--train-root", type=str, default="data/train")
    parser.add_argument("--val-root", type=str, default="data/test")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--n-way", type=int, default=2)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--episodes-per-epoch", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="results/fewshot")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def squeeze_episode(batch, device):
    return {
        "support_images": batch["support_images"].squeeze(0).to(device),
        "support_labels": batch["support_labels"].squeeze(0).to(device),
        "query_images": batch["query_images"].squeeze(0).to(device),
        "query_labels": batch["query_labels"].squeeze(0).to(device),
    }


def run_episode(model, batch, n_way):
    support_embeddings = model(batch["support_images"])
    query_embeddings = model(batch["query_images"])
    loss, accuracy, _ = prototypical_loss(
        support_embeddings,
        batch["support_labels"],
        query_embeddings,
        batch["query_labels"],
        n_way,
    )
    return loss, accuracy


def evaluate(model, dataloader, device, n_way):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = squeeze_episode(raw_batch, device)
            loss, accuracy = run_episode(model, batch, n_way)
            total_loss += loss.item()
            total_acc += accuracy
            total_steps += 1

    return total_loss / max(total_steps, 1), total_acc / max(total_steps, 1)


def plot_history(history, output_path):
    plt.figure(figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Episode Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EpisodeDataset(
        root_dir=args.train_root,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        transform=build_train_transform(args.image_size, DEFAULT_MEAN, DEFAULT_STD, augment=True),
        episodes_per_epoch=args.episodes_per_epoch,
        seed=args.seed,
    )
    val_dataset = EpisodeDataset(
        root_dir=args.val_root,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        transform=build_inference_transform(args.image_size, DEFAULT_MEAN, DEFAULT_STD),
        episodes_per_epoch=args.val_episodes,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ProtoNet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_root) / f"proto_{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_proto.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_steps = 0

        for raw_batch in tqdm(train_loader, desc=f"proto-train-{epoch}", leave=False):
            batch = squeeze_episode(raw_batch, device)
            loss, accuracy = run_episode(model, batch, args.n_way)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy
            total_steps += 1

        avg_train_loss = train_loss / max(total_steps, 1)
        avg_train_acc = train_acc / max(total_steps, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, args.n_way)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "embedding_dim": model.projection.out_features,
                    "n_way": args.n_way,
                    "k_shot": args.k_shot,
                    "q_query": args.q_query,
                    "class_names": train_dataset.class_names,
                },
                best_path,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 4),
                    "train_acc": round(avg_train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                },
                ensure_ascii=False,
            )
        )

    plot_history(history, output_dir / "proto_history.png")
    save_json(
        {
            "backbone": args.backbone,
            "best_val_acc": best_val_acc,
            "history": history,
            "checkpoint": str(best_path),
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "q_query": args.q_query,
        },
        output_dir / "proto_metrics.json",
    )


if __name__ == "__main__":
    main()
