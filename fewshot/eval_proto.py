import argparse
import json

import torch
from torch.utils.data import DataLoader

from fewshot.episode_dataset import EpisodeDataset
from fewshot.proto_net import ProtoNet, prototypical_loss
from utils.dataset import DEFAULT_MEAN, DEFAULT_STD, build_inference_transform


def parse_args():
    """解析命令行参数：数据路径、checkpoint 路径、n_way/k_shot/q_query、评估 episode 数等。"""
    parser = argparse.ArgumentParser(description="Evaluate a trained ProtoNet on new episodes.")
    parser.add_argument("--data-root", type=str, default="data/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-way", type=int, default=2)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    """主入口：加载训练好的 ProtoNet checkpoint，在指定数据上跑若干 episode 并输出平均 loss 与准确率。"""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ProtoNet(
        backbone=checkpoint.get("backbone", "resnet18"),
        embedding_dim=checkpoint.get("embedding_dim", 256),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = EpisodeDataset(
        root_dir=args.data_root,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        transform=build_inference_transform(args.image_size, DEFAULT_MEAN, DEFAULT_STD),
        episodes_per_epoch=args.episodes,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            support_images = batch["support_images"].squeeze(0).to(device)
            support_labels = batch["support_labels"].squeeze(0).to(device)
            query_images = batch["query_images"].squeeze(0).to(device)
            query_labels = batch["query_labels"].squeeze(0).to(device)

            support_embeddings = model(support_images)
            query_embeddings = model(query_images)
            loss, accuracy, _ = prototypical_loss(
                support_embeddings, support_labels, query_embeddings, query_labels, args.n_way
            )
            total_loss += loss.item()
            total_acc += accuracy
            total_steps += 1

    result = {
        "avg_episode_loss": total_loss / max(total_steps, 1),
        "avg_episode_acc": total_acc / max(total_steps, 1),
        "episodes": total_steps,
        "checkpoint": args.checkpoint,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
