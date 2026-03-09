import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    return confusion


def compute_classification_metrics(y_true, y_pred, class_names):
    confusion = compute_confusion_matrix(y_true, y_pred, len(class_names))
    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total else 0.0

    per_class = {}
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for index, class_name in enumerate(class_names):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        support = confusion[index, :].sum()

        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0
        recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": int(support),
        }
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "macro_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def plot_training_curves(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confusion = np.asarray(confusion_matrix)
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            plt.text(col, row, str(confusion[row, col]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def measure_inference_latency(model, dataloader, device, warmup_steps=5, measure_steps=20):
    model.eval()
    timings = []
    step_count = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - start

            if step_count >= warmup_steps:
                timings.append(duration / max(images.size(0), 1))
            step_count += 1
            if step_count >= warmup_steps + measure_steps:
                break

    if not timings:
        return 0.0
    return float(np.mean(timings) * 1000.0)
