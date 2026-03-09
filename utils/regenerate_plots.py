"""
从已有的 metrics.json 重新生成 history.png 和 confusion_matrix.png。
适用于训练已完成但图片丢失的情况。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也可运行
import matplotlib.pyplot as plt
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description="从 metrics.json 重新生成训练曲线和混淆矩阵")
    parser.add_argument(
        "metrics_path",
        type=str,
        nargs="?",
        default=None,
        help="metrics.json 路径，不指定则自动查找 results 下最新实验",
    )
    parser.add_argument(
        "--copy-to-docs",
        action="store_true",
        help="同时复制到 docs/images/ 供 README 使用",
    )
    args = parser.parse_args()

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        if not metrics_path.exists():
            raise FileNotFoundError(f"找不到文件: {metrics_path}")
    else:
        results_dir = Path("results")
        if not results_dir.exists():
            raise FileNotFoundError("results 目录不存在")
        json_files = list(results_dir.rglob("metrics.json"))
        if not json_files:
            raise FileNotFoundError("results 下没有找到 metrics.json")
        metrics_path = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"使用最新实验: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    experiment_dir = metrics_path.parent
    history = data.get("history")
    if not history:
        raise ValueError("metrics.json 中缺少 history 字段")

    history_png = experiment_dir / "history.png"
    confusion_png = experiment_dir / "confusion_matrix.png"

    plot_training_curves(history, history_png)
    print(f"已生成: {history_png}")

    test_metrics = data.get("test_metrics")
    if test_metrics and "confusion_matrix" in test_metrics:
        class_names = data.get("class_names", ["mask", "no_mask"])
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names,
            confusion_png,
        )
        print(f"已生成: {confusion_png}")

    if args.copy_to_docs:
        docs_images = Path("docs/images")
        docs_images.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(history_png, docs_images / "train-history.png")
        if confusion_png.exists():
            shutil.copy(confusion_png, docs_images / "confusion-matrix.png")
        print(f"已复制到 {docs_images}/")


if __name__ == "__main__":
    main()
