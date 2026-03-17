import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def count_parameters(model):
    """统计模型可训练参数总数（用于报告模型规模）。"""
    return sum(parameter.numel() for parameter in model.parameters())


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    计算分类混淆矩阵（行=真实类，列=预测类）。

    Args:
        y_true: 真实标签列表或数组。
        y_pred: 预测标签列表或数组。
        num_classes: 类别数量。

    Returns:
        np.ndarray: 形状 (num_classes, num_classes) 的整数矩阵。
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    return confusion


def compute_classification_metrics(y_true, y_pred, class_names):
    """
    根据真实标签与预测标签计算准确率、每类精确率/召回率/F1、宏平均及混淆矩阵。

    Args:
        y_true: 真实标签列表。
        y_pred: 预测标签列表。
        class_names: 类别名称列表，与标签索引对应。

    Returns:
        dict: accuracy, macro_precision, macro_recall, macro_f1, confusion_matrix（列表形式）, per_class（每类指标）。
    """
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
    """
    根据训练历史绘制 loss 与 accuracy 曲线（训练/验证）并保存为图片。

    Args:
        history: dict，需包含 train_loss, val_loss, train_acc, val_acc 四个列表。
        output_path: 输出图片路径（如 .png）。
    """
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
    """
    绘制混淆矩阵热力图并保存，坐标轴为类别名，格子内为数量。

    Args:
        confusion_matrix: 二维数组或列表，形状 (num_classes, num_classes)。
        class_names: 类别名称列表，用于坐标轴标签。
        output_path: 输出图片路径。
    """
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
    """
    将 Python 对象以 UTF-8、缩进 2 的 JSON 格式写入文件，ensure_ascii=False 以正确保存中文。

    Args:
        payload: 可 JSON 序列化的对象（如 dict、list）。
        output_path: 输出文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def measure_inference_latency(model, dataloader, device, warmup_steps=5, measure_steps=20):
    """
    在给定 DataLoader 上测量模型推理延迟：先 warmup 若干步，再统计后续步的每样本平均耗时（毫秒）。

    Args:
        model: 已加载的模型，会设为 eval 模式。
        dataloader: 提供 (images, labels) 的数据加载器。
        device: 推理设备（如 cuda/cpu）。
        warmup_steps: 预热步数，不计入统计。
        measure_steps: 正式计时的步数。

    Returns:
        float: 每张图片平均推理时间（毫秒）。
    """
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
