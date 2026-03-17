"""
评估与单图推理入口：加载 checkpoint、在测试集上计算分类指标，或对单张图片进行预测并输出 JSON。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from models import AVAILABLE_MODELS
from utils.checkpointing import build_model_from_checkpoint, get_inference_config, load_checkpoint
from utils.dataset import DEFAULT_MEAN, DEFAULT_STD, build_inference_transform, create_classification_dataloaders
from utils.inference import build_prediction_payload
from utils.metrics import compute_classification_metrics, plot_confusion_matrix, save_json
from utils.roi import RegionExtractor


def parse_args():
    """解析命令行参数：模型名、权重路径、数据根目录、批大小、图像尺寸、单图路径、ROI/置信度等。"""
    parser = argparse.ArgumentParser(description="Evaluate classification model or infer a single image.")
    parser.add_argument("--model", type=str, default="custom_resnet18", choices=AVAILABLE_MODELS)
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--mean", type=float, nargs=3, default=DEFAULT_MEAN)
    parser.add_argument("--std", type=float, nargs=3, default=DEFAULT_STD)
    parser.add_argument("--roi-mode", type=str, default="")
    parser.add_argument("--roi-fallback", type=str, default="")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--report-dir", type=str, default="")
    return parser.parse_args()


def evaluate_dataset(model, dataloader, device, class_names):
    """
    在测试集 DataLoader 上评估模型，收集所有预测与真实标签后计算分类指标（准确率、F1、混淆矩阵等）。

    Args:
        model: 已加载的分类模型。
        dataloader: 测试集 DataLoader。
        device: 设备。
        class_names: 类别名称列表。

    Returns:
        dict: compute_classification_metrics 的返回值（accuracy、macro_f1、confusion_matrix、per_class 等）。
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return compute_classification_metrics(y_true, y_pred, class_names)


def resolve_overrides(args):
    """
    将命令行参数中与默认值不同的项转为 get_inference_config 的 overrides 字典
    （如图像尺寸、均值方差、ROI、置信度阈值等）。
    """
    overrides = {}
    if args.image_size != 224:
        overrides["image_size"] = args.image_size
    if args.mean != DEFAULT_MEAN:
        overrides["mean"] = args.mean
    if args.std != DEFAULT_STD:
        overrides["std"] = args.std
    if args.roi_mode:
        overrides["roi_mode"] = args.roi_mode
    if args.roi_fallback:
        overrides["roi_fallback"] = args.roi_fallback
    if args.confidence_threshold is not None:
        overrides["confidence_threshold"] = args.confidence_threshold
    return overrides


def predict_single_image(model, image_path, inference_config, model_name, weights_path, device):
    """
    对单张图片进行 ROI 提取、变换、推理，并组装为 build_prediction_payload 结果后打印 JSON。
    若指定 report_dir 则可将结果保存为 single_image_prediction.json。

    Args:
        model: 已加载的模型。
        image_path: 图片路径。
        inference_config: 推理配置（含 roi_mode、roi_fallback、image_size、mean、std、class_names、label_map、confidence_threshold）。
        model_name: 模型名称。
        weights_path: 权重路径（字符串）。
        device: 设备。

    Returns:
        dict: 单图预测 payload。
    """
    extractor = RegionExtractor(
        mode=inference_config["roi_mode"],
        fallback_mode=inference_config["roi_fallback"],
    )
    transform = build_inference_transform(
        inference_config["image_size"],
        inference_config["mean"],
        inference_config["std"],
    )
    image = Image.open(image_path).convert("RGB")
    roi_image, roi_info = extractor.extract(image)
    image_tensor = transform(roi_image).unsqueeze(0).to(device)

    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = build_prediction_payload(
        probabilities=probabilities,
        class_names=inference_config["class_names"],
        label_map=inference_config["label_map"],
        model_name=model_name,
        weights_path=str(weights_path),
        confidence_threshold=inference_config["confidence_threshold"],
        roi_info=roi_info,
        device=device,
        input_source="single_image",
        extra_meta={"image_path": str(image_path), "inference_time_ms": inference_time_ms},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    """主入口：加载 checkpoint、构建 DataLoader 与模型；若指定 --image 则对单图预测并输出 JSON，否则在测试集上评估并输出指标与可选报告。"""
    args = parse_args()
    checkpoint = load_checkpoint(args.weights)
    inference_config = get_inference_config(checkpoint, overrides=resolve_overrides(args))
    class_names = inference_config["class_names"]

    dataloaders = create_classification_dataloaders(
        data_dir=args.data_root,
        image_size=inference_config["image_size"],
        batch_size=args.batch_size,
        mean=inference_config["mean"],
        std=inference_config["std"],
        augment=False,
        val_ratio=0.2,
        use_roi=True,
        roi_mode=inference_config["roi_mode"],
        roi_fallback=inference_config["roi_fallback"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = build_model_from_checkpoint(checkpoint, args.model, num_classes=len(class_names))
    model = model.to(device)

    report_dir = Path(args.report_dir) if args.report_dir else None
    if args.image:
        result = predict_single_image(model, args.image, inference_config, model_name, args.weights, device)
        if report_dir:
            save_json(result, report_dir / "single_image_prediction.json")
        return

    metrics = evaluate_dataset(model, dataloaders["test"], device, class_names)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if report_dir:
        save_json(metrics, report_dir / "evaluation_metrics.json")
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, report_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
