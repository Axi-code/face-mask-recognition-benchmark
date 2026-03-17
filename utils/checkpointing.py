"""
Checkpoint 相关工具：加载权重、从 config 构建推理配置、组装/解析 checkpoint 内容、按 checkpoint 构建模型并汇总摘要信息。
"""
from pathlib import Path

import torch

from models import get_model
from utils.dataset import DEFAULT_MEAN, DEFAULT_STD


DEFAULT_LABEL_MAP = {
    "mask": "已佩戴口罩",
    "no_mask": "未佩戴口罩",
}


def load_checkpoint(weights_path):
    """
    从磁盘加载模型权重文件，兼容「仅 state_dict」与「含 state_dict/config 等字段」的格式。

    Args:
        weights_path: 权重文件路径（.pth 等）。

    Returns:
        dict: 至少包含 "state_dict" 键；若原文件已是 dict 且含 state_dict 则原样返回，否则包装为 {"state_dict": checkpoint}。
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def build_inference_config(config, class_names):
    """
    从训练/应用配置中构建推理用配置（图像尺寸、均值方差、类别名、标签映射、置信度阈值、ROI 等）。

    Args:
        config: 全局配置 dict，通常含 data、inference 等键。
        class_names: 类别名称列表（如 ["mask", "no_mask"]）。

    Returns:
        dict: 包含 image_size, mean, std, class_names, label_map, confidence_threshold, roi_mode, roi_fallback。
    """
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    label_map = {
        class_name: inference_cfg.get("label_map", {}).get(class_name, DEFAULT_LABEL_MAP.get(class_name, class_name))
        for class_name in class_names
    }
    return {
        "image_size": data_cfg.get("image_size", 224),
        "mean": data_cfg.get("mean", DEFAULT_MEAN),
        "std": data_cfg.get("std", DEFAULT_STD),
        "class_names": class_names,
        "label_map": label_map,
        "confidence_threshold": inference_cfg.get("confidence_threshold", 0.65),
        "roi_mode": inference_cfg.get("roi_mode", "face"),
        "roi_fallback": inference_cfg.get("roi_fallback", "smart_crop"),
    }


def checkpoint_payload(model, config, epoch, best_metric, class_names):
    """
    组装保存 checkpoint 时的完整 payload（state_dict、config、epoch、最佳指标、类别名及推理配置）。

    Args:
        model: 当前模型，用于 state_dict。
        config: 训练/数据配置。
        epoch: 当前 epoch 数。
        best_metric: 当前最佳验证指标（如 best_val_acc）。
        class_names: 类别名称列表。

    Returns:
        dict: 可直接用于 torch.save 的字典。
    """
    return {
        "state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_val_acc": best_metric,
        "class_names": class_names,
        "inference": build_inference_config(config, class_names),
    }


def get_inference_config(checkpoint, overrides=None):
    """
    从已加载的 checkpoint 中解析出推理配置，overrides 可覆盖部分字段（如图像尺寸、均值方差等）。

    Args:
        checkpoint: 已加载的 checkpoint 字典（可含 config、inference、class_names 等）。
        overrides: 可选，用于覆盖的键值对，如 {"image_size": 224, "confidence_threshold": 0.7}。

    Returns:
        dict: 推理配置，含 image_size, mean, std, class_names, label_map, confidence_threshold, roi_mode, roi_fallback。
    """
    overrides = overrides or {}
    config = checkpoint.get("config", {})
    checkpoint_inference = checkpoint.get("inference", {})
    config_inference = config.get("inference", {})
    data_cfg = config.get("data", {})
    class_names = checkpoint.get("class_names") or checkpoint_inference.get("class_names") or overrides.get(
        "class_names"
    ) or ["mask", "no_mask"]
    label_map = checkpoint_inference.get("label_map") or config_inference.get("label_map") or {
        class_name: DEFAULT_LABEL_MAP.get(class_name, class_name) for class_name in class_names
    }
    return {
        "image_size": overrides.get("image_size", checkpoint_inference.get("image_size", data_cfg.get("image_size", 224))),
        "mean": overrides.get("mean", checkpoint_inference.get("mean", data_cfg.get("mean", DEFAULT_MEAN))),
        "std": overrides.get("std", checkpoint_inference.get("std", data_cfg.get("std", DEFAULT_STD))),
        "class_names": class_names,
        "label_map": label_map,
        "confidence_threshold": overrides.get(
            "confidence_threshold",
            checkpoint_inference.get("confidence_threshold", config_inference.get("confidence_threshold", 0.65)),
        ),
        "roi_mode": overrides.get("roi_mode", checkpoint_inference.get("roi_mode", config_inference.get("roi_mode", "face"))),
        "roi_fallback": overrides.get(
            "roi_fallback",
            checkpoint_inference.get("roi_fallback", config_inference.get("roi_fallback", "smart_crop")),
        ),
    }


def get_checkpoint_model_name(checkpoint, fallback_model_name):
    """
    从 checkpoint 中解析模型名称（用于 builder.get_model），优先 checkpoint.model_name，其次 config.model.name。

    Args:
        checkpoint: 已加载的 checkpoint。
        fallback_model_name: 当无法从 checkpoint 解析时使用的默认模型名。

    Returns:
        str: 模型名称。
    """
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    return checkpoint.get("model_name") or model_config.get("name") or fallback_model_name


def get_training_conditions(checkpoint):
    """从 checkpoint 的 config 中读取训练条件，用于前端展示（无需重新训练）。"""
    config = checkpoint.get("config", {})
    if not config:
        return None
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    return {
        "pretrained": model_cfg.get("pretrained", False),
        "augment": data_cfg.get("augment", False),
        "use_roi": data_cfg.get("use_roi", False),
        "roi_mode": data_cfg.get("roi_mode") or inference_cfg.get("roi_mode", "face"),
        "roi_fallback": data_cfg.get("roi_fallback") or inference_cfg.get("roi_fallback", "smart_crop"),
    }


def format_training_remark(conditions):
    """
    将训练条件格式化为简短说明，供前端显示（如「预训练、数据增强、ROI(face)」）。

    Args:
        conditions: 由 get_training_conditions 返回的 dict，可为 None。

    Returns:
        str: 简短说明文本，条件未知时返回「训练条件未知」。
    """
    if not conditions:
        return "训练条件未知"
    parts = []
    if conditions.get("pretrained"):
        parts.append("预训练")
    if conditions.get("augment"):
        parts.append("数据增强")
    if conditions.get("use_roi"):
        roi = conditions.get("roi_mode") or "ROI"
        parts.append(f"ROI({roi})")
    if not parts:
        return "未使用预训练/数据增强/ROI"
    return "、".join(parts)


def build_model_from_checkpoint(checkpoint, fallback_model_name, num_classes):
    """
    根据 checkpoint 中的配置与 state_dict 构建模型并加载权重。

    Args:
        checkpoint: 已加载的 checkpoint（含 state_dict、config 等）。
        fallback_model_name: 无法从 checkpoint 解析模型名时使用的默认名。
        num_classes: 分类数（需与 state_dict 中分类头一致）。

    Returns:
        tuple: (model, model_name)，模型已 load_state_dict，未 to(device)。
    """
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    model_name = get_checkpoint_model_name(checkpoint, fallback_model_name)
    pretrained = model_config.get("pretrained", False)
    classifier_dropout = model_config.get("classifier_dropout", 0.0)
    model = get_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        classifier_dropout=classifier_dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_name


def summarize_checkpoint(weights_path):
    """
    加载权重文件并汇总为前端/API 所需的摘要信息（模型名、类别、标签映射、推理参数、训练条件说明等）。

    Args:
        weights_path: 权重文件路径。

    Returns:
        dict: 含 model_name, weights_path, class_names, label_map, image_size, confidence_threshold,
              roi_mode, training_conditions, training_remark 等。
    """
    weights_path = Path(weights_path)
    checkpoint = load_checkpoint(weights_path)
    inference = get_inference_config(checkpoint)
    model_name = get_checkpoint_model_name(checkpoint, "custom_resnet18")
    conditions = get_training_conditions(checkpoint)
    return {
        "model_name": model_name,
        "weights_path": str(weights_path),
        "class_names": inference["class_names"],
        "label_map": inference["label_map"],
        "image_size": inference["image_size"],
        "confidence_threshold": inference["confidence_threshold"],
        "roi_mode": inference["roi_mode"],
        "training_conditions": conditions,
        "training_remark": format_training_remark(conditions),
    }
