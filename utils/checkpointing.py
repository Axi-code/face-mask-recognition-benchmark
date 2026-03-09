from pathlib import Path

import torch

from models import get_model
from utils.dataset import DEFAULT_MEAN, DEFAULT_STD


DEFAULT_LABEL_MAP = {
    "mask": "已佩戴口罩",
    "no_mask": "未佩戴口罩",
}


def load_checkpoint(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def build_inference_config(config, class_names):
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
    return {
        "state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_val_acc": best_metric,
        "class_names": class_names,
        "inference": build_inference_config(config, class_names),
    }


def get_inference_config(checkpoint, overrides=None):
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
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    return checkpoint.get("model_name") or model_config.get("name") or fallback_model_name


def build_model_from_checkpoint(checkpoint, fallback_model_name, num_classes):
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
    weights_path = Path(weights_path)
    checkpoint = load_checkpoint(weights_path)
    inference = get_inference_config(checkpoint)
    model_name = get_checkpoint_model_name(checkpoint, "custom_resnet18")
    return {
        "model_name": model_name,
        "weights_path": str(weights_path),
        "class_names": inference["class_names"],
        "label_map": inference["label_map"],
        "image_size": inference["image_size"],
        "confidence_threshold": inference["confidence_threshold"],
        "roi_mode": inference["roi_mode"],
    }
