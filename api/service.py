from io import BytesIO
from pathlib import Path
import time

import torch
from PIL import Image

from utils.checkpointing import build_model_from_checkpoint, get_inference_config, load_checkpoint
from utils.dataset import build_inference_transform
from utils.inference import analyze_truth_from_predictions, build_multi_model_payload, build_prediction_payload
from utils.roi import RegionExtractor

class MaskPredictor:
    """
    口罩分类推理核心：管理多个已加载模型，对上传图片进行 ROI 预处理与推理，
    支持单模型预测与多模型对比、真值分析等，供 Web API 与评估脚本调用。
    """

    def __init__(
        self,
        default_model="custom_resnet18",
        default_weights="best_model.pth",
        image_size=224,
        mean=None,
        std=None,
        model_options=None,
    ):
        """
        初始化预测器：设置设备、默认模型与权重路径；若提供 model_options 则注册并加载，否则若默认权重存在则仅注册该权重。

        Args:
            default_model: 默认模型名称（如 custom_resnet18）。
            default_weights: 默认权重文件路径。
            image_size: 默认输入图像尺寸（当 checkpoint 无配置时使用）。
            mean: 默认归一化均值（同上）。
            std: 默认归一化标准差（同上）。
            model_options: 可选，模型选项列表，每项含 model_name、weights_path、display_name 等。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_image_size = image_size
        self.default_mean = mean
        self.default_std = std
        self.loaded_models = {}
        self.model_options = []
        self.class_names = ["mask", "no_mask"]
        self.label_map = {class_name: class_name for class_name in self.class_names}
        self.default_model = default_model
        self.default_weights = str(Path(default_weights))

        if model_options:
            self.register_model_options(model_options)
        elif Path(default_weights).exists():
            self.register_model_options(
                [
                    {
                        "model_name": default_model,
                        "weights_path": str(Path(default_weights)),
                        "display_name": default_model,
                    }
                ]
            )

    def _load_model_entry(self, option):
        """
        根据单条模型选项加载 checkpoint、构建模型并封装为「条目」dict（含 model、推理配置、ROI 提取器等），供后续预测使用。
        """
        model_name = option["model_name"]
        weights_path = option["weights_path"]
        checkpoint = load_checkpoint(weights_path)
        overrides = {}
        if self.default_image_size is not None and "inference" not in checkpoint and "config" not in checkpoint:
            overrides["image_size"] = self.default_image_size
        if self.default_mean is not None and self.default_std is not None and "inference" not in checkpoint and "config" not in checkpoint:
            overrides["mean"] = self.default_mean
            overrides["std"] = self.default_std
        inference_config = get_inference_config(checkpoint, overrides=overrides)
        model, resolved_model_name = build_model_from_checkpoint(checkpoint, model_name, len(inference_config["class_names"]))
        model = model.to(self.device)
        model.eval()

        entry = {
            "model": model,
            "model_name": resolved_model_name,
            "display_name": option.get("display_name") or resolved_model_name,
            "weights_path": str(Path(weights_path)),
            "class_names": inference_config["class_names"],
            "label_map": inference_config["label_map"],
            "image_size": inference_config["image_size"],
            "mean": inference_config["mean"],
            "std": inference_config["std"],
            "confidence_threshold": inference_config["confidence_threshold"],
            "roi_mode": inference_config["roi_mode"],
            "roi_fallback": inference_config["roi_fallback"],
            "region_extractor": RegionExtractor(
                mode=inference_config["roi_mode"],
                fallback_mode=inference_config["roi_fallback"],
            ),
            "offline_metrics": {
                "best_val_accuracy": option.get("best_val_accuracy"),
                "test_accuracy": option.get("test_accuracy"),
                "macro_f1": option.get("macro_f1"),
                "avg_inference_latency_ms": option.get("avg_inference_latency_ms"),
                "parameter_count": option.get("parameter_count"),
            },
            "source": option.get("source", "results"),
            "kd_role": option.get("kd_role"),
            "training_remark": option.get("training_remark"),
        }
        return entry

    def register_model_options(self, model_options):
        """
        注册并加载一批模型选项：去重权重路径、逐个加载，更新 loaded_models 与 model_options，
        并同步 class_names/label_map 为第一个成功加载的模型。失败或路径不存在则跳过。

        Args:
            model_options: 模型选项列表，每项需含 weights_path，以及 model_name、display_name 等。

        Returns:
            dict: 含 status（ok/not_ready）、loaded_model_count、models（供前端展示的选项列表）。
        """
        self.loaded_models = {}
        self.model_options = []
        seen_paths = set()

        for option in model_options:
            weights_path = str(Path(option["weights_path"]))
            if not Path(weights_path).exists():
                continue
            resolved_path = str(Path(weights_path).resolve())
            if resolved_path in seen_paths:
                continue
            try:
                entry = self._load_model_entry({**option, "weights_path": weights_path})
            except Exception:
                continue
            self.loaded_models[resolved_path] = entry
            om = entry["offline_metrics"]
            self.model_options.append(
                {
                    "model_name": entry["model_name"],
                    "display_name": entry["display_name"],
                    "weights_path": entry["weights_path"],
                    "class_names": entry["class_names"],
                    "label_map": entry["label_map"],
                    "image_size": entry["image_size"],
                    "confidence_threshold": entry["confidence_threshold"],
                    "roi_mode": entry["roi_mode"],
                    "roi_fallback": entry["roi_fallback"],
                    "offline_metrics": om,
                    "source": entry["source"],
                    "kd_role": option.get("kd_role"),
                    "training_remark": entry.get("training_remark"),
                    "best_val_accuracy": om.get("best_val_accuracy"),
                    "test_accuracy": om.get("test_accuracy"),
                    "macro_f1": om.get("macro_f1"),
                    "avg_inference_latency_ms": om.get("avg_inference_latency_ms"),
                    "parameter_count": om.get("parameter_count"),
                }
            )
            seen_paths.add(resolved_path)

        if self.model_options:
            self.class_names = self.model_options[0]["class_names"]
            self.label_map = self.model_options[0]["label_map"]

        return {
            "status": "ok" if self.loaded_models else "not_ready",
            "loaded_model_count": len(self.loaded_models),
            "models": self.model_options,
        }

    def _preprocess_pil(self, image, entry):
        """
        对 PIL 图像做 RGB 转换、ROI 提取与推理变换，得到送入模型的 tensor 及 ROI 信息。
        """
        image = image.convert("RGB")
        roi_image, roi_info = entry["region_extractor"].extract(image)
        transform = build_inference_transform(entry["image_size"], entry["mean"], entry["std"])
        image_tensor = transform(roi_image).unsqueeze(0).to(self.device)
        return image_tensor, roi_info

    def _predict_with_entry(self, image, entry, input_source="upload"):
        """
        使用单个已加载的模型条目对图像进行推理，并组装为 build_prediction_payload 所需的单模型结果。
        """
        image_tensor, roi_info = self._preprocess_pil(image, entry)
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = entry["model"](image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return build_prediction_payload(
            probabilities=probabilities,
            class_names=entry["class_names"],
            label_map=entry["label_map"],
            model_name=entry["model_name"],
            weights_path=entry["weights_path"],
            confidence_threshold=entry["confidence_threshold"],
            roi_info=roi_info,
            device=self.device,
            input_source=input_source,
            display_name=entry["display_name"],
            offline_metrics=entry["offline_metrics"],
            training_remark=entry.get("training_remark"),
            extra_meta={
                "inference_time_ms": inference_time_ms,
                "roi_mode": entry["roi_mode"],
                "image_size": entry["image_size"],
                "confidence_threshold": entry["confidence_threshold"],
            },
        )

    def predict_pil(self, image, input_source="upload"):
        """
        对 PIL 图像使用所有已加载模型进行预测，并汇总为多模型 payload（含多数投票、一致性等）。
        若未加载任何模型则抛出 RuntimeError。
        """
        if not self.loaded_models:
            raise RuntimeError("Model is not loaded. Please train at least one model first.")
        predictions = [
            self._predict_with_entry(image.copy(), entry, input_source=input_source)
            for entry in self.loaded_models.values()
        ]
        return build_multi_model_payload(predictions, input_source=input_source)

    def predict_bytes(self, image_bytes, input_source="upload"):
        """将字节流解码为 PIL 图像后调用 predict_pil，供 API 上传接口使用。"""
        image = Image.open(BytesIO(image_bytes))
        return self.predict_pil(image, input_source=input_source)

    def analyze_truth(self, predictions, truth_class_name):
        """在已知真实标签下分析多模型预测结果（正确率、高置信度错判等），委托 utils.inference.analyze_truth_from_predictions。"""
        return analyze_truth_from_predictions(predictions, truth_class_name)

    def health(self):
        """返回当前预测器状态：是否就绪、类别名、标签映射、设备、已加载模型数量与选项等，供 /health 接口使用。"""
        return {
            "status": "ok" if self.loaded_models else "not_ready",
            "class_names": self.class_names,
            "label_map": self.label_map,
            "device": str(self.device),
            "loaded_model_count": len(self.loaded_models),
            "models": self.model_options,
            "default_model": self.default_model,
            "default_weights": self.default_weights,
        }
