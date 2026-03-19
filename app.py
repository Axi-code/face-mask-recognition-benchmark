from pathlib import Path
import json

from flask import Flask, render_template
import yaml

from api import MaskPredictor, create_api_blueprint
from utils.checkpointing import DEFAULT_LABEL_MAP, summarize_checkpoint


def load_app_config(config_path="configs/baseline.yaml"):
    """
    从 YAML 文件加载应用配置（模型、数据、推理等）。

    Args:
        config_path: 配置文件路径，默认 configs/baseline.yaml。

    Returns:
        dict: 解析后的配置字典。
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _summarize_protonet_checkpoint(weights_path, proto_data):
    """从 ProtoNet checkpoint 与 proto_metrics 汇总为模型选项格式。"""
    import torch
    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint.get("class_names") or proto_data.get("class_names") or ["mask", "no_mask"]
    label_map = {cn: DEFAULT_LABEL_MAP.get(cn, cn) for cn in class_names}
    return {
        "model_name": "protonet",
        "weights_path": str(Path(weights_path)),
        "class_names": class_names,
        "label_map": label_map,
        "image_size": 224,
        "confidence_threshold": 0.65,
        "roi_mode": "face",
        "roi_fallback": "smart_crop",
        "backbone": checkpoint.get("backbone") or proto_data.get("backbone", "resnet18"),
        "embedding_dim": checkpoint.get("embedding_dim") or proto_data.get("embedding_dim", 256),
        "n_way": proto_data.get("n_way", 2),
        "k_shot": proto_data.get("k_shot", 1),
    }


def collect_model_options(config):
    """
    从项目根目录的 best_model.pth 与 results 下各实验的 metrics.json 收集可用模型选项
    （权重路径、类别名、显示名、离线指标等），用于前端模型选择与 predictor 注册。
    若没有任何可用权重，则返回基于 config 的「待训练」占位选项。

    Args:
        config: 应用配置（含 model、data、inference 等）。

    Returns:
        list: 模型选项列表，每项含 model_name、weights_path、display_name、class_names、label_map 等。
    """
    results_root = Path("results")
    options = []
    seen_paths = set()
    default_checkpoint = Path("best_model.pth")

    if default_checkpoint.exists():
        try:
            summary = summarize_checkpoint(default_checkpoint)
            summary["source"] = "project_root"
            summary["display_name"] = f"{summary['model_name']}（根目录权重）"
            summary["best_val_accuracy"] = None
            summary["test_accuracy"] = None
            summary["macro_f1"] = None
            summary["avg_inference_latency_ms"] = None
            summary["parameter_count"] = None
            options.append(summary)
            seen_paths.add(default_checkpoint.resolve())
        except Exception:
            pass

    for metrics_path in sorted(results_root.glob("**/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        weights_path = metrics.get("best_checkpoint")
        if not weights_path:
            continue
        checkpoint_path = Path(weights_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            continue
        resolved_path = checkpoint_path.resolve()
        if resolved_path in seen_paths:
            continue
        option = summarize_checkpoint(checkpoint_path)
        option["source"] = "results"
        option["best_val_accuracy"] = metrics.get("best_val_accuracy")
        option["test_accuracy"] = metrics.get("test_metrics", {}).get("accuracy", 0.0)
        option["macro_f1"] = metrics.get("test_metrics", {}).get("macro_f1", 0.0)
        option["avg_inference_latency_ms"] = metrics.get("avg_inference_latency_ms")
        option["parameter_count"] = metrics.get("parameter_count")
        option["recommended_for_demo"] = metrics.get("recommended_for_demo", False)
        option["display_name"] = f"{option['model_name']} | test_acc={option['test_accuracy']:.3f}"
        options.append(option)
        seen_paths.add(resolved_path)

    for kd_metrics_path in sorted(results_root.glob("**/kd_results.json")):
        try:
            kd_data = json.loads(kd_metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        student_checkpoint = kd_data.get("checkpoint")
        if not student_checkpoint:
            continue
        checkpoint_path = Path(student_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            continue
        resolved_path = checkpoint_path.resolve()
        if resolved_path in seen_paths:
            continue
        try:
            option = summarize_checkpoint(checkpoint_path)
        except Exception:
            continue
        student_info = kd_data.get("student", {})
        teacher_info = kd_data.get("teacher", {})
        kd_settings = kd_data.get("kd_settings", {})
        option["source"] = "kd"
        option["best_val_accuracy"] = student_info.get("best_val_acc")
        option["test_accuracy"] = kd_data.get("test_metrics", {}).get("accuracy", 0.0)
        option["macro_f1"] = kd_data.get("test_metrics", {}).get("macro_f1", 0.0)
        option["avg_inference_latency_ms"] = student_info.get("latency_ms")
        option["parameter_count"] = student_info.get("params_M")
        option["recommended_for_demo"] = False
        option["kd_role"] = "kd_student"
        option["kd_teacher"] = teacher_info.get("name", "unknown")
        T = kd_settings.get("temperature", "?")
        alpha = kd_settings.get("alpha", "?")
        option["display_name"] = (
            f"{option['model_name']} (KD) | test_acc={option['test_accuracy']:.3f}"
        )
        option["training_remark"] = (
            f"KD: Teacher={teacher_info.get('name','?')}, "
            f"T={T}, \u03b1={alpha}"
        )
        options.append(option)
        seen_paths.add(resolved_path)

    for proto_metrics_path in sorted(results_root.glob("**/proto_metrics.json")):
        try:
            proto_data = json.loads(proto_metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        checkpoint_path = proto_data.get("checkpoint")
        if not checkpoint_path:
            continue
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            continue
        resolved_path = checkpoint_path.resolve()
        if resolved_path in seen_paths:
            continue
        try:
            option = _summarize_protonet_checkpoint(checkpoint_path, proto_data)
        except Exception:
            continue
        option["source"] = "fewshot"
        option["is_protonet"] = True
        option["support_root"] = str(Path("data/train"))
        option["best_val_accuracy"] = proto_data.get("best_val_acc")
        option["test_accuracy"] = proto_data.get("best_val_acc")
        acc = option.get("best_val_accuracy")
        option["display_name"] = f"ProtoNet（少样本）| val_acc={acc:.3f}" if acc is not None else "ProtoNet（少样本）"
        option["training_remark"] = f"少样本 ProtoNet, {proto_data.get('n_way', 2)}-way {proto_data.get('k_shot', 1)}-shot"
        options.append(option)
        seen_paths.add(resolved_path)

    for opt in options:
        if "kd_role" not in opt:
            opt["kd_role"] = None
        if "is_protonet" not in opt:
            opt["is_protonet"] = False

    if options:
        return options

    config_model_name = config["model"]["name"]
    return [
        {
            "model_name": config_model_name,
            "weights_path": str(default_checkpoint),
            "display_name": f"{config_model_name}（等待训练权重）",
            "class_names": ["mask", "no_mask"],
            "label_map": {"mask": "已佩戴口罩", "no_mask": "未佩戴口罩"},
            "image_size": config["data"].get("image_size", 224),
            "confidence_threshold": config.get("inference", {}).get("confidence_threshold", 0.7),
            "roi_mode": config.get("inference", {}).get("roi_mode", "face"),
            "source": "config",
            "best_val_accuracy": None,
            "test_accuracy": 0.0,
            "macro_f1": 0.0,
            "avg_inference_latency_ms": None,
            "parameter_count": None,
            "recommended_for_demo": False,
            "training_conditions": None,
            "training_remark": "待训练",
        }
    ]


def resolve_default_weights(config):
    """
    从配置与已有模型选项中解析默认使用的模型名、权重路径及完整选项列表。
    优先使用标记为 recommended_for_demo 的选项，否则选 test_accuracy 最高的。

    Args:
        config: 应用配置。

    Returns:
        tuple: (default_model_name, default_weights_path, model_options_list)。
    """
    model_options = collect_model_options(config)
    recommended_options = [option for option in model_options if option.get("recommended_for_demo")]
    selected = recommended_options[0] if recommended_options else max(
        model_options,
        key=lambda option: option.get("test_accuracy", 0.0),
    )
    return selected["model_name"], selected["weights_path"], model_options


def create_app():
    """
    创建 Flask 应用：加载配置、构建 MaskPredictor、注册 API 蓝图、绑定首页路由，返回 app 实例。
    """
    config = load_app_config()
    app = Flask(__name__)

    default_model_name, default_weights, model_options = resolve_default_weights(config)
    predictor = MaskPredictor(
        default_model=default_model_name,
        default_weights=default_weights,
        image_size=config["data"].get("image_size", 224),
        mean=config["data"].get("mean"),
        std=config["data"].get("std"),
        model_options=model_options,
    )

    app.register_blueprint(create_api_blueprint(predictor, predictor.model_options))

    @app.get("/")
    def index():
        """首页路由：渲染 index.html，传入模型目录、类别名、标签映射与健康状态供前端展示。"""
        health = predictor.health()
        return render_template(
            "index.html",
            model_catalog=predictor.model_options,
            class_names=health.get("class_names", []),
            label_map=health.get("label_map", {}),
            health=health,
        )

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=True)
