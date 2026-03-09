from pathlib import Path
import json

from flask import Flask, render_template
import yaml

from api import MaskPredictor, create_api_blueprint
from utils.checkpointing import summarize_checkpoint


def load_app_config(config_path="configs/baseline.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_model_options(config):
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
        }
    ]


def resolve_default_weights(config):
    model_options = collect_model_options(config)
    recommended_options = [option for option in model_options if option.get("recommended_for_demo")]
    selected = recommended_options[0] if recommended_options else max(
        model_options,
        key=lambda option: option.get("test_accuracy", 0.0),
    )
    return selected["model_name"], selected["weights_path"], model_options


def create_app():
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
