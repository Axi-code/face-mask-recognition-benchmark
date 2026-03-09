from flask import Blueprint, jsonify, request


def error_response(message, status_code=400, detail=""):
    return (
        jsonify(
            {
                "status": "error",
                "error_code": "request_failed",
                "error_message": detail or message,
                "user_message": message,
            }
        ),
        status_code,
    )


def create_api_blueprint(predictor, model_options=None):
    api = Blueprint("api", __name__)
    model_options = model_options or []

    @api.get("/health")
    def health():
        return jsonify(predictor.health())

    @api.get("/models")
    def models():
        return jsonify({"status": "ok", "models": model_options})

    @api.post("/predict")
    def predict():
        if "image" not in request.files:
            return error_response("请先选择图片后再识别。", detail="Missing image file.")

        image_file = request.files["image"]
        image_bytes = image_file.read()
        input_source = request.form.get("input_source", "upload")
        try:
            result = predictor.predict_bytes(image_bytes, input_source=input_source)
        except Exception as exc:
            return error_response("识别失败，请检查模型或图片格式。", detail=str(exc))
        return jsonify(result)

    @api.post("/analyze-truth")
    def analyze_truth():
        payload = request.get_json(silent=True) or {}
        truth_class_name = payload.get("truth_class_name")
        predictions = payload.get("predictions")
        if not truth_class_name:
            return error_response("请先填写真实标签。", detail="truth_class_name is required.")
        if not isinstance(predictions, list) or not predictions:
            return error_response("缺少预测结果，无法进行真值分析。", detail="predictions must be a non-empty list.")

        try:
            result = predictor.analyze_truth(predictions, truth_class_name)
        except Exception as exc:
            return error_response("真实标签分析失败，请检查输入内容。", detail=str(exc))
        return jsonify(result)

    return api
