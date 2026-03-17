from flask import Blueprint, jsonify, request


def error_response(message, status_code=400, detail=""):
    """
    构造 API 错误响应的 JSON 与状态码，供 Flask 路由返回。

    Args:
        message: 面向用户的错误提示（如「请先选择图片后再识别」）。
        status_code: HTTP 状态码，默认 400。
        detail: 可选的技术细节（如异常信息），便于调试。

    Returns:
        tuple: (Response, status_code)，可直接 return 给 Flask。
    """
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
    """
    创建 Flask Blueprint，注册 /health、/models、/predict、/analyze-truth 等路由，
    将请求委托给 predictor（MaskPredictor）完成推理与真值分析。

    Args:
        predictor: MaskPredictor 实例，提供 health()、predict_bytes()、analyze_truth() 等方法。
        model_options: 可选，模型选项列表（用于 /models 返回）；若为 None 则传空列表。

    Returns:
        Blueprint: 已注册路由的 API 蓝图。
    """
    api = Blueprint("api", __name__)
    model_options = model_options or []

    @api.get("/health")
    def health():
        """返回预测器健康状态与已加载模型信息。"""
        return jsonify(predictor.health())

    @api.get("/models")
    def models():
        """返回当前可用的模型选项列表。"""
        return jsonify({"status": "ok", "models": model_options})

    @api.post("/predict")
    def predict():
        """接收上传的图片，调用 predictor 进行识别，返回多模型预测结果。若未上传图片则返回错误。"""
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
        """接收真实标签与多模型预测结果，返回真值分析（正确率、高置信度错判等）。"""
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
