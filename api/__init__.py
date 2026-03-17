"""
API 模块：提供 MaskPredictor 推理服务与 Flask API 蓝图（/health、/models、/predict、/analyze-truth）。
"""
from .infer import create_api_blueprint
from .service import MaskPredictor

__all__ = ["MaskPredictor", "create_api_blueprint"]
