"""
模型模块：提供 get_model 工厂、AVAILABLE_MODELS 及自定义 ResNet18（Residual、ResNet18、custom_resnet18）。
"""
from .builder import AVAILABLE_MODELS, get_model
from .resnet18 import ResNet18, Residual, custom_resnet18

__all__ = ["AVAILABLE_MODELS", "ResNet18", "Residual", "custom_resnet18", "get_model"]
