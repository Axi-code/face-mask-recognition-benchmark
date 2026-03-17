from torch import nn
from torchvision import models as tv_models

from models.resnet18 import custom_resnet18


AVAILABLE_MODELS = ("custom_resnet18", "resnet18", "resnet34", "vgg16", "googlenet")


def _build_classifier(in_features, num_classes, classifier_dropout=0.0):
    """
    构建分类头：可选 Dropout + Linear。用于替换预训练模型的最后一层以适配 num_classes。

    Args:
        in_features:  backbone 输出特征维度。
        num_classes:   分类数（如 2 表示二分类）。
        classifier_dropout: 分类层前的 Dropout 概率，0 表示不使用。

    Returns:
        nn.Module: Sequential(Dropout?, Linear) 或单 Linear。
    """
    if classifier_dropout and classifier_dropout > 0:
        return nn.Sequential(nn.Dropout(p=classifier_dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def get_model(model_name, num_classes=2, pretrained=False, classifier_dropout=0.0):
    """
    根据模型名称构建分类模型（支持 custom_resnet18、resnet18、resnet34、vgg16、googlenet）。
    除 custom_resnet18 外均使用 torchvision 预定义结构并替换分类头。

    Args:
        model_name: 模型名称，需在 AVAILABLE_MODELS 中。
        num_classes: 输出类别数。
        pretrained: 是否加载 ImageNet 预训练权重（仅对 torchvision 模型有效）。
        classifier_dropout: 分类头前的 Dropout 概率。

    Returns:
        nn.Module: 未加载 checkpoint 的模型实例。

    Raises:
        ValueError: 不支持的 model_name 时抛出。
    """
    if model_name == "custom_resnet18":
        return custom_resnet18(num_classes=num_classes, classifier_dropout=classifier_dropout)

    if model_name == "resnet18":
        model = tv_models.resnet18(pretrained=pretrained)
        model.fc = _build_classifier(model.fc.in_features, num_classes, classifier_dropout=classifier_dropout)
        return model

    if model_name == "resnet34":
        model = tv_models.resnet34(pretrained=pretrained)
        model.fc = _build_classifier(model.fc.in_features, num_classes, classifier_dropout=classifier_dropout)
        return model

    if model_name == "vgg16":
        model = tv_models.vgg16(pretrained=pretrained)
        model.classifier[6] = _build_classifier(model.classifier[6].in_features, num_classes, classifier_dropout=classifier_dropout)
        return model

    if model_name == "googlenet":
        model = tv_models.googlenet(pretrained=pretrained, aux_logits=False)
        model.fc = _build_classifier(model.fc.in_features, num_classes, classifier_dropout=classifier_dropout)
        return model

    raise ValueError(f"Unsupported model: {model_name}. Available: {', '.join(AVAILABLE_MODELS)}")
