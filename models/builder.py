from torch import nn
from torchvision import models as tv_models

from models.resnet18 import custom_resnet18


AVAILABLE_MODELS = ("custom_resnet18", "resnet18", "resnet34", "vgg16", "googlenet")


def _build_classifier(in_features, num_classes, classifier_dropout=0.0):
    if classifier_dropout and classifier_dropout > 0:
        return nn.Sequential(nn.Dropout(p=classifier_dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def get_model(model_name, num_classes=2, pretrained=False, classifier_dropout=0.0):
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
