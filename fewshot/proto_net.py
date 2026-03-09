import torch
from torch import nn
from torchvision import models as tv_models


def build_encoder(backbone="resnet18", pretrained=False):
    if backbone == "resnet18":
        model = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        feature_dim = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder, feature_dim

    if backbone == "resnet34":
        model = tv_models.resnet34(
            weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        feature_dim = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder, feature_dim

    raise ValueError(f"Unsupported few-shot backbone: {backbone}")


class ProtoNet(nn.Module):
    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=False):
        super().__init__()
        encoder, feature_dim = build_encoder(backbone=backbone, pretrained=pretrained)
        self.encoder = encoder
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, images):
        embeddings = self.encoder(images)
        return self.projection(embeddings)


def compute_prototypes(support_embeddings, support_labels, n_way):
    prototypes = []
    for class_index in range(n_way):
        class_embeddings = support_embeddings[support_labels == class_index]
        prototypes.append(class_embeddings.mean(dim=0))
    return torch.stack(prototypes)


def prototypical_logits(query_embeddings, prototypes):
    return -torch.cdist(query_embeddings, prototypes)


def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
    logits = prototypical_logits(query_embeddings, prototypes)
    loss = nn.CrossEntropyLoss()(logits, query_labels)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean().item()
    return loss, accuracy, logits
