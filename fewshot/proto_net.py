import torch
from torch import nn
from torchvision import models as tv_models


def _load_resnet(model_fn, weights_attr, pretrained):
    """兼容新旧版 torchvision：新版用 weights，旧版用 pretrained。"""
    try:
        weights_enum = getattr(tv_models, weights_attr, None)
        if weights_enum is None:
            raise TypeError("旧版 API")
        w = weights_enum.DEFAULT if pretrained else None
        return model_fn(weights=w)
    except TypeError:
        return model_fn(pretrained=pretrained)


def build_encoder(backbone="resnet18", pretrained=False):
    """
    根据骨干网络名称构建特征编码器（去掉分类头的 CNN）。

    Args:
        backbone: 骨干网络名称，支持 "resnet18"、"resnet34"。
        pretrained: 是否使用 ImageNet 预训练权重。

    Returns:
        tuple: (encoder, feature_dim)，encoder 为 nn.Sequential，feature_dim 为最后一层特征维度。

    Raises:
        ValueError: 不支持的 backbone 时抛出。
    """
    if backbone == "resnet18":
        model = _load_resnet(tv_models.resnet18, "ResNet18_Weights", pretrained)
        feature_dim = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder, feature_dim

    if backbone == "resnet34":
        model = _load_resnet(tv_models.resnet34, "ResNet34_Weights", pretrained)
        feature_dim = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return encoder, feature_dim

    raise ValueError(f"Unsupported few-shot backbone: {backbone}")


class ProtoNet(nn.Module):
    """
    原型网络（Prototypical Network）：用编码器提取特征后投影到 embedding 空间，
    支撑集每类计算一个原型，查询样本通过到原型的距离进行分类。
    """

    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=False):
        """
        初始化原型网络。

        Args:
            backbone: 骨干编码器名称（如 resnet18、resnet34）。
            embedding_dim: 投影后的嵌入维度，用于计算原型与查询距离。
            pretrained: 骨干是否使用预训练权重。
        """
        super().__init__()
        encoder, feature_dim = build_encoder(backbone=backbone, pretrained=pretrained)
        self.encoder = encoder
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, images):
        """
        前向传播：对输入图像提取特征并投影到嵌入空间。

        Args:
            images: 输入图像张量，形状 (N, C, H, W)。

        Returns:
            torch.Tensor: 嵌入向量，形状 (N, embedding_dim)。
        """
        embeddings = self.encoder(images)
        return self.projection(embeddings)


def compute_prototypes(support_embeddings, support_labels, n_way):
    """
    根据支撑集嵌入和标签计算每个类别的原型（该类支撑样本嵌入的均值）。

    Args:
        support_embeddings: 支撑集嵌入，形状 (N_support, embedding_dim)。
        support_labels: 支撑集标签，形状 (N_support,)。
        n_way: 类别数。

    Returns:
        torch.Tensor: 原型张量，形状 (n_way, embedding_dim)。
    """
    prototypes = []
    for class_index in range(n_way):
        class_embeddings = support_embeddings[support_labels == class_index]
        prototypes.append(class_embeddings.mean(dim=0))
    return torch.stack(prototypes)


def prototypical_logits(query_embeddings, prototypes):
    """
    计算查询样本到各原型的负距离作为 logits（距离越小 logit 越大，越属于该类）。

    Args:
        query_embeddings: 查询嵌入，形状 (N_query, embedding_dim)。
        prototypes: 类别原型，形状 (n_way, embedding_dim)。

    Returns:
        torch.Tensor: logits，形状 (N_query, n_way)。
    """
    return -torch.cdist(query_embeddings, prototypes)


def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    """
    计算原型网络的损失与准确率：先算原型与 logits，再对查询集做交叉熵。

    Args:
        support_embeddings: 支撑集嵌入。
        support_labels: 支撑集标签。
        query_embeddings: 查询集嵌入。
        query_labels: 查询集真实标签。
        n_way: 类别数。

    Returns:
        tuple: (loss, accuracy, logits)，loss 为标量，accuracy 为 0~1 浮点数，logits 为 (N_query, n_way)。
    """
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
    logits = prototypical_logits(query_embeddings, prototypes)
    loss = nn.CrossEntropyLoss()(logits, query_labels)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean().item()
    return loss, accuracy, logits
