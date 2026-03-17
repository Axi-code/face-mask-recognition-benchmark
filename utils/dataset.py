from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.roi import RegionExtractor


DEFAULT_MEAN = [0.1726, 0.1515, 0.1427]
DEFAULT_STD = [0.0736, 0.0622, 0.0593]


class TransformSubset(Dataset):
    """
    对现有数据集（如 ImageFolder）的子集索引施加变换与可选的 ROI 提取，
    用于训练/验证/测试时统一 resize、归一化及 ROI 预处理。
    """

    def __init__(self, dataset, indices, transform=None, region_extractor=None):
        """
        初始化子集包装器。

        Args:
            dataset: 原始数据集（需有 .samples 与 .loader，如 ImageFolder）。
            indices: 要使用的样本索引列表（如 train_indices、val_indices）。
            transform: 可选，对单张 PIL 图像施加的变换（如 Compose([Resize, ToTensor, Normalize])）。
            region_extractor: 可选，RegionExtractor 实例；若提供则先对图像做 ROI 再 transform。
        """
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.region_extractor = region_extractor

    def __len__(self):
        """返回子集样本数量。"""
        return len(self.indices)

    def __getitem__(self, idx):
        """根据索引加载图片，可选 ROI 与 transform，返回 (image_tensor, label)。"""
        image_path, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(image_path).convert("RGB")
        if self.region_extractor is not None:
            image, _ = self.region_extractor.extract(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_transform(image_size, mean=None, std=None, augment=False):
    """
    构建训练时用的图像变换流水线：resize、可选数据增强（随机裁剪、翻转、颜色抖动等）、ToTensor、归一化、随机擦除。

    Args:
        image_size: 目标尺寸（正方形边长）。
        mean: 归一化均值，默认使用 DEFAULT_MEAN。
        std: 归一化标准差，默认使用 DEFAULT_STD。
        augment: 是否启用数据增强。

    Returns:
        torchvision.transforms.Compose: 变换组合。
    """
    mean = mean or DEFAULT_MEAN
    std = std or DEFAULT_STD
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_inference_transform(image_size, mean=None, std=None):
    """
    构建推理时用的图像变换：Resize、ToTensor、Normalize，无随机性。

    Args:
        image_size: 目标尺寸（正方形边长）。
        mean: 归一化均值。
        std: 归一化标准差。

    Returns:
        torchvision.transforms.Compose: 变换组合。
    """
    mean = mean or DEFAULT_MEAN
    std = std or DEFAULT_STD
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_image_as_tensor(image_path, image_size, mean=None, std=None):
    """
    从路径加载单张图片并应用推理变换，返回带 batch 维的 tensor (1, C, H, W)。

    Args:
        image_path: 图片文件路径。
        image_size: 目标尺寸。
        mean: 归一化均值。
        std: 归一化标准差。

    Returns:
        torch.Tensor: 形状 (1, 3, image_size, image_size)。
    """
    image = Image.open(image_path).convert("RGB")
    return build_inference_transform(image_size, mean, std)(image).unsqueeze(0)


def _split_indices_by_class(samples, val_ratio, seed):
    """
    按类别分层划分索引：每个类别内按 val_ratio 划分出验证集索引，保证 train/val 类别分布一致。

    Args:
        samples: ImageFolder 风格的 (path, label) 列表。
        val_ratio: 验证集比例（0~1）。
        seed: 随机种子。

    Returns:
        tuple: (train_indices, val_indices)。
    """
    label_to_indices = {}
    for index, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(index)

    random_generator = random.Random(seed)
    train_indices = []
    val_indices = []
    for indices in label_to_indices.values():
        random_generator.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        if val_ratio > 0 and len(indices) > 1:
            val_size = max(1, val_size)
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
    return train_indices, val_indices


def create_classification_dataloaders(
    data_dir,
    image_size=224,
    batch_size=32,
    num_workers=0,
    mean=None,
    std=None,
    augment=False,
    val_ratio=0.2,
    seed=42,
    use_roi=False,
    roi_mode="face",
    roi_fallback="smart_crop",
    prefer_explicit_val=True,
):
    """
    创建训练/验证/测试的 DataLoader 及元信息。期望目录结构：data_dir/train、data_dir/val（可选）、data_dir/test。
    若存在 val 目录则优先使用显式验证集，否则从 train 中按类别分层划分 val_ratio 作为验证集。
    可选对每张图先做 ROI 再变换。

    Args:
        data_dir: 数据根目录（含 train、test，可选 val）。
        image_size: 输入图像尺寸。
        batch_size: 批大小。
        num_workers: DataLoader 工作进程数。
        mean: 归一化均值。
        std: 归一化标准差。
        augment: 训练集是否做数据增强。
        val_ratio: 无显式 val 目录时从 train 划分的验证集比例。
        seed: 划分与增强的随机种子。
        use_roi: 是否对图像做 ROI 提取（人脸等）。
        roi_mode: ROI 模式（如 "face"）。
        roi_fallback: ROI 兜底方式（如 "smart_crop"）。
        prefer_explicit_val: 若为 True 且存在 val 目录则使用显式验证集。

    Returns:
        dict: train/val/test 三个 DataLoader，以及 class_names、train_size、val_size、test_size、
              use_roi、roi_mode、roi_fallback、validation_strategy。
    """
    data_root = Path(data_dir)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    base_train_dataset = ImageFolder(str(train_dir))
    base_test_dataset = ImageFolder(str(test_dir))
    class_names = base_train_dataset.classes
    train_transform = build_train_transform(image_size, mean, std, augment=augment)
    eval_transform = build_inference_transform(image_size, mean, std)
    region_extractor = RegionExtractor(mode=roi_mode, fallback_mode=roi_fallback) if use_roi else None

    if prefer_explicit_val and val_dir.exists():
        base_val_dataset = ImageFolder(str(val_dir))
        train_dataset = TransformSubset(
            base_train_dataset,
            range(len(base_train_dataset)),
            transform=train_transform,
            region_extractor=region_extractor,
        )
        val_dataset = TransformSubset(
            base_val_dataset,
            range(len(base_val_dataset)),
            transform=eval_transform,
            region_extractor=region_extractor,
        )
        validation_strategy = "explicit_val_dir"
    else:
        train_indices, val_indices = _split_indices_by_class(base_train_dataset.samples, val_ratio, seed)
        train_dataset = TransformSubset(
            base_train_dataset,
            train_indices,
            transform=train_transform,
            region_extractor=region_extractor,
        )
        val_dataset = TransformSubset(
            base_train_dataset,
            val_indices,
            transform=eval_transform,
            region_extractor=region_extractor,
        )
        validation_strategy = "random_split_from_train"

    test_dataset = TransformSubset(
        base_test_dataset,
        range(len(base_test_dataset)),
        transform=eval_transform,
        region_extractor=region_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "use_roi": use_roi,
        "roi_mode": roi_mode,
        "roi_fallback": roi_fallback,
        "validation_strategy": validation_strategy,
    }
