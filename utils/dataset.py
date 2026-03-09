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
    def __init__(self, dataset, indices, transform=None, region_extractor=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.region_extractor = region_extractor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(image_path).convert("RGB")
        if self.region_extractor is not None:
            image, _ = self.region_extractor.extract(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_transform(image_size, mean=None, std=None, augment=False):
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
    image = Image.open(image_path).convert("RGB")
    return build_inference_transform(image_size, mean, std)(image).unsqueeze(0)


def _split_indices_by_class(samples, val_ratio, seed):
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
