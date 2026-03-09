from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, root_dir, n_way, k_shot, q_query, transform=None, episodes_per_epoch=100, seed=42):
        self.root_dir = Path(root_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = transform
        self.episodes_per_epoch = episodes_per_epoch
        self.random = random.Random(seed)

        self.class_to_images = {}
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            image_paths = [
                path
                for path in sorted(class_dir.rglob("*"))
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            if len(image_paths) >= self.k_shot + self.q_query:
                self.class_to_images[class_dir.name] = image_paths

        self.class_names = sorted(self.class_to_images.keys())
        if len(self.class_names) < self.n_way:
            raise ValueError(
                f"Dataset at {self.root_dir} only has {len(self.class_names)} valid classes, "
                f"but n_way={self.n_way} was requested."
            )

    def __len__(self):
        return self.episodes_per_epoch

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        return image

    def __getitem__(self, index):
        del index
        sampled_classes = self.random.sample(self.class_names, self.n_way)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for label_idx, class_name in enumerate(sampled_classes):
            sampled_images = self.random.sample(
                self.class_to_images[class_name], self.k_shot + self.q_query
            )
            support_paths = sampled_images[: self.k_shot]
            query_paths = sampled_images[self.k_shot :]

            for image_path in support_paths:
                support_images.append(self._load_image(image_path))
                support_labels.append(label_idx)
            for image_path in query_paths:
                query_images.append(self._load_image(image_path))
                query_labels.append(label_idx)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "sampled_classes": sampled_classes,
        }
