"""
小样本学习用的 Episode 数据集：按 N-way K-shot 组织，每 epoch 生成固定数量的 episode 供原型网络训练与评估。
"""
from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    """
    小样本学习（Few-shot）的 Episode 数据集。
    按 N-way K-shot 方式组织：每个 episode 随机采样 N 个类别，每类 K 张支撑图 + Q 张查询图，
    用于原型网络等 meta-learning 训练与评估。
    """

    def __init__(self, root_dir, n_way, k_shot, q_query, transform=None, episodes_per_epoch=100, seed=42):
        """
        初始化 Episode 数据集。

        Args:
            root_dir: 数据根目录，其下每个子目录为一类（类名为目录名），内含该类图片。
            n_way: 每个 episode 采样的类别数（如 2 表示二分类 mask/no_mask）。
            k_shot: 每类支撑集（support set）样本数。
            q_query: 每类查询集（query set）样本数。
            transform: 可选，对图片施加的变换（如 resize、归一化）。
            episodes_per_epoch: 每个 epoch 生成的 episode 数量，即 __len__ 的返回值。
            seed: 随机种子，保证 episode 采样可复现。

        Raises:
            ValueError: 当有效类别数少于 n_way 时抛出。
        """
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
        """返回每个 epoch 的 episode 数量（用于 DataLoader 迭代）。"""
        return self.episodes_per_epoch

    def _load_image(self, image_path):
        """
        从路径加载单张图片并转为 RGB，若有 transform 则应用后返回张量。

        Args:
            image_path: 图片文件路径（Path 或 str）。

        Returns:
            变换后的张量或 PIL Image（取决于是否设置了 transform）。
        """
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        return image

    def __getitem__(self, index):
        """
        获取第 index 个 episode 的数据（index 仅用于满足 DataLoader，实际随机采样）。

        Returns:
            dict: 包含 support_images, support_labels, query_images, query_labels（均为 tensor），
                  以及 sampled_classes（本 episode 采样的类别名列表）。
        """
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
