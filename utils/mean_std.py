import argparse
from pathlib import Path

import numpy as np
from PIL import Image


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    """解析命令行参数，获取数据根目录（用于计算数据集的 RGB 均值和标准差）。"""
    parser = argparse.ArgumentParser(description="Compute dataset mean and std.")
    parser.add_argument("--data-root", type=str, default="dataset")
    return parser.parse_args()


def iter_images(root_dir):
    """
    递归遍历目录下所有图片文件（.jpg/.jpeg/.png/.bmp/.webp），逐个 yield 路径。

    Args:
        root_dir: 根目录路径（str 或 Path）。

    Yields:
        Path: 每个图片文件的路径。
    """
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.suffix.lower() in VALID_SUFFIXES:
            yield path


def main():
    """主入口：遍历数据集中所有图片，计算 RGB 三通道的均值和标准差并打印。"""
    args = parse_args()
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for image_path in iter_images(args.data_root):
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        pixels = image_array.reshape(-1, 3)

        channel_sum += pixels.sum(axis=0)
        channel_squared_sum += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = channel_sum / total_pixels
    std = np.sqrt(channel_squared_sum / total_pixels - np.square(mean))

    print("Mean:", mean.tolist())
    print("Std:", std.tolist())


if __name__ == "__main__":
    main()
