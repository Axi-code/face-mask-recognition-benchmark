import argparse
from pathlib import Path

import numpy as np
from PIL import Image


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute dataset mean and std.")
    parser.add_argument("--data-root", type=str, default="dataset")
    return parser.parse_args()


def iter_images(root_dir):
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.suffix.lower() in VALID_SUFFIXES:
            yield path


def main():
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
