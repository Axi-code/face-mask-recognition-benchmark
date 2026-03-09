import argparse
import random
import shutil
from pathlib import Path


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split raw dataset into train/test folders.")
    parser.add_argument("--source-root", type=str, default="dataset")
    parser.add_argument("--output-root", type=str, default="data")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-separator",
        type=str,
        default="_",
        help="Keep similarly named files together when splitting, e.g. person1_001.jpg and person1_002.jpg.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing train/test folders before copying new files.",
    )
    return parser.parse_args()


def collect_images(class_dir):
    return [path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in VALID_SUFFIXES]


def group_images(images, separator):
    grouped_images = {}
    for image_path in images:
        stem = image_path.stem
        group_key = stem.split(separator)[0] if separator and separator in stem else stem
        grouped_images.setdefault(group_key, []).append(image_path)
    return grouped_images


def main():
    args = parse_args()
    random.seed(args.seed)

    # 默认路径相对于项目根目录（脚本所在 utils/ 的上级）
    project_root = Path(__file__).resolve().parent.parent
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not source_root.is_absolute():
        source_root = project_root / source_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    train_root = output_root / "train"
    test_root = output_root / "test"

    if not source_root.exists():
        print(f"错误：源数据目录不存在: {source_root}")
        print(f"请先创建该目录，并按类别放入子文件夹，例如：")
        print(f"  {source_root}/mask/")
        print(f"  {source_root}/no_mask/")
        return

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue

        images = collect_images(class_dir)
        if not images:
            continue

        image_groups = list(group_images(images, args.group_separator).values())
        random.shuffle(image_groups)
        test_target = max(1, int(len(images) * args.test_ratio))
        test_images = []
        train_images = []
        for group in image_groups:
            if len(test_images) < test_target:
                needed = test_target - len(test_images)
                if len(group) <= needed:
                    test_images.extend(group)
                else:
                    # 单组过大时拆分，避免整类全进 test
                    test_images.extend(group[:needed])
                    train_images.extend(group[needed:])
            else:
                train_images.extend(group)

        (train_root / class_dir.name).mkdir(parents=True, exist_ok=True)
        (test_root / class_dir.name).mkdir(parents=True, exist_ok=True)

        for image_path in train_images:
            shutil.copy2(image_path, train_root / class_dir.name / image_path.name)
        for image_path in test_images:
            shutil.copy2(image_path, test_root / class_dir.name / image_path.name)

        print(
            f"{class_dir.name}: train={len(train_images)} test={len(test_images)}"
        )

    print("dataset split finished")


if __name__ == "__main__":
    main()