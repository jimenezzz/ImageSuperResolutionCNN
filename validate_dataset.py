from __future__ import annotations

import argparse
from pathlib import Path

from sr_project.image_utils import paired_div2k_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DIV2K"))
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8])
    return parser.parse_args()


def summarize_split(dataset_root: Path, split: str, scale: int) -> tuple[int, Path, Path]:
    pairs = paired_div2k_paths(dataset_root, split, scale)
    hr_dir = dataset_root / split / "HR"
    lr_dir = dataset_root / split / "LR_bicubic" / f"X{scale}"
    return len(pairs), hr_dir, lr_dir


def main() -> None:
    args = parse_args()
    print(f"Validating dataset layout under {args.dataset_root.resolve()} for scale x{args.scale}.")

    total = 0
    for split in ["train", "val", "test"]:
        count, hr_dir, lr_dir = summarize_split(args.dataset_root, split, args.scale)
        total += count
        print(f"{split}: {count} paired images")
        print(f"  HR: {hr_dir}")
        print(f"  LR: {lr_dir}")

    print(f"Total paired images: {total}")


if __name__ == "__main__":
    main()
