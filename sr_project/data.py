from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sr_project.image_utils import (
    bicubic_upsample,
    image_to_numpy,
    load_rgb_image,
    mod_crop,
    paired_div2k_paths,
    rgb_to_y,
)


SRCNN_PATCH_SIZE = 33
SRCNN_STRIDE = 14
FSRCNN_PATCH_SIZE = {2: 10, 3: 7, 4: 6, 8: 3}


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_top_left(height: int, width: int, patch_size: int, step: int) -> tuple[int, int]:
    max_top = height - patch_size
    max_left = width - patch_size
    if max_top < 0 or max_left < 0:
        raise ValueError(f"Image is too small for patch size {patch_size}: got {(height, width)}")

    if max_top == 0:
        top = 0
    else:
        top = random.randrange(0, max_top + 1, step)

    if max_left == 0:
        left = 0
    else:
        left = random.randrange(0, max_left + 1, step)

    return top, left


def _augment_pair(lr_patch: np.ndarray, hr_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = random.randint(0, 3)
    if rotations:
        lr_patch = np.rot90(lr_patch, rotations).copy()
        hr_patch = np.rot90(hr_patch, rotations).copy()

    if random.random() < 0.5:
        lr_patch = np.fliplr(lr_patch).copy()
        hr_patch = np.fliplr(hr_patch).copy()

    if random.random() < 0.5:
        lr_patch = np.flipud(lr_patch).copy()
        hr_patch = np.flipud(hr_patch).copy()

    return lr_patch, hr_patch


def _srcnn_pair_from_paths(hr_path: Path, lr_path: Path, scale: int) -> tuple[np.ndarray, np.ndarray]:
    hr_image = mod_crop(load_rgb_image(hr_path), scale)
    lr_image = load_rgb_image(lr_path)
    lr_upscaled = bicubic_upsample(lr_image, scale)
    hr_y = rgb_to_y(image_to_numpy(hr_image)).astype(np.float32)
    lr_y = rgb_to_y(image_to_numpy(lr_upscaled)).astype(np.float32)
    return lr_y, hr_y


def _fsrcnn_pair_from_paths(hr_path: Path, lr_path: Path, scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hr_image = mod_crop(load_rgb_image(hr_path), scale)
    lr_image = load_rgb_image(lr_path)
    bicubic = bicubic_upsample(lr_image, scale)
    hr_y = rgb_to_y(image_to_numpy(hr_image)).astype(np.float32)
    lr_y = rgb_to_y(image_to_numpy(lr_image)).astype(np.float32)
    bicubic_y = rgb_to_y(image_to_numpy(bicubic)).astype(np.float32)
    return lr_y, hr_y, bicubic_y


class FolderTrainDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path | str,
        split: str,
        model_name: str,
        scale: int,
        patches_per_image: int,
        augment: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.model_name = model_name
        self.scale = scale
        self.patches_per_image = patches_per_image
        self.augment = augment and model_name == "fsrcnn"
        self.pairs = paired_div2k_paths(self.dataset_root, split, scale)
        if not self.pairs:
            raise ValueError(f"No image pairs found in {self.dataset_root / split}")

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_index = (index // self.patches_per_image) % len(self.pairs)
        hr_path, lr_path = self.pairs[pair_index]

        if self.model_name == "srcnn":
            lr_y, hr_y = _srcnn_pair_from_paths(hr_path, lr_path, self.scale)
            top, left = _sample_top_left(lr_y.shape[0], lr_y.shape[1], SRCNN_PATCH_SIZE, SRCNN_STRIDE)
            lr_patch = lr_y[top : top + SRCNN_PATCH_SIZE, left : left + SRCNN_PATCH_SIZE]
            hr_patch = hr_y[top : top + SRCNN_PATCH_SIZE, left : left + SRCNN_PATCH_SIZE]
        else:
            lr_y, hr_y, _ = _fsrcnn_pair_from_paths(hr_path, lr_path, self.scale)
            patch_size = FSRCNN_PATCH_SIZE[self.scale]
            top, left = _sample_top_left(lr_y.shape[0], lr_y.shape[1], patch_size, self.scale)
            lr_patch = lr_y[top : top + patch_size, left : left + patch_size]
            hr_patch = hr_y[
                top * self.scale : top * self.scale + patch_size * self.scale,
                left * self.scale : left * self.scale + patch_size * self.scale,
            ]

        if self.augment:
            lr_patch, hr_patch = _augment_pair(lr_patch, hr_patch)

        lr_tensor = torch.from_numpy((lr_patch / 255.0).astype(np.float32)).unsqueeze(0)
        hr_tensor = torch.from_numpy((hr_patch / 255.0).astype(np.float32)).unsqueeze(0)
        return lr_tensor, hr_tensor


class FolderEvalDataset(Dataset):
    def __init__(self, dataset_root: Path | str, split: str, model_name: str, scale: int) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.model_name = model_name
        self.scale = scale
        self.pairs = paired_div2k_paths(self.dataset_root, split, scale)
        if not self.pairs:
            raise ValueError(f"No image pairs found in {self.dataset_root / split}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        hr_path, lr_path = self.pairs[index]
        if self.model_name == "srcnn":
            model_input, hr_y = _srcnn_pair_from_paths(hr_path, lr_path, self.scale)
            bicubic_y = model_input
        else:
            model_input, hr_y, bicubic_y = _fsrcnn_pair_from_paths(hr_path, lr_path, self.scale)

        return (
            torch.from_numpy((model_input / 255.0).astype(np.float32)).unsqueeze(0),
            torch.from_numpy((hr_y / 255.0).astype(np.float32)).unsqueeze(0),
            torch.from_numpy((bicubic_y / 255.0).astype(np.float32)).unsqueeze(0),
            hr_path.name,
        )
