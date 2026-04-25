from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and is_image_file(path))


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def mod_crop(image: Image.Image, scale: int) -> Image.Image:
    width = (image.width // scale) * scale
    height = (image.height // scale) * scale
    return image.resize((width, height), resample=Image.Resampling.BICUBIC)


def bicubic_downsample(image: Image.Image, scale: int) -> Image.Image:
    return image.resize((image.width // scale, image.height // scale), resample=Image.Resampling.BICUBIC)


def bicubic_upsample(image: Image.Image, scale: int) -> Image.Image:
    return image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.BICUBIC)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    return 16.0 + (64.738 * image[..., 0] + 129.057 * image[..., 1] + 25.064 * image[..., 2]) / 256.0


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    y = 16.0 + (64.738 * image[..., 0] + 129.057 * image[..., 1] + 25.064 * image[..., 2]) / 256.0
    cb = 128.0 + (-37.945 * image[..., 0] - 74.494 * image[..., 1] + 112.439 * image[..., 2]) / 256.0
    cr = 128.0 + (112.439 * image[..., 0] - 94.154 * image[..., 1] - 18.285 * image[..., 2]) / 256.0
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    r = 298.082 * image[..., 0] / 256.0 + 408.583 * image[..., 2] / 256.0 - 222.921
    g = (
        298.082 * image[..., 0] / 256.0
        - 100.291 * image[..., 1] / 256.0
        - 208.120 * image[..., 2] / 256.0
        + 135.576
    )
    b = 298.082 * image[..., 0] / 256.0 + 516.412 * image[..., 1] / 256.0 - 276.836
    return np.stack([r, g, b], axis=-1)


def y_channel_tensor(y_channel: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(y_channel / 255.0).float().unsqueeze(0).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_y_channel(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0.0, 1.0).cpu().squeeze().numpy() * 255.0


def save_rgb_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0.0, 255.0).astype(np.uint8)).save(path)


def paired_div2k_paths(dataset_root: Path, split: str, scale: int) -> list[tuple[Path, Path]]:
    hr_dir = dataset_root / split / "HR"
    lr_dir = dataset_root / split / "LR_bicubic" / f"X{scale}"
    hr_paths = list_images(hr_dir)
    pairs: list[tuple[Path, Path]] = []
    for hr_path in hr_paths:
        lr_name = f"{hr_path.stem}x{scale}{hr_path.suffix}"
        lr_path = lr_dir / lr_name
        if not lr_path.exists():
            raise FileNotFoundError(f"Missing LR image for {hr_path.name}: {lr_path}")
        pairs.append((hr_path, lr_path))
    return pairs

