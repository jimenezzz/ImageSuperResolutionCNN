from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def calc_psnr_torch(preds: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((preds - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def shave_border(array: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return array
    return array[border:-border, border:-border]


def calc_psnr_numpy(pred: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    pred = shave_border(pred, border)
    target = shave_border(target, border)
    mse = np.mean(((pred - target) / 255.0) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def calc_ssim_numpy(pred: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    pred = shave_border(pred, border)
    target = shave_border(target, border)
    return float(structural_similarity(target, pred, data_range=255.0))

