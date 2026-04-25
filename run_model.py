from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sr_project.image_utils import (
    bicubic_upsample,
    image_to_numpy,
    is_image_file,
    list_images,
    load_rgb_image,
    mod_crop,
    rgb_to_y,
    rgb_to_ycbcr,
    save_rgb_image,
    tensor_to_y_channel,
    y_channel_tensor,
    ycbcr_to_rgb,
)
from sr_project.metrics import calc_psnr_numpy, calc_ssim_numpy
from sr_project.models import build_model
from sr_project.runtime import load_checkpoint, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["srcnn", "fsrcnn"], required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"))
    parser.add_argument("--paired-hr-dir", type=Path)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--degrade-hr-first", action="store_true")
    return parser.parse_args()


def gather_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return list_images(input_path)
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    raise FileNotFoundError(f"No valid image input found at {input_path}")


def predict_single(model_name: str, model: torch.nn.Module, image_path: Path, scale: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    image = load_rgb_image(image_path)
    if model_name == "srcnn":
        lr_rgb = image
        bicubic_rgb = bicubic_upsample(lr_rgb, scale)
        ycbcr = rgb_to_ycbcr(image_to_numpy(bicubic_rgb))
        network_input = y_channel_tensor(ycbcr[..., 0], device=device)
    else:
        lr_rgb = image
        bicubic_rgb = bicubic_upsample(lr_rgb, scale)
        ycbcr = rgb_to_ycbcr(image_to_numpy(bicubic_rgb))
        network_input = y_channel_tensor(rgb_to_y(image_to_numpy(lr_rgb)), device=device)

    with torch.no_grad():
        prediction = model(network_input).clamp(0.0, 1.0)

    pred_y = tensor_to_y_channel(prediction)
    output_ycbcr = np.stack([pred_y, ycbcr[..., 1], ycbcr[..., 2]], axis=-1)
    output_rgb = ycbcr_to_rgb(output_ycbcr)
    return output_rgb, image_to_numpy(bicubic_rgb)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.model, args.scale).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    inputs = gather_inputs(args.input)
    results: list[dict] = []

    for image_path in tqdm(inputs, desc=f"running {args.model}", leave=False):
        inference_path = image_path
        hr_path = None

        if args.degrade_hr_first:
            hr_image = mod_crop(load_rgb_image(image_path), args.scale)
            lr_image = hr_image.resize(
                (hr_image.width // args.scale, hr_image.height // args.scale),
                resample=Image.Resampling.BICUBIC,
            )
            temp_input = args.output_dir / "_tmp_lr_inputs" / f"{image_path.stem}_x{args.scale}{image_path.suffix}"
            temp_input.parent.mkdir(parents=True, exist_ok=True)
            lr_image.save(temp_input)
            inference_path = temp_input
            hr_path = image_path
        elif args.paired_hr_dir is not None:
            hr_candidate = args.paired_hr_dir / image_path.name.replace(f"x{args.scale}", "")
            if hr_candidate.exists():
                hr_path = hr_candidate

        output_rgb, bicubic_rgb = predict_single(args.model, model, inference_path, args.scale, device)
        output_path = args.output_dir / args.model / f"x{args.scale}" / f"{image_path.stem}_{args.model}.png"
        save_rgb_image(output_rgb, output_path)

        sample = {"input": str(image_path), "output": str(output_path)}
        if hr_path is not None and hr_path.exists():
            hr_image = mod_crop(load_rgb_image(hr_path), args.scale)
            hr_y = rgb_to_y(image_to_numpy(hr_image))
            pred_y = rgb_to_y(np.clip(output_rgb, 0.0, 255.0))
            bicubic_y = rgb_to_y(np.clip(bicubic_rgb, 0.0, 255.0))
            border = args.scale
            sample.update(
                {
                    "psnr": calc_psnr_numpy(pred_y, hr_y, border=border),
                    "ssim": calc_ssim_numpy(pred_y, hr_y, border=border),
                    "bicubic_psnr": calc_psnr_numpy(bicubic_y, hr_y, border=border),
                    "bicubic_ssim": calc_ssim_numpy(bicubic_y, hr_y, border=border),
                }
            )
        results.append(sample)

    metrics_path = args.output_dir / args.model / f"x{args.scale}" / "run_summary.json"
    save_json(metrics_path, {"model": args.model, "scale": args.scale, "results": results})
    print(f"Saved predictions to {args.output_dir / args.model / f'x{args.scale}'}")


if __name__ == "__main__":
    main()
