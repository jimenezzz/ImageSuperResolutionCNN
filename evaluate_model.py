from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr_project.data import FolderEvalDataset
from sr_project.metrics import AverageMeter, calc_psnr_numpy, calc_ssim_numpy
from sr_project.models import build_model
from sr_project.runtime import load_checkpoint, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["srcnn", "fsrcnn"], required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DIV2K"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FolderEvalDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        model_name=args.model,
        scale=args.scale,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = resolve_device(args.device)

    model = build_model(args.model, args.scale).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    border = args.scale
    model_psnr = AverageMeter()
    model_ssim = AverageMeter()
    bicubic_psnr = AverageMeter()
    bicubic_ssim = AverageMeter()
    per_image: list[dict] = []

    for model_input, hr, bicubic, name in tqdm(loader, desc=f"evaluating {args.model}", leave=False):
        model_input = model_input.to(device)
        hr = hr.to(device)
        with torch.no_grad():
            preds = model(model_input).clamp(0.0, 1.0)

        bicubic_np = bicubic.squeeze().numpy() * 255.0
        hr_np = hr.cpu().squeeze().numpy() * 255.0
        pred_np = preds.cpu().squeeze().numpy() * 255.0

        sample = {
            "name": name[0],
            "bicubic_psnr": calc_psnr_numpy(bicubic_np, hr_np, border=border),
            "bicubic_ssim": calc_ssim_numpy(bicubic_np, hr_np, border=border),
            "model_psnr": calc_psnr_numpy(pred_np, hr_np, border=border),
            "model_ssim": calc_ssim_numpy(pred_np, hr_np, border=border),
        }
        bicubic_psnr.update(sample["bicubic_psnr"])
        bicubic_ssim.update(sample["bicubic_ssim"])
        model_psnr.update(sample["model_psnr"])
        model_ssim.update(sample["model_ssim"])
        per_image.append(sample)

    summary = {
        "model": args.model,
        "scale": args.scale,
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "crop_border": border,
        "bicubic": {"psnr": bicubic_psnr.avg, "ssim": bicubic_ssim.avg},
        "model_results": {"psnr": model_psnr.avg, "ssim": model_ssim.avg},
        "per_image": per_image,
    }

    print(f"Bicubic  PSNR={bicubic_psnr.avg:.4f} SSIM={bicubic_ssim.avg:.4f}")
    print(f"{args.model.upper()} PSNR={model_psnr.avg:.4f} SSIM={model_ssim.avg:.4f}")

    output_json = args.output_json or args.checkpoint.parent / f"{args.model}_x{args.scale}_{args.split}_metrics.json"
    save_json(output_json, summary)
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
