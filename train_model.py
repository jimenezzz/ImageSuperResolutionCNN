from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr_project.data import FolderEvalDataset, FolderTrainDataset
from sr_project.metrics import AverageMeter, calc_psnr_torch
from sr_project.models import build_model
from sr_project.runtime import resolve_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["srcnn", "fsrcnn"], required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DIV2K"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patches-per-image", type=int, default=128)
    parser.add_argument("--fsrcnn-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int)
    return parser.parse_args()


def default_hyperparameters(model_name: str) -> tuple[float, int]:
    if model_name == "srcnn":
        return 1e-4, 30
    return 1e-3, 20


def build_optimizer(model_name: str, model: torch.nn.Module, learning_rate: float) -> Adam:
    if model_name == "srcnn":
        return Adam(
            [
                {"params": model.conv1.parameters()},
                {"params": model.conv2.parameters()},
                {"params": model.conv3.parameters(), "lr": learning_rate * 0.1},
            ],
            lr=learning_rate,
        )
    return Adam(
        [
            {"params": model.first_part.parameters()},
            {"params": model.mid_part.parameters()},
            {"params": model.last_part.parameters(), "lr": learning_rate * 0.1},
        ],
        lr=learning_rate,
    )


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for model_input, hr, _, _ in dataloader:
            model_input = model_input.to(device)
            hr = hr.to(device)
            preds = model(model_input).clamp(0.0, 1.0)
            meter.update(calc_psnr_torch(preds, hr), model_input.size(0))
    return meter.avg


def main() -> None:
    args = parse_args()
    learning_rate, num_epochs = default_hyperparameters(args.model)
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.epochs is not None:
        num_epochs = args.epochs

    device = resolve_device(args.device)
    set_seed(args.seed)

    model = build_model(args.model, args.scale).to(device)
    optimizer = build_optimizer(args.model, model, learning_rate)
    criterion = nn.MSELoss()

    pin_memory = device.type == "cuda"
    train_dataset = FolderTrainDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        model_name=args.model,
        scale=args.scale,
        patches_per_image=args.patches_per_image,
        augment=args.fsrcnn_augment,
    )
    eval_dataset = FolderEvalDataset(
        dataset_root=args.dataset_root,
        split=args.val_split,
        model_name=args.model,
        scale=args.scale,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    experiment_dir = args.experiment_root / args.model / f"x{args.scale}"
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_psnr = 0.0
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())

    print(f"Training on {device} using {len(train_dataset)} sampled patches per epoch from {args.dataset_root / args.train_split}")

    for epoch in range(num_epochs):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(train_loader, desc=f"{args.model} epoch {epoch + 1}/{num_epochs}", leave=False)

        for step, (model_input, hr) in enumerate(progress, start=1):
            model_input = model_input.to(device)
            hr = hr.to(device)

            preds = model(model_input)
            loss = criterion(preds, hr)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), model_input.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.6f}")

            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break

        val_psnr = evaluate(model, eval_loader, device)
        epoch_summary = {"epoch": epoch + 1, "train_loss": loss_meter.avg, "val_psnr": val_psnr}
        history.append(epoch_summary)
        print(f"epoch={epoch + 1} loss={loss_meter.avg:.6f} val_psnr={val_psnr:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1}.pth",
                model,
                optimizer,
                epoch + 1,
                {"val_psnr": val_psnr, "train_loss": loss_meter.avg},
                {
                    "model": args.model,
                    "scale": args.scale,
                    "dataset_root": str(args.dataset_root.resolve()),
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "seed": args.seed,
                    "learning_rate": learning_rate,
                    "patches_per_image": args.patches_per_image,
                    "fsrcnn_augment": args.fsrcnn_augment,
                },
            )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    save_checkpoint(
        experiment_dir / "best.pth",
        model,
        optimizer,
        best_epoch,
        {"best_val_psnr": best_psnr},
        {
            "model": args.model,
            "scale": args.scale,
            "dataset_root": str(args.dataset_root.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "seed": args.seed,
            "learning_rate": learning_rate,
            "patches_per_image": args.patches_per_image,
            "fsrcnn_augment": args.fsrcnn_augment,
        },
    )
    save_json(
        experiment_dir / "history.json",
        {
            "model": args.model,
            "scale": args.scale,
            "dataset_root": str(args.dataset_root.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "best_epoch": best_epoch,
            "best_val_psnr": best_psnr,
            "epochs": history,
        },
    )
    print(f"Best checkpoint saved to {experiment_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
