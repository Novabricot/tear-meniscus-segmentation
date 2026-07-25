#!/usr/bin/env python3
"""
Train a U-Net with uncertainty-weighted loss.

Idea:
    Manual pixel-level annotations are less certain near object boundaries.
    We define an "uncertain boundary" region using GT dilation - GT erosion.
    Pixels in this boundary region receive a lower loss weight.

Regions:
    - certain positive: eroded GT
    - uncertain boundary: dilated GT - eroded GT
    - certain negative: outside dilated GT

Loss:
    weighted BCE/CE + weighted Dice

This script trains from scratch using an existing YAML config for data/model/optimizer.
It saves:
    - best_model.pth
    - last_model.pth
    - test_metrics.yaml
    - training_log.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_unet import UNetTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net with uncertainty-weighted loss.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for results.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=44, help="Random seed.")
    parser.add_argument("--boundary-radius", type=int, default=1, help="Boundary radius in pixels.")
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.5,
        help="Loss weight for uncertain boundary pixels.",
    )
    parser.add_argument(
        "--foreground-dice-weight",
        type=float,
        default=1.0,
        help="Weight for Dice loss term.",
    )
    parser.add_argument(
        "--pixel-loss-weight",
        type=float,
        default=1.0,
        help="Weight for BCE/CE pixel loss term.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        images = batch["image"]
        targets = batch["mask"]
    else:
        images, targets = batch[:2]

    images = images.to(device)
    targets = targets.to(device)

    if targets.ndim == 4:
        targets = targets.squeeze(1)

    targets = targets.long()
    return images, targets


def binary_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    mask: (B, H, W), bool or 0/1.
    """
    if radius <= 0:
        return mask.bool()

    x = mask.float().unsqueeze(1)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(1) > 0.5


def binary_erosion(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    mask: (B, H, W), bool or 0/1.
    """
    if radius <= 0:
        return mask.bool()

    inv = 1.0 - mask.float()
    kernel_size = 2 * radius + 1
    dilated_inv = F.max_pool2d(inv.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - dilated_inv.squeeze(1)
    return eroded > 0.5


def make_uncertainty_weights(
    targets: torch.Tensor,
    radius: int,
    boundary_weight: float,
) -> torch.Tensor:
    """
    targets: (B, H, W), values 0/1.

    Returns:
        weights: (B, H, W)
    """
    gt = targets.bool()

    eroded = binary_erosion(gt, radius)
    dilated = binary_dilation(gt, radius)
    uncertain_boundary = dilated & (~eroded)

    weights = torch.ones_like(targets, dtype=torch.float32)
    weights[uncertain_boundary] = boundary_weight
    return weights


def get_foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns foreground probability of shape (B, H, W).
    Supports:
        - 1-channel binary logits
        - 2-channel class logits
    """
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0, :, :])
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1, :, :]
    raise ValueError(f"Expected 1 or 2 output channels, got {tuple(logits.shape)}")


def weighted_pixel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted BCE for 1-channel outputs.
    Weighted CE for 2-channel outputs.
    """
    if logits.shape[1] == 1:
        target_float = targets.float()
        loss_map = F.binary_cross_entropy_with_logits(
            logits[:, 0, :, :],
            target_float,
            reduction="none",
        )
    elif logits.shape[1] == 2:
        loss_map = F.cross_entropy(logits, targets, reduction="none")
    else:
        raise ValueError(f"Expected 1 or 2 output channels, got {tuple(logits.shape)}")

    return (loss_map * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted soft Dice loss for foreground class.
    """
    probs = get_foreground_probability(logits)
    target_float = targets.float()

    probs = probs * weights
    target_weighted = target_float * weights

    intersection = (probs * target_float * weights).sum(dim=(1, 2))
    denominator = probs.sum(dim=(1, 2)) + target_weighted.sum(dim=(1, 2))

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def uncertainty_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    boundary_radius: int,
    boundary_weight: float,
    pixel_loss_weight: float,
    foreground_dice_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = make_uncertainty_weights(
        targets=targets,
        radius=boundary_radius,
        boundary_weight=boundary_weight,
    )

    pixel = weighted_pixel_loss(logits, targets, weights)
    dice = weighted_dice_loss(logits, targets, weights)

    loss = pixel_loss_weight * pixel + foreground_dice_weight * dice

    parts = {
        "loss": float(loss.detach().cpu()),
        "pixel_loss": float(pixel.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
    }
    return loss, parts


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()

    tp = 0.0
    fp = 0.0
    fn = 0.0
    intersection_total = 0.0
    union_total = 0.0
    losses: List[float] = []

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        images, targets = get_batch(batch, device)
        logits = model(images)

        # Standard unweighted evaluation loss for monitoring.
        weights = torch.ones_like(targets, dtype=torch.float32)
        if logits.shape[1] == 1:
            loss_map = F.binary_cross_entropy_with_logits(
                logits[:, 0, :, :],
                targets.float(),
                reduction="none",
            )
        else:
            loss_map = F.cross_entropy(logits, targets, reduction="none")
        losses.append(float((loss_map * weights).mean().detach().cpu()))

        probs = get_foreground_probability(logits)
        preds = probs >= 0.5
        gt = targets.bool()

        tp += float((preds & gt).sum().item())
        fp += float((preds & (~gt)).sum().item())
        fn += float(((~preds) & gt).sum().item())

        intersection_total += float((preds & gt).sum().item())
        union_total += float((preds | gt).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    miou = intersection_total / (union_total + 1e-8)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "miou": miou,
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def write_training_log(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Config: {args.config}")
    print(f"Boundary radius: {args.boundary_radius}")
    print(f"Boundary weight: {args.boundary_weight}")

    trainer = UNetTrainer(config_path=args.config)
    train_loader, val_loader, test_loader = trainer.build_dataloaders()

    model = trainer.build_model()
    model.to(device)

    optimizer = trainer.build_optimizer(model)

    try:
        scheduler = trainer.build_scheduler(optimizer)
    except Exception:
        scheduler = None

    best_val_f1 = -1.0
    training_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_losses = []
        epoch_pixel_losses = []
        epoch_dice_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            images, targets = get_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)

            loss, parts = uncertainty_weighted_loss(
                logits=logits,
                targets=targets,
                boundary_radius=args.boundary_radius,
                boundary_weight=args.boundary_weight,
                pixel_loss_weight=args.pixel_loss_weight,
                foreground_dice_weight=args.foreground_dice_weight,
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(parts["loss"])
            epoch_pixel_losses.append(parts["pixel_loss"])
            epoch_dice_losses.append(parts["dice_loss"])

            pbar.set_postfix(loss=np.mean(epoch_losses))

        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                pass

        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_pixel_loss": float(np.mean(epoch_pixel_losses)),
            "train_dice_loss": float(np.mean(epoch_dice_losses)),
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_miou": val_metrics["miou"],
            "val_loss": val_metrics["loss"],
        }
        training_rows.append(row)
        write_training_log(output_dir / "training_log.csv", training_rows)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={row['train_loss']:.4f} | "
            f"val F1={row['val_f1']:.4f} | "
            f"val mIoU={row['val_miou']:.4f}"
        )

        save_checkpoint(
            checkpoint_dir / "last_model.pth",
            model,
            optimizer,
            epoch,
            val_metrics,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model,
                optimizer,
                epoch,
                val_metrics,
            )
            print(f"Saved new best model at epoch {epoch}: val F1={best_val_f1:.4f}")

    # Evaluate best model on test set.
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    test_metrics_path = output_dir / "test_metrics.yaml"
    with test_metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(test_metrics, f, sort_keys=False)

    print("Final test metrics:")
    print(test_metrics)
    print(f"Saved test metrics: {test_metrics_path}")


if __name__ == "__main__":
    main()