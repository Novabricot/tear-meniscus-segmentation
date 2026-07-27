#!/usr/bin/env python3
"""
Analyze segmentation failure cases by mask size and shape.

This script evaluates an existing U-Net checkpoint on the test set and saves:
    - per-image metrics
    - summary by mask-size group
    - summary by boundary-to-area group
    - best/worst prediction visualizations

Goal:
    Understand whether segmentation errors are associated with:
        - small masks
        - large masks
        - thin / boundary-heavy masks
        - visually ambiguous examples

This script does not train anything.
"""

from __future__ import annotations

import argparse
import csv
import math
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
    parser = argparse.ArgumentParser(description="Analyze failure cases by mask size.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of best/worst examples.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device.",
    )
    return parser.parse_args()


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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


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


def compute_binary_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    pred, gt: bool tensors of shape (H, W)
    """
    tp = float((pred & gt).sum().item())
    fp = float((pred & (~gt)).sum().item())
    fn = float(((~pred) & gt).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    intersection = float((pred & gt).sum().item())
    union = float((pred | gt).sum().item())
    miou = intersection / (union + 1e-8)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "miou": miou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def binary_erosion(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """
    mask: bool tensor of shape (H, W)
    """
    x = mask.float().unsqueeze(0).unsqueeze(0)
    inv = 1.0 - x
    kernel_size = 2 * radius + 1
    dilated_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - dilated_inv
    return eroded.squeeze(0).squeeze(0) > 0.5


def mask_shape_stats(gt: torch.Tensor) -> Dict[str, float]:
    """
    gt: bool tensor of shape (H, W)
    """
    h, w = gt.shape
    area = float(gt.sum().item())
    image_area = float(h * w)
    area_ratio = area / image_area if image_area > 0 else 0.0

    if area <= 0:
        return {
            "gt_area": 0.0,
            "gt_area_ratio": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "bbox_fill_ratio": 0.0,
            "boundary_pixels": 0.0,
            "boundary_to_area": 0.0,
        }

    ys, xs = torch.where(gt)
    x_min = int(xs.min().item())
    x_max = int(xs.max().item())
    y_min = int(ys.min().item())
    y_max = int(ys.max().item())

    bbox_width = float(x_max - x_min + 1)
    bbox_height = float(y_max - y_min + 1)
    bbox_area = bbox_width * bbox_height
    bbox_fill_ratio = area / bbox_area if bbox_area > 0 else 0.0

    eroded = binary_erosion(gt, radius=1)
    boundary = gt & (~eroded)
    boundary_pixels = float(boundary.sum().item())
    boundary_to_area = boundary_pixels / area if area > 0 else 0.0

    return {
        "gt_area": area,
        "gt_area_ratio": area_ratio,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_fill_ratio": bbox_fill_ratio,
        "boundary_pixels": boundary_pixels,
        "boundary_to_area": boundary_to_area,
    }


def assign_tertile_groups(rows: List[Dict[str, float]], key: str, group_key: str) -> None:
    values = np.array([row[key] for row in rows], dtype=float)

    q1 = float(np.quantile(values, 1.0 / 3.0))
    q2 = float(np.quantile(values, 2.0 / 3.0))

    for row in rows:
        value = row[key]
        if value <= q1:
            row[group_key] = "small" if key == "gt_area" else "low"
        elif value <= q2:
            row[group_key] = "medium"
        else:
            row[group_key] = "large" if key == "gt_area" else "high"


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def summarize_by_group(rows: List[Dict[str, float]], group_key: str) -> List[Dict[str, float]]:
    groups = sorted(set(str(row[group_key]) for row in rows))
    summary_rows = []

    for group in groups:
        group_rows = [row for row in rows if str(row[group_key]) == group]

        f1_mean, f1_std = mean_std([row["f1"] for row in group_rows])
        miou_mean, miou_std = mean_std([row["miou"] for row in group_rows])
        precision_mean, precision_std = mean_std([row["precision"] for row in group_rows])
        recall_mean, recall_std = mean_std([row["recall"] for row in group_rows])
        area_mean, area_std = mean_std([row["gt_area"] for row in group_rows])
        boundary_ratio_mean, boundary_ratio_std = mean_std(
            [row["boundary_to_area"] for row in group_rows]
        )

        summary_rows.append(
            {
                group_key: group,
                "n_images": len(group_rows),
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "miou_mean": miou_mean,
                "miou_std": miou_std,
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
                "gt_area_mean": area_mean,
                "gt_area_std": area_std,
                "boundary_to_area_mean": boundary_ratio_mean,
                "boundary_to_area_std": boundary_ratio_std,
            }
        )

    return summary_rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_for_display(image: torch.Tensor) -> np.ndarray:
    """
    image: (C, H, W)
    Returns display image in [0, 1].
    """
    img = image.detach().cpu().float()

    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img[:3].permute(1, 2, 0)
    else:
        img = img.squeeze()

    arr = img.numpy()
    arr = arr.astype(np.float32)

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr)

    return arr


def save_examples(
    examples: List[Dict],
    output_dir: Path,
    name: str,
    max_examples: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not import matplotlib, skipping examples: {exc}")
        return

    selected = examples[:max_examples]
    if not selected:
        return

    examples_dir = output_dir / name
    examples_dir.mkdir(parents=True, exist_ok=True)

    for rank, example in enumerate(selected, start=1):
        image = image_for_display(example["image"])
        gt = example["gt"].detach().cpu().numpy().astype(bool)
        pred = example["pred"].detach().cpu().numpy().astype(bool)

        false_positive = pred & (~gt)
        false_negative = (~pred) & gt

        error_map = np.zeros((*gt.shape, 3), dtype=np.float32)
        error_map[..., 0] = false_positive.astype(np.float32)
        error_map[..., 1] = false_negative.astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground truth")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")

        axes[3].imshow(error_map)
        axes[3].set_title("Errors: FP red, FN green")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{name} #{rank} | image index={example['image_index']} | "
            f"F1={example['f1']:.4f} | area={example['gt_area']:.0f}"
        )
        fig.tight_layout()

        fig_path = examples_dir / f"{rank:02d}_idx{example['image_index']}_f1_{example['f1']:.4f}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    print(f"Saved {name} examples to {examples_dir}")


def try_make_plots(
    size_summary: List[Dict[str, float]],
    shape_summary: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not import matplotlib, skipping plots: {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # F1 by mask size group
    size_order = ["small", "medium", "large"]
    size_map = {row["mask_size_group"]: row for row in size_summary}
    x = [group for group in size_order if group in size_map]
    y = [size_map[group]["f1_mean"] for group in x]
    yerr = [size_map[group]["f1_std"] for group in x]

    plt.figure(figsize=(7, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xlabel("Mask size group")
    plt.ylabel("F1")
    plt.title("F1 by ground-truth mask size")
    plt.ylim(max(0.0, min(y) - 0.1), min(1.0, max(y) + 0.05))
    plt.tight_layout()
    path = figures_dir / "f1_by_mask_size.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved figure: {path}")

    # F1 by boundary-to-area group
    shape_order = ["low", "medium", "high"]
    shape_map = {row["boundary_to_area_group"]: row for row in shape_summary}
    x = [group for group in shape_order if group in shape_map]
    y = [shape_map[group]["f1_mean"] for group in x]
    yerr = [shape_map[group]["f1_std"] for group in x]

    plt.figure(figsize=(7, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xlabel("Boundary-to-area group")
    plt.ylabel("F1")
    plt.title("F1 by mask boundary-to-area ratio")
    plt.ylim(max(0.0, min(y) - 0.1), min(1.0, max(y) + 0.05))
    plt.tight_layout()
    path = figures_dir / "f1_by_boundary_to_area.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved figure: {path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    trainer = UNetTrainer(config_path=args.config)
    model = trainer.build_model()
    model.to(device)
    load_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    _train_loader, _val_loader, test_loader = trainer.build_dataloaders()

    rows: List[Dict[str, float]] = []
    examples: List[Dict] = []

    image_index = 0

    for batch in tqdm(test_loader, desc="Analyze test set"):
        images, targets = get_batch(batch, device)

        logits = model(images)
        probs = get_foreground_probability(logits)
        preds = probs >= 0.5

        for i in range(images.shape[0]):
            pred_i = preds[i]
            gt_i = targets[i].bool()

            metrics = compute_binary_metrics(pred_i, gt_i)
            shape = mask_shape_stats(gt_i)

            row = {
                "image_index": image_index,
                **metrics,
                **shape,
            }
            rows.append(row)

            examples.append(
                {
                    "image_index": image_index,
                    "image": images[i].detach().cpu(),
                    "gt": gt_i.detach().cpu(),
                    "pred": pred_i.detach().cpu(),
                    "f1": metrics["f1"],
                    "miou": metrics["miou"],
                    "gt_area": shape["gt_area"],
                    "boundary_to_area": shape["boundary_to_area"],
                }
            )

            image_index += 1

    assign_tertile_groups(rows, key="gt_area", group_key="mask_size_group")
    assign_tertile_groups(rows, key="boundary_to_area", group_key="boundary_to_area_group")

    size_summary = summarize_by_group(rows, group_key="mask_size_group")
    shape_summary = summarize_by_group(rows, group_key="boundary_to_area_group")

    write_csv(output_dir / "per_image_metrics.csv", rows)
    write_csv(output_dir / "summary_by_mask_size.csv", size_summary)
    write_csv(output_dir / "summary_by_boundary_to_area.csv", shape_summary)

    with (output_dir / "summary_by_mask_size.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(size_summary, f, sort_keys=False)

    with (output_dir / "summary_by_boundary_to_area.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(shape_summary, f, sort_keys=False)

    examples_sorted_worst = sorted(examples, key=lambda x: x["f1"])
    examples_sorted_best = sorted(examples, key=lambda x: x["f1"], reverse=True)

    save_examples(examples_sorted_worst, output_dir, "worst_examples", args.num_examples)
    save_examples(examples_sorted_best, output_dir, "best_examples", args.num_examples)

    try_make_plots(size_summary, shape_summary, output_dir)

    print("\nSummary by mask size:")
    for row in size_summary:
        print(row)

    print("\nSummary by boundary-to-area:")
    for row in shape_summary:
        print(row)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
    