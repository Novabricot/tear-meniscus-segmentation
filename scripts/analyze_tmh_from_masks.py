"""
Analyze tear meniscus height from predicted and ground-truth masks.

This script evaluates a trained U-Net checkpoint on the test split of a config,
computes several automatic height measurements from the predicted masks and
ground-truth masks, and compares them.

The goal is not to perform a clinical diagnosis, but to check whether a
clinically relevant derived quantity, such as tear meniscus height, remains
stable even when pixel-wise segmentation metrics are imperfect.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_unet import UNetTrainer  # noqa: E402


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)


def extract_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Supports common dataset output formats:
    - (images, masks)
    - (images, masks, names)
    - dict with image/mask/name keys
    """

    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        masks = batch.get("mask", batch.get("masks", batch.get("target", batch.get("targets"))))
        names = batch.get("name", batch.get("names", batch.get("filename", batch.get("filenames"))))

        if images is None or masks is None:
            raise ValueError(f"Could not extract image/mask from batch keys: {batch.keys()}")

        if names is None:
            names = [f"sample_{i}" for i in range(images.shape[0])]
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        return images, masks, names

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, masks = batch
            names = [f"sample_{i}" for i in range(images.shape[0])]
            return images, masks, names

        if len(batch) >= 3:
            images, masks, names = batch[0], batch[1], batch[2]
            if isinstance(names, str):
                names = [names]
            else:
                names = list(names)
            return images, masks, names

    raise ValueError(f"Unsupported batch format: {type(batch)}")


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Converts model logits to binary masks.

    Supports:
    - one output channel: sigmoid threshold
    - two output channels: softmax argmax foreground class
    """

    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape (B, C, H, W), got {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits[:, 0])
        return (probs >= threshold).long()

    if logits.shape[1] == 2:
        probs = F.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).long()

    raise ValueError(f"Expected 1 or 2 output channels, got {logits.shape[1]}")


def prepare_gt_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a dataset mask to shape (B, H, W), binary.
    """

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[1] == 2:
        mask = torch.argmax(mask, dim=1)
    elif mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    return (mask > 0).long()


def column_heights(mask: np.ndarray) -> np.ndarray:
    """
    Computes vertical thickness for each column containing at least one foreground pixel.

    For each x-column:
    height(x) = max_y(mask[:, x]) - min_y(mask[:, x]) + 1

    Returns an array of heights in pixels.
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    mask = mask.astype(bool)
    heights: List[int] = []

    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x])[0]
        if ys.size > 0:
            heights.append(int(ys.max() - ys.min() + 1))

    return np.array(heights, dtype=np.float32)


def central_column_heights(mask: np.ndarray, central_fraction: float = 0.20) -> np.ndarray:
    """
    Computes column heights only around the horizontal center of the mask.

    This can be more clinically meaningful than the global maximum if the user
    wants a central tear meniscus height approximation.
    """

    mask = mask.astype(bool)
    xs = np.where(mask.any(axis=0))[0]

    if xs.size == 0:
        return np.array([], dtype=np.float32)

    x_min, x_max = int(xs.min()), int(xs.max())
    width = x_max - x_min + 1

    center = (x_min + x_max) / 2.0
    half_window = max(1, int(round(width * central_fraction / 2.0)))

    left = max(0, int(round(center - half_window)))
    right = min(mask.shape[1] - 1, int(round(center + half_window)))

    cropped = mask[:, left : right + 1]
    return column_heights(cropped)


def height_stats(mask: np.ndarray, central_fraction: float = 0.20) -> Dict[str, float]:
    """
    Computes height-related features from a binary mask.
    """

    mask = mask.astype(bool)
    area = int(mask.sum())

    if area == 0:
        return {
            "area_px": 0.0,
            "bbox_width_px": 0.0,
            "bbox_height_px": 0.0,
            "mean_height_px": 0.0,
            "median_height_px": 0.0,
            "max_height_px": 0.0,
            "p95_height_px": 0.0,
            "central_mean_height_px": 0.0,
            "central_median_height_px": 0.0,
        }

    ys, xs = np.where(mask)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)

    heights = column_heights(mask)
    central_heights = central_column_heights(mask, central_fraction=central_fraction)

    return {
        "area_px": float(area),
        "bbox_width_px": float(bbox_width),
        "bbox_height_px": float(bbox_height),
        "mean_height_px": float(np.mean(heights)) if heights.size else 0.0,
        "median_height_px": float(np.median(heights)) if heights.size else 0.0,
        "max_height_px": float(np.max(heights)) if heights.size else 0.0,
        "p95_height_px": float(np.percentile(heights, 95)) if heights.size else 0.0,
        "central_mean_height_px": float(np.mean(central_heights)) if central_heights.size else 0.0,
        "central_median_height_px": float(np.median(central_heights)) if central_heights.size else 0.0,
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_metric(rows: List[Dict[str, Any]], metric: str) -> Dict[str, float]:
    gt = np.array([float(r[f"gt_{metric}"]) for r in rows], dtype=np.float64)
    pred = np.array([float(r[f"pred_{metric}"]) for r in rows], dtype=np.float64)

    abs_error = np.abs(pred - gt)
    signed_error = pred - gt

    valid_relative = gt > 0
    if valid_relative.any():
        rel_abs_error = abs_error[valid_relative] / gt[valid_relative]
    else:
        rel_abs_error = np.array([], dtype=np.float64)

    return {
        "gt_mean": float(np.mean(gt)),
        "pred_mean": float(np.mean(pred)),
        "mae_px": float(np.mean(abs_error)),
        "median_ae_px": float(np.median(abs_error)),
        "bias_px": float(np.mean(signed_error)),
        "relative_mae_percent": float(np.mean(rel_abs_error) * 100.0) if rel_abs_error.size else float("nan"),
        "pearson_correlation": pearson_corr(gt, pred),
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--central-fraction", default=0.20, type=float)
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    safe_mkdir(args.output_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = UNetTrainer(config_path=str(args.config))
    dataloaders = trainer.build_dataloaders()

    if isinstance(dataloaders, dict):
        test_loader = dataloaders["test"]
    else:
        test_loader = dataloaders[-1]

    model = trainer.build_model()
    model.to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    sample_index = 0

    with torch.no_grad():
        for batch in test_loader:
            images, gt_masks, names = extract_batch(batch)
            images = images.to(device)
            gt_masks = prepare_gt_mask(gt_masks).cpu()

            logits = model(images)
            pred_masks = logits_to_mask(logits, threshold=args.threshold).cpu()

            for i in range(images.shape[0]):
                name = names[i] if i < len(names) else f"sample_{sample_index}"

                gt_np = gt_masks[i].numpy().astype(np.uint8)
                pred_np = pred_masks[i].numpy().astype(np.uint8)

                gt_stats = height_stats(gt_np, central_fraction=args.central_fraction)
                pred_stats = height_stats(pred_np, central_fraction=args.central_fraction)

                row: Dict[str, Any] = {
                    "sample_index": sample_index,
                    "name": str(name),
                }

                for key, value in gt_stats.items():
                    row[f"gt_{key}"] = value

                for key, value in pred_stats.items():
                    row[f"pred_{key}"] = value

                for key in gt_stats.keys():
                    row[f"error_{key}"] = pred_stats[key] - gt_stats[key]
                    row[f"abs_error_{key}"] = abs(pred_stats[key] - gt_stats[key])

                rows.append(row)
                sample_index += 1

    per_image_path = args.output_dir / "tmh_per_image.csv"
    save_csv(per_image_path, rows)

    height_metrics = [
        "bbox_height_px",
        "mean_height_px",
        "median_height_px",
        "max_height_px",
        "p95_height_px",
        "central_mean_height_px",
        "central_median_height_px",
    ]

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_images": len(rows),
        "threshold": args.threshold,
        "central_fraction": args.central_fraction,
        "metrics": {},
    }

    for metric in height_metrics:
        summary["metrics"][metric] = summarize_metric(rows, metric)

    summary_path = args.output_dir / "tmh_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("Saved:")
    print(f"  {per_image_path}")
    print(f"  {summary_path}")
    print()
    print("Main height summaries:")
    for metric in ["median_height_px", "p95_height_px", "central_mean_height_px", "bbox_height_px"]:
        s = summary["metrics"][metric]
        print(f"\n{metric}")
        print(f"  GT mean:              {s['gt_mean']:.3f} px")
        print(f"  Pred mean:            {s['pred_mean']:.3f} px")
        print(f"  MAE:                  {s['mae_px']:.3f} px")
        print(f"  Median AE:            {s['median_ae_px']:.3f} px")
        print(f"  Bias:                 {s['bias_px']:.3f} px")
        print(f"  Relative MAE:         {s['relative_mae_percent']:.2f} %")
        print(f"  Pearson correlation:  {s['pearson_correlation']:.4f}")


if __name__ == "__main__":
    main()