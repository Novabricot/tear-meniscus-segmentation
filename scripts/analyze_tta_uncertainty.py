#!/usr/bin/env python3
"""
Test-time augmentation uncertainty analysis for U-Net segmentation models.

Goal:
    Estimate model uncertainty by running several slightly modified versions
    of each test image and measuring the variability of the predicted foreground
    probability.

Then, compare uncertainty inside:
    - certain positive region: eroded GT
    - uncertain boundary region: dilated GT - eroded GT
    - certain negative region: outside dilated GT

This script does not train anything.
It only evaluates an existing checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from scripts.train_unet import UNetTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TTA uncertainty by annotation region.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Boundary radii in pixels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def binary_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    mask: tensor of shape (B, H, W), values 0/1.
    Returns dilated binary mask.
    """
    if radius <= 0:
        return mask.bool()

    x = mask.float().unsqueeze(1)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)
    return dilated.squeeze(1) > 0.5


def binary_erosion(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    mask: tensor of shape (B, H, W), values 0/1.
    Erosion is implemented as the complement of dilation of the complement.
    """
    if radius <= 0:
        return mask.bool()

    inv = 1.0 - mask.float()
    dilated_inv = F.max_pool2d(
        inv.unsqueeze(1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    eroded = 1.0 - dilated_inv.squeeze(1)
    return eroded > 0.5


def apply_tta(images: torch.Tensor) -> List[Tuple[str, torch.Tensor, str]]:
    """
    Return list of augmented images.

    Each item:
        (name, augmented_images, inverse_transform_name)

    Only horizontal flip needs spatial inverse transform.
    Brightness/contrast/noise do not need inverse spatial correction.
    """
    tta_items: List[Tuple[str, torch.Tensor, str]] = []

    tta_items.append(("identity", images, "identity"))

    tta_items.append(("hflip", torch.flip(images, dims=[3]), "hflip"))

    # Small intensity perturbations directly in tensor space.
    # These are not meant as normal data augmentation for training, but as
    # simple inference-time perturbations to estimate prediction stability.
    tta_items.append(("brightness_plus", images + 0.05, "identity"))
    tta_items.append(("brightness_minus", images - 0.05, "identity"))

    mean = images.mean(dim=(2, 3), keepdim=True)
    tta_items.append(("contrast_plus", (images - mean) * 1.05 + mean, "identity"))
    tta_items.append(("contrast_minus", (images - mean) * 0.95 + mean, "identity"))

    noise = torch.randn_like(images) * 0.02
    tta_items.append(("noise", images + noise, "identity"))

    return tta_items


def invert_prediction(prob: torch.Tensor, inverse_transform_name: str) -> torch.Tensor:
    """
    prob: tensor of shape (B, H, W)
    """
    if inverse_transform_name == "hflip":
        return torch.flip(prob, dims=[2])
    return prob


@torch.no_grad()
def predict_tta_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        probs: tensor of shape (T, B, H, W), foreground probabilities.
    """
    probs = []

    for _name, aug_images, inverse_name in apply_tta(images):
        logits = model(aug_images)
        prob_fg = torch.softmax(logits, dim=1)[:, 1, :, :]
        prob_fg = invert_prediction(prob_fg, inverse_name)
        probs.append(prob_fg)

    return torch.stack(probs, dim=0)


def safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute mean of values inside mask.
    Returns nan if mask is empty.
    """
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


def analyze_batch(
    uncertainty: torch.Tensor,
    targets: torch.Tensor,
    radii: List[int],
) -> List[Dict[str, float]]:
    """
    uncertainty: (B, H, W)
    targets: (B, H, W)
    """
    batch_rows: List[Dict[str, float]] = []
    gt = targets.bool()

    for radius in radii:
        eroded = binary_erosion(gt, radius)
        dilated = binary_dilation(gt, radius)

        certain_positive = eroded
        uncertain_boundary = dilated & (~eroded)
        certain_negative = ~dilated

        for i in range(gt.shape[0]):
            row = {
                "radius": radius,
                "mean_uncertainty_all": uncertainty[i].mean().item(),
                "mean_uncertainty_certain_positive": safe_mean(
                    uncertainty[i], certain_positive[i]
                ),
                "mean_uncertainty_uncertain_boundary": safe_mean(
                    uncertainty[i], uncertain_boundary[i]
                ),
                "mean_uncertainty_certain_negative": safe_mean(
                    uncertainty[i], certain_negative[i]
                ),
                "pixels_certain_positive": int(certain_positive[i].sum().item()),
                "pixels_uncertain_boundary": int(uncertain_boundary[i].sum().item()),
                "pixels_certain_negative": int(certain_negative[i].sum().item()),
            }
            batch_rows.append(row)

    return batch_rows


def mean_ignore_nan(values: List[float]) -> float:
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = tensor[~torch.isnan(tensor)]
    if tensor.numel() == 0:
        return float("nan")
    return tensor.mean().item()


def summarize_rows(rows: List[Dict[str, float]], radii: List[int]) -> List[Dict[str, float]]:
    summary = []

    for radius in radii:
        radius_rows = [row for row in rows if row["radius"] == radius]

        summary_row = {
            "radius": radius,
            "mean_uncertainty_all": mean_ignore_nan(
                [row["mean_uncertainty_all"] for row in radius_rows]
            ),
            "mean_uncertainty_certain_positive": mean_ignore_nan(
                [row["mean_uncertainty_certain_positive"] for row in radius_rows]
            ),
            "mean_uncertainty_uncertain_boundary": mean_ignore_nan(
                [row["mean_uncertainty_uncertain_boundary"] for row in radius_rows]
            ),
            "mean_uncertainty_certain_negative": mean_ignore_nan(
                [row["mean_uncertainty_certain_negative"] for row in radius_rows]
            ),
        }

        # Helpful ratios for interpretation.
        cp = summary_row["mean_uncertainty_certain_positive"]
        ub = summary_row["mean_uncertainty_uncertain_boundary"]
        cn = summary_row["mean_uncertainty_certain_negative"]

        summary_row["boundary_over_certain_positive"] = ub / cp if cp and cp == cp else float("nan")
        summary_row["boundary_over_certain_negative"] = ub / cn if cn and cn == cn else float("nan")

        summary.append(summary_row)

    return summary


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_make_plot(summary_rows: List[Dict[str, float]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not import matplotlib, skipping plot: {exc}")
        return

    radii = [row["radius"] for row in summary_rows]
    cp = [row["mean_uncertainty_certain_positive"] for row in summary_rows]
    ub = [row["mean_uncertainty_uncertain_boundary"] for row in summary_rows]
    cn = [row["mean_uncertainty_certain_negative"] for row in summary_rows]

    fig_path = output_dir / "tta_uncertainty_by_region.png"

    plt.figure(figsize=(8, 5))
    plt.plot(radii, cp, marker="o", label="Certain positive")
    plt.plot(radii, ub, marker="o", label="Uncertain boundary")
    plt.plot(radii, cn, marker="o", label="Certain negative")
    plt.xlabel("Boundary radius (pixels)")
    plt.ylabel("Mean TTA uncertainty")
    plt.title("TTA uncertainty by annotation region")
    plt.xticks(radii)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved figure: {fig_path}")


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    trainer = UNetTrainer(config_path=str(config_path))
    model = trainer.build_model()
    model.to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    _train_loader, _val_loader, test_loader = trainer.build_dataloaders()

    all_rows: List[Dict[str, float]] = []

    for batch in tqdm(test_loader, desc="TTA uncertainty"):
        if isinstance(batch, dict):
            images = batch["image"]
            targets = batch["mask"]
        else:
            images, targets = batch[:2]

        images = images.to(device)
        targets = targets.to(device)

        if targets.ndim == 4:
            targets = targets.squeeze(1)

        tta_probs = predict_tta_probabilities(model, images)
        uncertainty = tta_probs.std(dim=0)

        rows = analyze_batch(uncertainty=uncertainty, targets=targets, radii=args.radii)
        all_rows.extend(rows)

    summary_rows = summarize_rows(all_rows, args.radii)

    per_image_path = output_dir / "tta_uncertainty_per_image.csv"
    summary_csv_path = output_dir / "tta_uncertainty_summary.csv"
    summary_yaml_path = output_dir / "tta_uncertainty_summary.yaml"
    summary_json_path = output_dir / "tta_uncertainty_summary.json"

    write_csv(per_image_path, all_rows)
    write_csv(summary_csv_path, summary_rows)

    with summary_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary_rows, f, sort_keys=False)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    try_make_plot(summary_rows, output_dir)

    print(f"Saved per-image CSV: {per_image_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary YAML: {summary_yaml_path}")

    print("\nSummary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()