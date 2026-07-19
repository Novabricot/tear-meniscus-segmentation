import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from train_unet import UNetTrainer, ensure_mask_channel


logger = logging.getLogger(__name__)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    logger.info("Loaded checkpoint: %s", checkpoint_path)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return (eroded > 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    dilated = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return (dilated > 0).astype(np.uint8)


def build_uncertainty_zones(
    gt_mask: np.ndarray,
    radius: int,
) -> Dict[str, np.ndarray]:
    gt = gt_mask.astype(np.uint8)

    eroded = erode_mask(gt, radius)
    dilated = dilate_mask(gt, radius)

    certain_positive = eroded.astype(bool)

    uncertain_boundary = np.logical_and(
        dilated.astype(bool),
        np.logical_not(eroded.astype(bool)),
    )

    certain_negative = np.logical_not(dilated.astype(bool))

    return {
        "certain_positive": certain_positive,
        "uncertain_boundary": uncertain_boundary,
        "certain_negative": certain_negative,
        "eroded": eroded.astype(bool),
        "dilated": dilated.astype(bool),
    }


def compute_zone_counts(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    zones: Dict[str, np.ndarray],
) -> Dict[str, float]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    false_positive = np.logical_and(pred, np.logical_not(gt))
    false_negative = np.logical_and(np.logical_not(pred), gt)
    error = np.logical_xor(pred, gt)

    certain_positive = zones["certain_positive"]
    uncertain_boundary = zones["uncertain_boundary"]
    certain_negative = zones["certain_negative"]

    total_pixels = gt.size

    gt_positive_pixels = int(gt.sum())
    pred_positive_pixels = int(pred.sum())

    certain_positive_pixels = int(certain_positive.sum())
    uncertain_boundary_pixels = int(uncertain_boundary.sum())
    certain_negative_pixels = int(certain_negative.sum())

    fp_total = int(false_positive.sum())
    fn_total = int(false_negative.sum())
    error_total = int(error.sum())

    error_certain_positive = int(
        np.logical_and(error, certain_positive).sum()
    )

    error_uncertain_boundary = int(
        np.logical_and(error, uncertain_boundary).sum()
    )

    error_certain_negative = int(
        np.logical_and(error, certain_negative).sum()
    )

    fp_uncertain_boundary = int(
        np.logical_and(false_positive, uncertain_boundary).sum()
    )

    fn_uncertain_boundary = int(
        np.logical_and(false_negative, uncertain_boundary).sum()
    )

    fp_certain_negative = int(
        np.logical_and(false_positive, certain_negative).sum()
    )

    fn_certain_positive = int(
        np.logical_and(false_negative, certain_positive).sum()
    )

    certain_region_errors = (
        error_certain_positive + error_certain_negative
    )

    epsilon = 1e-7

    return {
        "total_pixels": total_pixels,
        "gt_positive_pixels": gt_positive_pixels,
        "pred_positive_pixels": pred_positive_pixels,
        "certain_positive_pixels": certain_positive_pixels,
        "uncertain_boundary_pixels": uncertain_boundary_pixels,
        "certain_negative_pixels": certain_negative_pixels,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "error_total": error_total,
        "error_certain_positive": error_certain_positive,
        "error_uncertain_boundary": error_uncertain_boundary,
        "error_certain_negative": error_certain_negative,
        "fp_uncertain_boundary": fp_uncertain_boundary,
        "fn_uncertain_boundary": fn_uncertain_boundary,
        "fp_certain_negative": fp_certain_negative,
        "fn_certain_positive": fn_certain_positive,
        "certain_region_errors": certain_region_errors,
        "error_rate_total": error_total / (total_pixels + epsilon),
        "error_rate_certain_positive": (
            error_certain_positive
            / (certain_positive_pixels + epsilon)
        ),
        "error_rate_uncertain_boundary": (
            error_uncertain_boundary
            / (uncertain_boundary_pixels + epsilon)
        ),
        "error_rate_certain_negative": (
            error_certain_negative
            / (certain_negative_pixels + epsilon)
        ),
        "pct_errors_in_uncertain_boundary": (
            error_uncertain_boundary / (error_total + epsilon)
        ),
        "pct_errors_in_certain_regions": (
            certain_region_errors / (error_total + epsilon)
        ),
        "pct_false_positives_in_uncertain_boundary": (
            fp_uncertain_boundary / (fp_total + epsilon)
        ),
        "pct_false_negatives_in_uncertain_boundary": (
            fn_uncertain_boundary / (fn_total + epsilon)
        ),
    }


def summarize_by_radius(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = {}

    for row in rows:
        radius = int(row["radius"])
        grouped.setdefault(radius, []).append(row)

    count_fields = [
        "total_pixels",
        "gt_positive_pixels",
        "pred_positive_pixels",
        "certain_positive_pixels",
        "uncertain_boundary_pixels",
        "certain_negative_pixels",
        "fp_total",
        "fn_total",
        "error_total",
        "error_certain_positive",
        "error_uncertain_boundary",
        "error_certain_negative",
        "fp_uncertain_boundary",
        "fn_uncertain_boundary",
        "fp_certain_negative",
        "fn_certain_positive",
        "certain_region_errors",
    ]

    mean_fields = [
        "error_rate_total",
        "error_rate_certain_positive",
        "error_rate_uncertain_boundary",
        "error_rate_certain_negative",
        "pct_errors_in_uncertain_boundary",
        "pct_errors_in_certain_regions",
        "pct_false_positives_in_uncertain_boundary",
        "pct_false_negatives_in_uncertain_boundary",
    ]

    summary_rows = []

    epsilon = 1e-7

    for radius, group_rows in sorted(grouped.items()):
        summary = {
            "radius": radius,
            "n_images": len(group_rows),
        }

        for field in count_fields:
            summary[field] = int(
                sum(int(row[field]) for row in group_rows)
            )

        total_errors = summary["error_total"]
        total_fp = summary["fp_total"]
        total_fn = summary["fn_total"]

        summary["global_pct_errors_in_uncertain_boundary"] = (
            summary["error_uncertain_boundary"]
            / (total_errors + epsilon)
        )

        summary["global_pct_errors_in_certain_regions"] = (
            summary["certain_region_errors"]
            / (total_errors + epsilon)
        )

        summary["global_pct_false_positives_in_uncertain_boundary"] = (
            summary["fp_uncertain_boundary"]
            / (total_fp + epsilon)
        )

        summary["global_pct_false_negatives_in_uncertain_boundary"] = (
            summary["fn_uncertain_boundary"]
            / (total_fn + epsilon)
        )

        summary["global_error_rate_total"] = (
            total_errors
            / (summary["total_pixels"] + epsilon)
        )

        summary["global_error_rate_certain_positive"] = (
            summary["error_certain_positive"]
            / (summary["certain_positive_pixels"] + epsilon)
        )

        summary["global_error_rate_uncertain_boundary"] = (
            summary["error_uncertain_boundary"]
            / (summary["uncertain_boundary_pixels"] + epsilon)
        )

        summary["global_error_rate_certain_negative"] = (
            summary["error_certain_negative"]
            / (summary["certain_negative_pixels"] + epsilon)
        )

        for field in mean_fields:
            values = np.array(
                [float(row[field]) for row in group_rows],
                dtype=np.float64,
            )

            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_std"] = float(values.std())

        summary_rows.append(summary)

    return summary_rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    threshold: float,
    radii: List[int],
) -> None:
    trainer = UNetTrainer(
        config_path=str(config_path),
    )

    _, _, test_loader = trainer.build_dataloaders()

    model = trainer.build_model()

    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=trainer.device,
    )

    model.eval()

    rows = []

    image_index = 0

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc="Analyze uncertainty zones",
        ):
            images = batch["image"].to(
                trainer.device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                trainer.device,
                non_blocking=True,
            )

            masks = ensure_mask_channel(masks)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= threshold
            ).detach().cpu().numpy()

            ground_truths = (
                masks >= 0.5
            ).detach().cpu().numpy()

            batch_size = predictions.shape[0]

            for batch_index in range(batch_size):
                pred_mask = predictions[
                    batch_index,
                    0,
                ].astype(np.uint8)

                gt_mask = ground_truths[
                    batch_index,
                    0,
                ].astype(np.uint8)

                for radius in radii:
                    zones = build_uncertainty_zones(
                        gt_mask=gt_mask,
                        radius=radius,
                    )

                    counts = compute_zone_counts(
                        prediction=pred_mask,
                        ground_truth=gt_mask,
                        zones=zones,
                    )

                    row = {
                        "image_index": image_index,
                        "radius": radius,
                        **counts,
                    }

                    rows.append(row)

                image_index += 1

    summary_rows = summarize_by_radius(rows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_csv = output_dir / "uncertainty_zones_per_image.csv"
    summary_csv = output_dir / "uncertainty_zones_summary.csv"
    summary_yaml = output_dir / "uncertainty_zones_summary.yaml"

    write_csv(rows, per_image_csv)
    write_csv(summary_rows, summary_csv)

    yaml_summary = {
        "description": (
            "Analysis of model errors inside certain positive, "
            "uncertain boundary, and certain negative regions. "
            "The uncertain boundary is defined as dilation(GT) - erosion(GT)."
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "radii": radii,
        "outputs": {
            "per_image_csv": str(per_image_csv),
            "summary_csv": str(summary_csv),
        },
        "summary": summary_rows,
    }

    with summary_yaml.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            yaml_summary,
            file,
            sort_keys=False,
        )

    logger.info("Saved per-image results to %s", per_image_csv)
    logger.info("Saved summary results to %s", summary_csv)
    logger.info("Saved YAML summary to %s", summary_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether model errors occur mainly inside "
            "an uncertain boundary region around the ground truth."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    run_analysis(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        threshold=args.threshold,
        radii=args.radii,
    )


if __name__ == "__main__":
    main()