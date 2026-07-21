import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

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


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)

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


def compute_strict_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall + epsilon)
    )

    union = np.logical_or(pred, gt).sum()
    iou = tp / (union + epsilon)

    return {
        "strict_f1": float(f1),
        "strict_precision": float(precision),
        "strict_recall": float(recall),
        "strict_miou": float(iou),
        "strict_tp": int(tp),
        "strict_fp": int(fp),
        "strict_fn": int(fn),
    }


def compute_relaxed_f1(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    radius: int,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    pred = prediction.astype(np.uint8)
    gt = ground_truth.astype(np.uint8)

    if radius == 0:
        strict = compute_strict_metrics(
            prediction=pred,
            ground_truth=gt,
            epsilon=epsilon,
        )

        return {
            "radius": radius,
            "relaxed_f1": strict["strict_f1"],
            "relaxed_precision": strict["strict_precision"],
            "relaxed_recall": strict["strict_recall"],
            "accepted_pred_pixels": strict["strict_tp"],
            "accepted_gt_pixels": strict["strict_tp"],
            "pred_positive_pixels": int(pred.sum()),
            "gt_positive_pixels": int(gt.sum()),
        }

    dilated_gt = dilate_mask(
        mask=gt,
        radius=radius,
    ).astype(bool)

    dilated_pred = dilate_mask(
        mask=pred,
        radius=radius,
    ).astype(bool)

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    pred_positive_pixels = int(pred_bool.sum())
    gt_positive_pixels = int(gt_bool.sum())

    accepted_pred_pixels = int(
        np.logical_and(
            pred_bool,
            dilated_gt,
        ).sum()
    )

    accepted_gt_pixels = int(
        np.logical_and(
            gt_bool,
            dilated_pred,
        ).sum()
    )

    relaxed_precision = (
        accepted_pred_pixels
        / (pred_positive_pixels + epsilon)
    )

    relaxed_recall = (
        accepted_gt_pixels
        / (gt_positive_pixels + epsilon)
    )

    relaxed_f1 = (
        2.0
        * relaxed_precision
        * relaxed_recall
        / (relaxed_precision + relaxed_recall + epsilon)
    )

    return {
        "radius": radius,
        "relaxed_f1": float(relaxed_f1),
        "relaxed_precision": float(relaxed_precision),
        "relaxed_recall": float(relaxed_recall),
        "accepted_pred_pixels": accepted_pred_pixels,
        "accepted_gt_pixels": accepted_gt_pixels,
        "pred_positive_pixels": pred_positive_pixels,
        "gt_positive_pixels": gt_positive_pixels,
    }


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    grouped = {}

    for row in rows:
        radius = int(row["radius"])
        grouped.setdefault(radius, []).append(row)

    summary_rows = []

    metric_fields = [
        "strict_f1",
        "strict_precision",
        "strict_recall",
        "strict_miou",
        "relaxed_f1",
        "relaxed_precision",
        "relaxed_recall",
    ]

    count_fields = [
        "strict_tp",
        "strict_fp",
        "strict_fn",
        "accepted_pred_pixels",
        "accepted_gt_pixels",
        "pred_positive_pixels",
        "gt_positive_pixels",
    ]

    epsilon = 1e-7

    for radius, radius_rows in sorted(grouped.items()):
        summary = {
            "radius": radius,
            "n_images": len(radius_rows),
        }

        for field in metric_fields:
            values = np.array(
                [
                    float(row[field])
                    for row in radius_rows
                ],
                dtype=np.float64,
            )

            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_std"] = float(values.std())

        for field in count_fields:
            summary[field] = int(
                sum(int(row[field]) for row in radius_rows)
            )

        global_relaxed_precision = (
            summary["accepted_pred_pixels"]
            / (summary["pred_positive_pixels"] + epsilon)
        )

        global_relaxed_recall = (
            summary["accepted_gt_pixels"]
            / (summary["gt_positive_pixels"] + epsilon)
        )

        global_relaxed_f1 = (
            2.0
            * global_relaxed_precision
            * global_relaxed_recall
            / (
                global_relaxed_precision
                + global_relaxed_recall
                + epsilon
            )
        )

        global_strict_precision = (
            summary["strict_tp"]
            / (
                summary["strict_tp"]
                + summary["strict_fp"]
                + epsilon
            )
        )

        global_strict_recall = (
            summary["strict_tp"]
            / (
                summary["strict_tp"]
                + summary["strict_fn"]
                + epsilon
            )
        )

        global_strict_f1 = (
            2.0
            * global_strict_precision
            * global_strict_recall
            / (
                global_strict_precision
                + global_strict_recall
                + epsilon
            )
        )

        summary["global_strict_f1"] = float(global_strict_f1)
        summary["global_strict_precision"] = float(global_strict_precision)
        summary["global_strict_recall"] = float(global_strict_recall)

        summary["global_relaxed_f1"] = float(global_relaxed_f1)
        summary["global_relaxed_precision"] = float(global_relaxed_precision)
        summary["global_relaxed_recall"] = float(global_relaxed_recall)

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


def run_evaluation(
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
            desc="Evaluate relaxed boundary metrics",
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

                strict_metrics = compute_strict_metrics(
                    prediction=pred_mask,
                    ground_truth=gt_mask,
                )

                for radius in radii:
                    relaxed_metrics = compute_relaxed_f1(
                        prediction=pred_mask,
                        ground_truth=gt_mask,
                        radius=radius,
                    )

                    row = {
                        "image_index": image_index,
                        **strict_metrics,
                        **relaxed_metrics,
                    }

                    rows.append(row)

                image_index += 1

    summary_rows = summarize_rows(rows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_csv = output_dir / "relaxed_boundary_metrics_per_image.csv"
    summary_csv = output_dir / "relaxed_boundary_metrics_summary.csv"
    summary_yaml = output_dir / "relaxed_boundary_metrics_summary.yaml"

    write_csv(rows, per_image_csv)
    write_csv(summary_rows, summary_csv)

    yaml_summary = {
        "description": (
            "Tolerance-aware relaxed F1 evaluation. "
            "A predicted positive pixel is accepted if it falls within "
            "a dilated ground truth. A ground-truth positive pixel is "
            "accepted if it falls within a dilated prediction."
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
            "Evaluate strict and tolerance-aware relaxed boundary metrics."
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
        default=[0, 1, 2, 3],
        help=(
            "Tolerance radii. Radius 0 is equivalent to strict F1."
        ),
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

    run_evaluation(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        threshold=args.threshold,
        radii=args.radii,
    )


if __name__ == "__main__":
    main()