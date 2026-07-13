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

    logger.info(
        "Loaded checkpoint: %s",
        checkpoint_path,
    )


def perturb_mask(
    mask: np.ndarray,
    operation: str,
    radius: int,
) -> np.ndarray:
    if operation == "original":
        return mask.astype(np.uint8)

    if radius < 1:
        raise ValueError(
            "Radius must be >= 1 for erosion/dilation."
        )

    kernel_size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    if operation == "erosion":
        perturbed = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
        )

    elif operation == "dilation":
        perturbed = cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
        )

    else:
        raise ValueError(
            f"Unknown operation: {operation}"
        )

    return (perturbed > 0).astype(np.uint8)


def compute_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    true_positive = np.logical_and(gt, pred).sum()
    false_positive = np.logical_and(~gt, pred).sum()
    false_negative = np.logical_and(gt, ~pred).sum()

    precision = true_positive / (
        true_positive + false_positive + epsilon
    )

    recall = true_positive / (
        true_positive + false_negative + epsilon
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall + epsilon)
    )

    union = np.logical_or(gt, pred).sum()

    iou = true_positive / (union + epsilon)

    gt_area = gt.sum()
    pred_area = pred.sum()

    area_ratio = pred_area / (gt_area + epsilon)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "miou": float(iou),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "pred_to_gt_area_ratio": float(area_ratio),
    }


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}

    for row in rows:
        key = (
            row["operation"],
            int(row["radius"]),
        )

        grouped.setdefault(key, []).append(row)

    metric_names = [
        "f1",
        "precision",
        "recall",
        "miou",
        "gt_area",
        "pred_area",
        "pred_to_gt_area_ratio",
    ]

    summary_rows = []

    operation_order = {
        "original": 0,
        "erosion": 1,
        "dilation": 2,
    }

    for (operation, radius), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            operation_order.get(item[0][0], 99),
            item[0][1],
        ),
    ):
        summary = {
            "operation": operation,
            "radius": radius,
            "n_images": len(group_rows),
        }

        for metric_name in metric_names:
            values = np.array(
                [
                    float(row[metric_name])
                    for row in group_rows
                ],
                dtype=np.float64,
            )

            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())

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

    operations = [
        ("original", 0),
    ]

    for radius in radii:
        operations.append(
            ("erosion", radius)
        )

    for radius in radii:
        operations.append(
            ("dilation", radius)
        )

    image_index = 0

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc="Evaluate prediction boundary sensitivity",
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

                for operation, radius in operations:
                    perturbed_gt = perturb_mask(
                        mask=gt_mask,
                        operation=operation,
                        radius=radius,
                    )

                    metrics = compute_metrics(
                        ground_truth=perturbed_gt,
                        prediction=pred_mask,
                    )

                    row = {
                        "image_index": image_index,
                        "operation": operation,
                        "radius": radius,
                        **metrics,
                    }

                    rows.append(row)

                image_index += 1

    summary_rows = summarize_rows(rows)

    per_image_csv = (
        output_dir
        / "prediction_boundary_sensitivity_per_image.csv"
    )

    summary_csv = (
        output_dir
        / "prediction_boundary_sensitivity_summary.csv"
    )

    summary_yaml = (
        output_dir
        / "prediction_boundary_sensitivity_summary.yaml"
    )

    write_csv(rows, per_image_csv)
    write_csv(summary_rows, summary_csv)

    yaml_summary = {
        "description": (
            "Model prediction evaluated against original, eroded, "
            "and dilated ground-truth masks."
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_yaml.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            yaml_summary,
            file,
            sort_keys=False,
        )

    logger.info(
        "Saved per-image results to %s",
        per_image_csv,
    )

    logger.info(
        "Saved summary results to %s",
        summary_csv,
    )

    logger.info(
        "Saved YAML summary to %s",
        summary_yaml,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a model prediction is closer to "
            "the original, eroded, or dilated ground-truth mask."
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
        default=[
            1,
            2,
            3,
        ],
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