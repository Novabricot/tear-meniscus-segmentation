import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image


logger = logging.getLogger(__name__)


DEFAULT_CENTERS = [
    "Colour1",
    "Colour2",
    "Infrared1",
    "Infrared2",
    "Infrared3",
]


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_binary_mask(mask_path: Path, threshold: int = 127) -> np.ndarray:
    with Image.open(mask_path) as image:
        image = image.convert("L")
        array = np.array(image)

    mask = array > threshold

    return mask.astype(np.uint8)


def find_masks(raw_root: Path, center_name: str) -> List[Path]:
    mask_dir = raw_root / center_name / "Label"

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    mask_paths = [
        path
        for path in sorted(mask_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not mask_paths:
        raise RuntimeError(f"No mask files found in {mask_dir}")

    return mask_paths


def perturb_mask(
    mask: np.ndarray,
    operation: str,
    radius: int,
) -> np.ndarray:
    if radius < 1:
        raise ValueError("Radius must be >= 1")

    kernel_size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    if operation == "erosion":
        perturbed = cv2.erode(
            mask,
            kernel,
            iterations=1,
        )

    elif operation == "dilation":
        perturbed = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

    elif operation == "opening":
        perturbed = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

    elif operation == "closing":
        perturbed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    else:
        raise ValueError(f"Unknown operation: {operation}")

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

    intersection = true_positive

    union = np.logical_or(gt, pred).sum()

    iou = intersection / (union + epsilon)

    gt_area = gt.sum()
    pred_area = pred.sum()

    area_ratio = pred_area / (gt_area + epsilon)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "miou": float(iou),
        "gt_area": float(gt_area),
        "perturbed_area": float(pred_area),
        "area_ratio": float(area_ratio),
    }


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int], List[Dict]] = {}

    for row in rows:
        key = (
            row["center"],
            row["operation"],
            int(row["radius"]),
        )

        grouped.setdefault(key, []).append(row)

    summary_rows = []

    metric_names = [
        "f1",
        "precision",
        "recall",
        "miou",
        "gt_area",
        "perturbed_area",
        "area_ratio",
    ]

    for (center, operation, radius), group_rows in sorted(grouped.items()):
        summary = {
            "center": center,
            "operation": operation,
            "radius": radius,
            "n_masks": len(group_rows),
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


def add_modality_summary(summary_rows: List[Dict]) -> List[Dict]:
    modality_groups: Dict[Tuple[str, str, int], List[Dict]] = {}

    for row in summary_rows:
        center = row["center"]

        if center.startswith("Colour"):
            modality = "Colour"
        elif center.startswith("Infrared"):
            modality = "Infrared"
        else:
            modality = "Unknown"

        key = (
            modality,
            row["operation"],
            int(row["radius"]),
        )

        modality_groups.setdefault(key, []).append(row)

    modality_rows = []

    metric_names = [
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "miou_mean",
        "gt_area_mean",
        "perturbed_area_mean",
        "area_ratio_mean",
    ]

    for (modality, operation, radius), group_rows in sorted(
        modality_groups.items()
    ):
        total_masks = sum(
            int(row["n_masks"])
            for row in group_rows
        )

        modality_summary = {
            "modality": modality,
            "operation": operation,
            "radius": radius,
            "n_centers": len(group_rows),
            "n_masks": total_masks,
        }

        for metric_name in metric_names:
            values = np.array(
                [
                    float(row[metric_name])
                    for row in group_rows
                ],
                dtype=np.float64,
            )

            modality_summary[metric_name] = float(values.mean())

        modality_rows.append(modality_summary)

    return modality_rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how F1 and mIoU change when ground-truth "
            "segmentation masks are slightly eroded or dilated."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/Open DataSet2",
        help="Path to the raw dataset root.",
    )

    parser.add_argument(
        "--centers",
        nargs="+",
        default=DEFAULT_CENTERS,
        help="Centers to analyze.",
    )

    parser.add_argument(
        "--operations",
        nargs="+",
        default=[
            "erosion",
            "dilation",
        ],
        choices=[
            "erosion",
            "dilation",
            "opening",
            "closing",
        ],
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
        help="Pixel radii for morphological perturbations.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--output-dir",
        default="results/annotation_sensitivity",
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

    raw_root = Path(args.input)
    output_dir = Path(args.output_dir)

    all_rows = []

    for center_name in args.centers:
        mask_paths = find_masks(
            raw_root=raw_root,
            center_name=center_name,
        )

        logger.info(
            "%s: %d masks",
            center_name,
            len(mask_paths),
        )

        for mask_path in mask_paths:
            original_mask = load_binary_mask(
                mask_path=mask_path,
                threshold=args.threshold,
            )

            for operation in args.operations:
                for radius in args.radii:
                    perturbed_mask = perturb_mask(
                        mask=original_mask,
                        operation=operation,
                        radius=radius,
                    )

                    metrics = compute_metrics(
                        ground_truth=original_mask,
                        prediction=perturbed_mask,
                    )

                    row = {
                        "center": center_name,
                        "mask_name": mask_path.name,
                        "operation": operation,
                        "radius": radius,
                        **metrics,
                    }

                    all_rows.append(row)

    summary_rows = summarize_rows(all_rows)
    modality_rows = add_modality_summary(summary_rows)

    detailed_csv = output_dir / "boundary_sensitivity_per_mask.csv"
    summary_csv = output_dir / "boundary_sensitivity_by_center.csv"
    modality_csv = output_dir / "boundary_sensitivity_by_modality.csv"
    summary_yaml = output_dir / "boundary_sensitivity_summary.yaml"

    write_csv(all_rows, detailed_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(modality_rows, modality_csv)

    yaml_summary = {
        "description": (
            "Boundary sensitivity analysis. Original ground-truth "
            "masks are compared against eroded/dilated versions."
        ),
        "input": str(raw_root),
        "centers": args.centers,
        "operations": args.operations,
        "radii": args.radii,
        "threshold": args.threshold,
        "outputs": {
            "per_mask_csv": str(detailed_csv),
            "by_center_csv": str(summary_csv),
            "by_modality_csv": str(modality_csv),
        },
        "by_center": summary_rows,
        "by_modality": modality_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            yaml_summary,
            file,
            sort_keys=False,
        )

    logger.info("Saved per-mask results to %s", detailed_csv)
    logger.info("Saved center summary to %s", summary_csv)
    logger.info("Saved modality summary to %s", modality_csv)
    logger.info("Saved YAML summary to %s", summary_yaml)

    logger.info("Boundary sensitivity analysis complete")


if __name__ == "__main__":
    main()