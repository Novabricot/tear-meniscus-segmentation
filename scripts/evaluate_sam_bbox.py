import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_image_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_mask_binary(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    return mask > 0


def mask_to_box(mask: np.ndarray, padding: int = 5):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = max(int(xs.min()) - padding, 0)
    y_min = max(int(ys.min()) - padding, 0)
    x_max = min(int(xs.max()) + padding, mask.shape[1] - 1)
    y_max = min(int(ys.max()) + padding, mask.shape[0] - 1)

    return np.array([x_min, y_min, x_max, y_max])


def compute_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "miou": float(iou),
    }


def find_pairs(dataset_path: Path):
    image_dir = dataset_path / "test" / "images"
    mask_dir = dataset_path / "test" / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    pairs = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue

        mask_path = mask_dir / image_path.name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {dataset_path}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM with oracle bounding boxes derived from ground-truth masks."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-padding", type=int, default=5)
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    config = load_config(Path(args.config))
    dataset_path = Path(config["data"]["dataset_path"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = output_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    pairs = find_pairs(dataset_path)

    all_metrics = []
    skipped_empty_masks = 0

    for image_path, mask_path in tqdm(pairs, desc="Evaluating SAM"):
        image = read_image_rgb(image_path)
        gt_mask = read_mask_binary(mask_path)

        box = mask_to_box(gt_mask, padding=args.box_padding)

        if box is None:
            skipped_empty_masks += 1
            continue

        predictor.set_image(image)

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        pred_mask = masks[best_idx]

        metrics = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)

        if args.save_predictions:
            out_path = pred_dir / image_path.name
            cv2.imwrite(str(out_path), (pred_mask.astype(np.uint8) * 255))

    if not all_metrics:
        raise RuntimeError("No valid masks were evaluated.")

    mean_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in ["f1", "precision", "recall", "miou"]
    }

    mean_metrics["num_images"] = len(all_metrics)
    mean_metrics["skipped_empty_masks"] = skipped_empty_masks
    mean_metrics["model_type"] = args.model_type
    mean_metrics["checkpoint"] = str(args.checkpoint)
    mean_metrics["config"] = str(args.config)
    mean_metrics["dataset_path"] = str(dataset_path)
    mean_metrics["box_padding"] = args.box_padding
    mean_metrics["prompt_type"] = "oracle_gt_bbox"

    output_path = output_dir / "test_metrics.yaml"

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mean_metrics, f, sort_keys=False)

    print("SAM evaluation complete")
    print(f"Saved metrics to: {output_path}")
    print(mean_metrics)


if __name__ == "__main__":
    main()