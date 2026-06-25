import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
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

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


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


def load_medsam_model(checkpoint_path: Path, model_type: str, device: str):
    model = sam_model_registry[model_type](checkpoint=None)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    # Some checkpoints are raw state_dict, others are {"model": state_dict}.
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    if missing_keys:
        print(f"Warning: missing keys while loading checkpoint: {len(missing_keys)}")

    if unexpected_keys:
        print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected_keys)}")

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_medsam_mask(model, image: np.ndarray, box: np.ndarray, device: str, threshold: float):
    original_size = image.shape[:2]

    transform = ResizeLongestSide(model.image_encoder.img_size)

    resized_image = transform.apply_image(image)
    input_size = resized_image.shape[:2]

    input_tensor = torch.as_tensor(resized_image, device=device)
    input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_tensor = model.preprocess(input_tensor)

    transformed_box = transform.apply_boxes(box[None, :], original_size)
    box_tensor = torch.as_tensor(transformed_box, dtype=torch.float, device=device)

    image_embedding = model.image_encoder(input_tensor)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    upscaled_logits = model.postprocess_masks(
        low_res_logits,
        input_size=input_size,
        original_size=original_size,
    )

    pred_prob = torch.sigmoid(upscaled_logits)
    pred_mask = pred_prob[0, 0].detach().cpu().numpy() > threshold

    return pred_mask


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM with oracle bounding boxes derived from ground-truth masks."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-padding", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
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

    model = load_medsam_model(
        checkpoint_path=Path(args.checkpoint),
        model_type=args.model_type,
        device=device,
    )

    pairs = find_pairs(dataset_path)

    all_metrics = []
    skipped_empty_masks = 0

    for image_path, mask_path in tqdm(pairs, desc="Evaluating MedSAM"):
        image = read_image_rgb(image_path)
        gt_mask = read_mask_binary(mask_path)

        box = mask_to_box(gt_mask, padding=args.box_padding)

        if box is None:
            skipped_empty_masks += 1
            continue

        pred_mask = predict_medsam_mask(
            model=model,
            image=image,
            box=box,
            device=device,
            threshold=args.threshold,
        )

        metrics = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)

        if args.save_predictions:
            out_path = pred_dir / image_path.name
            cv2.imwrite(str(out_path), pred_mask.astype(np.uint8) * 255)

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
    mean_metrics["threshold"] = args.threshold
    mean_metrics["prompt_type"] = "oracle_gt_bbox"
    mean_metrics["model_family"] = "MedSAM"

    output_path = output_dir / "test_metrics.yaml"

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mean_metrics, f, sort_keys=False)

    print("MedSAM evaluation complete")
    print(f"Saved metrics to: {output_path}")
    print(mean_metrics)


if __name__ == "__main__":
    main()