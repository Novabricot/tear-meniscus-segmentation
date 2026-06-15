import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import yaml
from transformers import SegformerForSemanticSegmentation

from scripts.utils.augmentation import build_validation_transforms
from scripts.utils.data_loader import TearMeniscusDataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_mask(mask):
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.cpu().numpy()


def denormalize_image(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_config.yaml")
    parser.add_argument("--checkpoint", default="models/segformer/best_model.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="results/segformer/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_name" in checkpoint:
        model_name = checkpoint["model_name"]

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_cfg = config["data"]
    transform = build_validation_transforms(
        data_cfg["input_height"],
        data_cfg["input_width"],
    )

    dataset = TearMeniscusDataset(
        processed_root=data_cfg["dataset_path"],
        split=args.split,
        transform=transform,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(dataset))

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]

            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"]

            outputs = model(pixel_values=image)
            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            prob = torch.sigmoid(logits)[0, 0].cpu()
            pred = (prob > args.threshold).float().numpy()

            img_vis = denormalize_image(sample["image"])
            mask_vis = ensure_mask(mask)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_vis)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(mask_vis, cmap="gray")
            axes[1].set_title("Ground truth")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("SegFormer prediction")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = output_dir / f"prediction_{i:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            print(f"Saved {save_path}")


if __name__ == "__main__":
    main()