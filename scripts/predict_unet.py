import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import yaml
import segmentation_models_pytorch as smp

from scripts.utils.data_loader import TearMeniscusDataset
from scripts.utils.augmentation import build_validation_transforms


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config, device):
    model_cfg = config["model"]

    model = smp.Unet(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
        activation=None,
    )

    return model.to(device)


def denormalize_image(img_tensor):
    img = img_tensor.detach().cpu()

    if img.ndim == 3:
        img = img.permute(1, 2, 0)

    img = img.numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_config.yaml")
    parser.add_argument("--checkpoint", default="models/unet/best_model.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="results/unet/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = build_model(config, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(dataset))

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]

            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"]

            if mask.ndim == 3:
                mask = mask.squeeze(0)

            logits = model(image)
            prob = torch.sigmoid(logits)[0, 0].cpu()
            pred = (prob > args.threshold).float()

            img_vis = denormalize_image(sample["image"])
            mask_vis = mask.cpu().numpy()
            pred_vis = pred.numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_vis)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(mask_vis, cmap="gray")
            axes[1].set_title("Ground truth")
            axes[1].axis("off")

            axes[2].imshow(pred_vis, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = output_dir / f"prediction_{i:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            print(f"Saved {save_path}")


if __name__ == "__main__":
    main()