import argparse
import logging
import random
from pathlib import Path
from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class DenoisingTransform:
    """
    Creates a clean image tensor and a corrupted version of it.

    The model receives the corrupted image and tries to reconstruct
    the clean image.
    """

    def __init__(
        self,
        image_size: int = 256,
        noise_std: float = 0.08,
        erase_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.noise_std = noise_std
        self.erase_prob = erase_prob

        self.clean_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def _random_blur(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if random.random() < 0.35:
            radius = random.uniform(0.3, 1.2)
            return image.filter(
                ImageFilter.GaussianBlur(radius=radius)
            )

        return image

    def _random_brightness_contrast(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() < 0.5:
            brightness_factor = random.uniform(
                0.8,
                1.2,
            )

            tensor = tensor * brightness_factor

        if random.random() < 0.5:
            mean = tensor.mean(
                dim=(1, 2),
                keepdim=True,
            )

            contrast_factor = random.uniform(
                0.8,
                1.2,
            )

            tensor = (
                tensor - mean
            ) * contrast_factor + mean

        return tensor.clamp(0.0, 1.0)

    def _random_erasing(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if random.random() > self.erase_prob:
            return tensor

        _, height, width = tensor.shape

        erase_height = random.randint(
            max(4, height // 32),
            max(8, height // 8),
        )

        erase_width = random.randint(
            max(4, width // 32),
            max(8, width // 8),
        )

        top = random.randint(
            0,
            max(0, height - erase_height),
        )

        left = random.randint(
            0,
            max(0, width - erase_width),
        )

        fill_value = tensor.mean()

        tensor[
            :,
            top : top + erase_height,
            left : left + erase_width,
        ] = fill_value

        return tensor

    def _corrupt(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        image = self._random_blur(image)

        tensor = self.clean_transform(image)

        if random.random() < 0.8:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise

        tensor = self._random_brightness_contrast(tensor)

        tensor = self._random_erasing(tensor)

        return tensor.clamp(0.0, 1.0)

    def __call__(
        self,
        image: Image.Image,
    ):
        image = image.convert("RGB")

        clean = self.clean_transform(image)
        corrupted = self._corrupt(image)

        return corrupted, clean


class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        transform: DenoisingTransform,
    ) -> None:
        if not image_paths:
            raise ValueError("No image paths were provided.")

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            corrupted, clean = self.transform(image)

        return {
            "corrupted": corrupted,
            "clean": clean,
        }


def find_source_images(
    raw_root: Path,
    source_centers: List[str],
) -> List[Path]:
    image_paths = []

    for center_name in source_centers:
        image_dir = raw_root / center_name / "Original"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        center_images = [
            path
            for path in sorted(image_dir.iterdir())
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]

        if not center_images:
            raise RuntimeError(
                f"No images found in {image_dir}"
            )

        logger.info(
            "%s: %d images",
            center_name,
            len(center_images),
        )

        image_paths.extend(center_images)

    return image_paths


def save_encoder_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    epoch: int,
    loss: float,
) -> None:
    encoder_state_dict = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if key.startswith("encoder.")
    }

    checkpoint = {
        "model_state_dict": encoder_state_dict,
        "epoch": epoch,
        "loss": loss,
        "method": "denoising_autoencoder",
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        output_path,
    )


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    l1 = F.l1_loss(
        prediction,
        target,
    )

    mse = F.mse_loss(
        prediction,
        target,
    )

    return l1 + 0.5 * mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised denoising autoencoder "
            "pretraining for a U-Net encoder."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/Open DataSet2",
    )

    parser.add_argument(
        "--source-centers",
        nargs="+",
        default=[
            "Infrared1",
            "Infrared2",
        ],
    )

    parser.add_argument(
        "--encoder-name",
        default="resnet34",
    )

    parser.add_argument(
        "--encoder-weights",
        default="imagenet",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--erase-prob",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "models/representation_learning/"
            "dae_ir1_ir2"
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

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "Using device: %s",
        device,
    )

    raw_root = Path(args.input)
    output_dir = Path(args.output_dir)

    image_paths = find_source_images(
        raw_root=raw_root,
        source_centers=args.source_centers,
    )

    logger.info(
        "Total self-supervised images: %d",
        len(image_paths),
    )

    transform = DenoisingTransform(
        image_size=args.image_size,
        noise_std=args.noise_std,
        erase_prob=args.erase_prob,
    )

    dataset = ImageOnlyDataset(
        image_paths=image_paths,
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if len(data_loader) == 0:
        raise RuntimeError(
            "No complete batch was created. "
            "Reduce the batch size."
        )

    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=3,
        activation=None,
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")

    best_checkpoint_path = (
        output_dir / "best_encoder.pth"
    )

    logger.info(
        "Starting denoising autoencoder pretraining "
        "for %d epochs",
        args.epochs,
    )

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(
            data_loader,
            desc=(
                f"DAE Epoch "
                f"{epoch + 1}/{args.epochs}"
            ),
        )

        for batch in progress_bar:
            corrupted = batch["corrupted"].to(
                device,
                non_blocking=True,
            )

            clean = batch["clean"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with autocast(enabled=use_amp):
                prediction = model(corrupted)
                prediction = torch.sigmoid(prediction)

                loss = reconstruction_loss(
                    prediction,
                    clean,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        epoch_loss = running_loss / max(batch_count, 1)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d | loss=%.6f | lr=%.8f",
            epoch + 1,
            args.epochs,
            epoch_loss,
            current_lr,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            save_encoder_checkpoint(
                model=model,
                output_path=best_checkpoint_path,
                epoch=epoch + 1,
                loss=epoch_loss,
            )

            logger.info(
                "Saved best encoder: %s",
                best_checkpoint_path,
            )

        if (epoch + 1) % 10 == 0:
            periodic_checkpoint_path = (
                output_dir
                / f"encoder_epoch_{epoch + 1}.pth"
            )

            save_encoder_checkpoint(
                model=model,
                output_path=periodic_checkpoint_path,
                epoch=epoch + 1,
                loss=epoch_loss,
            )

    logger.info(
        "Denoising autoencoder pretraining complete"
    )

    logger.info(
        "Best training loss: %.6f",
        best_loss,
    )

    logger.info(
        "Best encoder checkpoint: %s",
        best_checkpoint_path,
    )


if __name__ == "__main__":
    main()