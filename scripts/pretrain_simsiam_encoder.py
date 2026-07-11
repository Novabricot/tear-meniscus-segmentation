import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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


class TwoViewTransform:
    """Create two stochastic augmented views of the same image."""

    def __init__(self, image_size: int = 224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.5, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomAutocontrast(p=0.2),
                transforms.GaussianBlur(
                    kernel_size=23,
                    sigma=(0.1, 2.0),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.transform(image),
            self.transform(image),
        )


class RepresentationDataset(Dataset):
    """Image-only dataset for self-supervised representation learning."""

    def __init__(
        self,
        image_paths: List[Path],
        transform: TwoViewTransform,
    ) -> None:
        if not image_paths:
            raise ValueError("The image path list is empty.")

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        view1, view2 = self.transform(image)

        return view1, view2


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                output_dim,
                bias=False,
            ),
            nn.BatchNorm1d(
                output_dim,
                affine=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class PredictionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 2048,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class SimSiamModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        feature_dim = self._infer_feature_dim(
            image_size=image_size,
        )

        logger.info(
            "Encoder output feature dimension: %d",
            feature_dim,
        )

        self.projector = ProjectionMLP(
            input_dim=feature_dim,
        )

        self.predictor = PredictionMLP()

    def _infer_feature_dim(
        self,
        image_size: int,
    ) -> int:
        encoder_was_training = self.encoder.training

        self.encoder.eval()

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                image_size,
                image_size,
            )

            features = self.encoder(dummy)
            final_feature = features[-1]

            pooled = F.adaptive_avg_pool2d(
                final_feature,
                output_size=1,
            )

            feature_dim = pooled.flatten(1).shape[1]

        if encoder_was_training:
            self.encoder.train()

        return feature_dim

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(x)
        final_feature = features[-1]

        pooled = F.adaptive_avg_pool2d(
            final_feature,
            output_size=1,
        )

        return pooled.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(x)

        projection = self.projector(
            representation
        )

        prediction = self.predictor(
            projection
        )

        return prediction, projection


def negative_cosine_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target = target.detach()

    return -F.cosine_similarity(
        prediction,
        target,
        dim=1,
    ).mean()


def simsiam_loss(
    prediction1: torch.Tensor,
    projection1: torch.Tensor,
    prediction2: torch.Tensor,
    projection2: torch.Tensor,
) -> torch.Tensor:
    loss1 = negative_cosine_similarity(
        prediction1,
        projection2,
    )

    loss2 = negative_cosine_similarity(
        prediction2,
        projection1,
    )

    return 0.5 * (loss1 + loss2)


def find_source_images(
    raw_root: Path,
    source_centers: List[str],
) -> List[Path]:
    image_paths = []

    for center_name in source_centers:
        image_dir = (
            raw_root
            / center_name
            / "Original"
        )

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        center_images = [
            path
            for path in sorted(image_dir.iterdir())
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
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
    model: SimSiamModel,
    output_path: Path,
    epoch: int,
    loss: float,
) -> None:
    encoder_state_dict = {
        f"encoder.{key}": value.cpu()
        for key, value
        in model.encoder.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": encoder_state_dict,
        "epoch": epoch,
        "loss": loss,
        "method": "simsiam",
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        output_path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised SimSiam pretraining of a "
            "U-Net-compatible encoder."
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
        default=224,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
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
            "simsiam_ir1_ir2"
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

    transform = TwoViewTransform(
        image_size=args.image_size,
    )

    dataset = RepresentationDataset(
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
            "No complete training batch was created. "
            "Reduce the batch size."
        )

    model = SimSiamModel(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        image_size=args.image_size,
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
        "Starting SimSiam pretraining for %d epochs",
        args.epochs,
    )

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(
            data_loader,
            desc=(
                f"SimSiam Epoch "
                f"{epoch + 1}/{args.epochs}"
            ),
        )

        for view1, view2 in progress_bar:
            view1 = view1.to(
                device,
                non_blocking=True,
            )

            view2 = view2.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with autocast(enabled=use_amp):
                prediction1, projection1 = model(
                    view1
                )

                prediction2, projection2 = model(
                    view2
                )

                loss = simsiam_loss(
                    prediction1=prediction1,
                    projection1=projection1,
                    prediction2=prediction2,
                    projection2=projection2,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        epoch_loss = (
            running_loss
            / max(batch_count, 1)
        )

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
        "SimSiam pretraining complete"
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