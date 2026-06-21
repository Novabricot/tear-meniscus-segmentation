import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.augmentation import (
    build_train_transforms,
    build_validation_transforms,
    progressive_intensity,
)
from scripts.utils.data_loader import TearMeniscusDataset
from scripts.utils.metrics import batch_metrics


logger = logging.getLogger(__name__)


def ensure_mask_channel(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    return masks.float()


class UNetTransferTrainer:
    def __init__(
        self,
        config_path: str,
        epochs: int = None,
        batch_size: int = None,
        init_checkpoint: str = None,
        output_name: str = "transfer",
    ):
        self.config = self.load_config(config_path)
        self.epochs = epochs if epochs is not None else self.config["training"]["epochs"]
        self.batch_size = batch_size if batch_size is not None else self.config["training"]["batch_size"]
        self.init_checkpoint = init_checkpoint
        self.output_name = output_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dir = Path("results") / output_name
        self.checkpoint_dir = Path("models") / output_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        logger.info(f"Using device: {self.device}")
        logger.info(f"Output name: {self.output_name}")

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_model(self):
        model_config = self.config["model"]

        model = smp.Unet(
            encoder_name=model_config.get("encoder_name", "resnet34"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            in_channels=model_config.get("in_channels", 3),
            classes=model_config.get("classes", 1),
            activation=None,
        )

        model = model.to(self.device)

        if self.init_checkpoint is not None:
            checkpoint = torch.load(self.init_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded initial checkpoint: {self.init_checkpoint}")

        return model

    def build_optimizer(self, model):
        optimizer_name = self.config["training"].get("optimizer", "AdamW")
        lr = float(self.config["training"].get("learning_rate", 5e-4))
        weight_decay = float(self.config["training"].get("weight_decay", 1e-5))

        if optimizer_name.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_scheduler(self, optimizer):
        scheduler_name = self.config["training"].get("scheduler", "CosineAnnealingLR")

        if scheduler_name == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.config["training"].get("T_max", self.epochs)),
                eta_min=float(self.config["training"].get("min_lr", 1e-6)),
            )

        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.config["training"].get("step_size", 20)),
            gamma=float(self.config["training"].get("gamma", 0.5)),
        )

    def build_dataloaders(self):
        data_config = self.config["data"]

        train_transform = build_train_transforms(
            self.config,
            data_config["input_height"],
            data_config["input_width"],
            intensity=1.0,
        )

        val_transform = build_validation_transforms(
            data_config["input_height"],
            data_config["input_width"],
        )

        train_dataset = TearMeniscusDataset(
            processed_root=data_config["dataset_path"],
            split="train",
            transform=train_transform,
        )

        val_dataset = TearMeniscusDataset(
            processed_root=data_config["dataset_path"],
            split="val",
            transform=val_transform,
        )

        test_dataset = TearMeniscusDataset(
            processed_root=data_config["dataset_path"],
            split="test",
            transform=val_transform,
        )

        common_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": int(data_config.get("num_workers", 4)),
            "pin_memory": self.device.type == "cuda",
            "drop_last": False,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

        return train_loader, val_loader, test_loader

    def compute_loss(self, logits, masks):
        masks = ensure_mask_channel(masks)

        loss_cfg = self.config["loss"]
        bce_weight = float(loss_cfg.get("bce_weight", 1.0))
        dice_weight = float(loss_cfg.get("dice_weight", 1.0))

        bce = nn.BCEWithLogitsLoss()(logits, masks)
        dice = smp.losses.DiceLoss(mode="binary", from_logits=True)(logits, masks)

        return bce_weight * bce + dice_weight * dice

    def apply_mixup(self, images, masks):
        masks = ensure_mask_channel(masks)

        alpha = float(self.config["loss"].get("mixup_alpha", 0.2))

        if alpha <= 0 or images.size(0) < 2:
            return images, masks

        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(images.size(0), device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_masks = lam * masks + (1.0 - lam) * masks[perm]

        return mixed_images, mixed_masks

    def train_epoch(self, model, loader, optimizer, scaler, epoch):
        model.train()

        total_loss = 0.0
        steps = 0
        mixup_enabled = self.config["loss"].get("mixup_enabled", True)

        for batch in tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            masks = ensure_mask_channel(masks)

            if mixup_enabled:
                images, masks = self.apply_mixup(images, masks)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config["training"].get("use_amp", False)):
                logits = model(images)
                loss = self.compute_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def validate(self, model, loader, name="Validate"):
        model.eval()

        total_loss = 0.0
        metrics_sum = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "miou": 0.0}
        steps = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=name, leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                masks = ensure_mask_channel(masks)

                logits = model(images)
                loss = self.compute_loss(logits, masks)

                total_loss += loss.item()

                values = batch_metrics(logits, masks)
                for key, value in values.items():
                    metrics_sum[key] += value

                steps += 1

        avg = {key: value / steps for key, value in metrics_sum.items()}
        avg["loss"] = total_loss / steps

        return avg

    def save_checkpoint(self, model, epoch, best=False):
        filename = "best_model.pth" if best else f"epoch_{epoch + 1}.pth"
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": self.config,
                "init_checkpoint": self.init_checkpoint,
            },
            path,
        )

        logger.info(f"Saved checkpoint: {path}")

    def train(self):
        train_loader, val_loader, test_loader = self.build_dataloaders()

        model = self.build_model()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        scaler = GradScaler(enabled=self.config["training"].get("use_amp", False))

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(self.epochs):
            intensity = progressive_intensity(self.config, epoch, self.epochs)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} | augmentation intensity={intensity:.3f}")

            train_loader.dataset.set_transform(
                build_train_transforms(
                    self.config,
                    self.config["data"]["input_height"],
                    self.config["data"]["input_width"],
                    intensity=intensity,
                )
            )

            train_loss = self.train_epoch(model, train_loader, optimizer, scaler, epoch)
            val_metrics = self.validate(model, val_loader)
            scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, "
                f"precision={val_metrics['precision']:.4f}, "
                f"recall={val_metrics['recall']:.4f}, "
                f"miou={val_metrics['miou']:.4f}"
            )

            self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch + 1)
            self.writer.add_scalar("Metrics/f1", val_metrics["f1"], epoch + 1)
            self.writer.add_scalar("Metrics/precision", val_metrics["precision"], epoch + 1)
            self.writer.add_scalar("Metrics/recall", val_metrics["recall"], epoch + 1)
            self.writer.add_scalar("Metrics/miou", val_metrics["miou"], epoch + 1)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch + 1
                self.save_checkpoint(model, epoch, best=True)

        logger.info(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
        self.save_checkpoint(model, self.epochs - 1, best=False)

        logger.info("Running final test evaluation")
        test_metrics = self.validate(model, test_loader, name="Test")

        metrics_path = self.log_dir / "test_metrics.yaml"

        with open(metrics_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(test_metrics, f)

        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Saved test metrics to {metrics_path}")

        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net with optional transfer learning")

    parser.add_argument("--config", type=str, default="configs/unet_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="transfer")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    trainer = UNetTransferTrainer(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        init_checkpoint=args.init_checkpoint,
        output_name=args.output_name,
    )

    trainer.train()


if __name__ == "__main__":
    main()