import argparse
import logging
import sys
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.augmentation import (
    build_train_transforms,
    build_validation_transforms,
    progressive_intensity,
)
from scripts.utils.data_loader import TearMeniscusDataset
from scripts.utils.metrics import batch_metrics


logger = logging.getLogger(__name__)


class UNetTrainer:
    def __init__(self, config_path: str, epochs: int = None, batch_size: int = None):
        self.config = self.load_config(config_path)
        self.epochs = epochs if epochs is not None else self.config["training"]["epochs"]
        self.batch_size = batch_size if batch_size is not None else self.config["training"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_dirs()
        self.writer = self.build_summary_writer()
        logger.info(f"Using device: {self.device}")

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup_dirs(self):
        self.log_dir = Path(self.config["logging"]["log_dir"])
        self.checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_summary_writer(self):
        if self.config["logging"].get("tensorboard_enabled", True):
            return SummaryWriter(log_dir=str(self.log_dir))
        return None

    def build_model(self):
        model_config = self.config["model"]
        model = smp.Unet(
            encoder_name=model_config.get("encoder_name", "resnet34"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            in_channels=model_config.get("in_channels", 3),
            classes=model_config.get("classes", 1),
            activation=None,
        )
        return model.to(self.device)

    def build_optimizer(self, model):
        optimizer_name = self.config["training"].get("optimizer", "AdamW")
        lr = self.config["training"].get("learning_rate", 5e-4)
        weight_decay = self.config["training"].get("weight_decay", 1e-5)

        if optimizer_name.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_scheduler(self, optimizer):
        scheduler_name = self.config["training"].get("scheduler", "CosineAnnealingLR")
        if scheduler_name == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config["training"].get("T_max", self.epochs),
                eta_min=self.config["training"].get("min_lr", 1e-6),
            )
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["training"].get("step_size", 20),
            gamma=self.config["training"].get("gamma", 0.5),
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
            data_config["input_height"], data_config["input_width"]
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=False,
        )

        return train_loader, val_loader, test_loader

    def compute_loss(self, logits, masks):
        loss_weights = self.config["loss"]
        bce_weight = loss_weights.get("bce_weight", 1.0)
        dice_weight = loss_weights.get("dice_weight", 1.0)

        bce = nn.BCEWithLogitsLoss()(logits, masks)
        dice = smp.losses.DiceLoss(mode="binary", from_logits=True)(logits, masks)
        return bce_weight * bce + dice_weight * dice

    def apply_mixup(self, images, masks):
        mixup_cfg = self.config["loss"]
        alpha = mixup_cfg.get("mixup_alpha", 0.2)
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
        step = 0
        mixup_enabled = self.config["loss"].get("mixup_enabled", True)

        for batch in tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            if mixup_enabled:
                images, masks = self.apply_mixup(images, masks)

            optimizer.zero_grad()
            with autocast(enabled=self.config["training"].get("use_amp", False)):
                logits = model(images)
                loss = self.compute_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            step += 1

        return total_loss / max(step, 1)

    def validate(self, model, loader):
        model.eval()
        total_loss = 0.0
        metrics_sum = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "miou": 0.0}
        step = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validate", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                logits = model(images)
                loss = self.compute_loss(logits, masks)
                total_loss += loss.item()

                batch_metrics_values = batch_metrics(logits, masks)
                for key, value in batch_metrics_values.items():
                    metrics_sum[key] += value
                step += 1

        if step == 0:
            raise RuntimeError("Validation loader is empty")

        avg_metrics = {key: value / step for key, value in metrics_sum.items()}
        avg_metrics["loss"] = total_loss / step
        return avg_metrics

    def save_checkpoint(self, model, epoch, best: bool = False):
        filename = f"epoch_{epoch + 1}.pth"
        if best and self.config["logging"].get("save_best_only", True):
            filename = "best_model.pth"
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "config": self.config,
        }, path)
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
            logger.info(f"Epoch {epoch+1}/{self.epochs} | augmentation intensity={intensity:.3f}")

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
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, precision={val_metrics['precision']:.4f}, "
                f"recall={val_metrics['recall']:.4f}, miou={val_metrics['miou']:.4f}"
            )

            if self.writer is not None:
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

            if (epoch + 1) % self.config["logging"].get("checkpoint_interval", 5) == 0:
                self.save_checkpoint(model, epoch, best=False)

        logger.info(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
        self.save_checkpoint(model, self.epochs - 1, best=False)

        logger.info("Running final test evaluation")
        test_metrics = self.validate(model, test_loader)
        logger.info(
            f"Test metrics: f1={test_metrics['f1']:.4f}, precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, miou={test_metrics['miou']:.4f}"
        )
        metrics_path = self.log_dir / "test_metrics.yaml"
        with open(metrics_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(test_metrics, f)
        logger.info(f"Saved test metrics to {metrics_path}")

        if self.writer is not None:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_config.yaml",
        help="Path to the training config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Starting U-Net training pipeline")
    trainer = UNetTrainer(config_path=args.config, epochs=args.epochs, batch_size=args.batch_size)
    trainer.train()
    logger.info("U-Net training pipeline complete")


if __name__ == "__main__":
    main()
