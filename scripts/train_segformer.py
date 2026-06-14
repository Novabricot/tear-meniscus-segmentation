import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.augmentation import build_train_transforms, build_validation_transforms
from scripts.utils.data_loader import TearMeniscusDataset
from scripts.utils.metrics import batch_metrics


logger = logging.getLogger(__name__)


def ensure_mask_channel(mask):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


class SegFormerTrainer:
    def __init__(self, config_path, epochs=None, batch_size=None, model_name=None):
        self.config = self.load_config(config_path)
        self.epochs = epochs or self.config["training"].get("epochs", 60)
        self.batch_size = batch_size or self.config["training"].get("batch_size", 2)
        self.model_name = model_name or "nvidia/segformer-b0-finetuned-ade-512-512"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dir = Path("results/segformer")
        self.checkpoint_dir = Path("models/segformer")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using model: {self.model_name}")

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_model(self):
        model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        return model.to(self.device)

    def build_dataloaders(self):
        data_cfg = self.config["data"]

        train_transform = build_train_transforms(
            self.config,
            data_cfg["input_height"],
            data_cfg["input_width"],
            intensity=1.0,
        )

        val_transform = build_validation_transforms(
            data_cfg["input_height"],
            data_cfg["input_width"],
        )

        train_dataset = TearMeniscusDataset(
            processed_root=data_cfg["dataset_path"],
            split="train",
            transform=train_transform,
        )

        val_dataset = TearMeniscusDataset(
            processed_root=data_cfg["dataset_path"],
            split="val",
            transform=val_transform,
        )

        test_dataset = TearMeniscusDataset(
            processed_root=data_cfg["dataset_path"],
            split="test",
            transform=val_transform,
        )

        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": int(data_cfg.get("num_workers", 4)),
            "pin_memory": self.device.type == "cuda",
            "drop_last": False,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

    def compute_loss(self, logits, masks):
        masks = ensure_mask_channel(masks)

        logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        bce = nn.BCEWithLogitsLoss()(logits, masks)
        return bce, logits

    def train_epoch(self, model, loader, optimizer, scaler, epoch):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            masks = ensure_mask_channel(masks)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config["training"].get("use_amp", False)):
                outputs = model(pixel_values=images)
                loss, _ = self.compute_loss(outputs.logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def validate(self, model, loader):
        model.eval()
        total_loss = 0.0
        metrics_sum = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "miou": 0.0}
        steps = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validate", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                masks = ensure_mask_channel(masks)

                outputs = model(pixel_values=images)
                loss, logits = self.compute_loss(outputs.logits, masks)

                total_loss += loss.item()

                values = batch_metrics(logits, masks)
                for k, v in values.items():
                    metrics_sum[k] += v

                steps += 1

        avg = {k: v / steps for k, v in metrics_sum.items()}
        avg["loss"] = total_loss / steps
        return avg

    def save_checkpoint(self, model, epoch, best=False):
        filename = "best_model.pth" if best else f"epoch_{epoch + 1}.pth"
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_name": self.model_name,
                "config": self.config,
            },
            path,
        )

        logger.info(f"Saved checkpoint: {path}")

    def train(self):
        train_loader, val_loader, test_loader = self.build_dataloaders()
        model = self.build_model()

        lr = float(self.config["training"].get("learning_rate", 5e-5))
        weight_decay = float(self.config["training"].get("weight_decay", 1e-5))

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = GradScaler(enabled=self.config["training"].get("use_amp", False))

        best_f1 = 0.0

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

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
            self.writer.add_scalar("Metrics/miou", val_metrics["miou"], epoch + 1)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(model, epoch, best=True)

        self.save_checkpoint(model, self.epochs - 1, best=False)

        logger.info("Running final test evaluation")
        test_metrics = self.validate(model, test_loader)

        with open(self.log_dir / "test_metrics.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(test_metrics, f)

        logger.info(f"Test metrics: {test_metrics}")
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", default="nvidia/segformer-b0-finetuned-ade-512-512")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    trainer = SegFormerTrainer(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )

    trainer.train()


if __name__ == "__main__":
    main()