import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from train_unet import UNetTrainer, ensure_mask_channel


logger = logging.getLogger(__name__)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def load_encoder_only(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    encoder_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    if not encoder_state_dict:
        raise RuntimeError(
            "No encoder parameters were found in the checkpoint."
        )

    incompatible_keys = model.load_state_dict(
        encoder_state_dict,
        strict=False,
    )

    logger.info(
        "Loaded %d encoder tensors from checkpoint: %s",
        len(encoder_state_dict),
        checkpoint_path,
    )

    logger.info(
        "Missing keys after encoder-only loading: %d",
        len(incompatible_keys.missing_keys),
    )

    logger.info(
        "Unexpected keys after encoder-only loading: %d",
        len(incompatible_keys.unexpected_keys),
    )

    unexpected_encoder_keys = [
        key
        for key in incompatible_keys.unexpected_keys
        if key.startswith("encoder.")
    ]

    if unexpected_encoder_keys:
        raise RuntimeError(
            "Unexpected encoder keys while loading checkpoint: "
            f"{unexpected_encoder_keys}"
        )


def freeze_encoder(model: torch.nn.Module) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    model.encoder.eval()

    encoder_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info(
        "Encoder frozen | encoder parameters=%d | "
        "trainable decoder/head parameters=%d",
        encoder_parameters,
        trainable_parameters,
    )


def build_trainable_optimizer(
    trainer: UNetTrainer,
    model: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    weight_decay = float(
        trainer.config["training"].get(
            "weight_decay",
            1e-5,
        )
    )

    return torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_epoch_frozen_encoder(
    trainer: UNetTrainer,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
) -> float:
    model.train()

    # Keep the pretrained encoder completely frozen.
    # This also prevents BatchNorm running statistics
    # from being updated.
    model.encoder.eval()

    total_loss = 0.0
    step = 0

    mixup_enabled = bool(
        trainer.config["loss"].get(
            "mixup_enabled",
            True,
        )
    )

    use_amp = (
        bool(
            trainer.config["training"].get(
                "use_amp",
                False,
            )
        )
        and trainer.device.type == "cuda"
    )

    for batch in tqdm(
        loader,
        desc=f"Train Epoch {epoch + 1}",
        leave=False,
    ):
        images = batch["image"].to(
            trainer.device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            trainer.device,
            non_blocking=True,
        )

        masks = ensure_mask_channel(masks)

        if mixup_enabled:
            images, masks = trainer.apply_mixup(
                images,
                masks,
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=use_amp,
        ):
            logits = model(images)

            loss = trainer.compute_loss(
                logits,
                masks,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        step += 1

    return total_loss / max(step, 1)


def run_training(
    trainer: UNetTrainer,
    checkpoint_path: Path,
    learning_rate: float,
) -> None:
    train_loader, val_loader, test_loader = (
        trainer.build_dataloaders()
    )

    # build_model creates a new U-Net.
    # The decoder and segmentation head therefore start
    # from their normal fresh initialization.
    model = trainer.build_model()

    load_encoder_only(
        model=model,
        checkpoint_path=checkpoint_path,
        device=trainer.device,
    )

    freeze_encoder(model)

    optimizer = build_trainable_optimizer(
        trainer=trainer,
        model=model,
        learning_rate=learning_rate,
    )

    scheduler = trainer.build_scheduler(optimizer)

    use_amp = (
        bool(
            trainer.config["training"].get(
                "use_amp",
                False,
            )
        )
        and trainer.device.type == "cuda"
    )

    scaler = GradScaler(enabled=use_amp)

    best_f1 = 0.0
    best_epoch = 0

    logger.info(
        "Encoder-only transfer learning rate: %s",
        learning_rate,
    )

    for epoch in range(trainer.epochs):
        train_loss = train_epoch_frozen_encoder(
            trainer=trainer,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
        )

        val_metrics = trainer.validate(
            model=model,
            loader=val_loader,
            desc="Validate",
        )

        scheduler.step()

        logger.info(
            "Epoch %d/%d: "
            "train_loss=%.4f, "
            "val_loss=%.4f, "
            "f1=%.4f, "
            "precision=%.4f, "
            "recall=%.4f, "
            "miou=%.4f",
            epoch + 1,
            trainer.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["miou"],
        )

        if trainer.writer is not None:
            trainer.writer.add_scalar(
                "Loss/train",
                train_loss,
                epoch + 1,
            )

            trainer.writer.add_scalar(
                "Loss/val",
                val_metrics["loss"],
                epoch + 1,
            )

            trainer.writer.add_scalar(
                "Metrics/f1",
                val_metrics["f1"],
                epoch + 1,
            )

            trainer.writer.add_scalar(
                "Metrics/precision",
                val_metrics["precision"],
                epoch + 1,
            )

            trainer.writer.add_scalar(
                "Metrics/recall",
                val_metrics["recall"],
                epoch + 1,
            )

            trainer.writer.add_scalar(
                "Metrics/miou",
                val_metrics["miou"],
                epoch + 1,
            )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1

            trainer.save_checkpoint(
                model=model,
                epoch=epoch,
                best=True,
            )

    logger.info(
        "Best validation F1: %.4f at epoch %d",
        best_f1,
        best_epoch,
    )

    best_checkpoint_path = (
        trainer.checkpoint_dir / "best_model.pth"
    )

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            "Best checkpoint was not created: "
            f"{best_checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=trainer.device,
    )

    best_state_dict = extract_state_dict(
        best_checkpoint
    )

    model.load_state_dict(best_state_dict)

    logger.info(
        "Reloaded best encoder-only transfer checkpoint: %s",
        best_checkpoint_path,
    )

    logger.info("Running final test evaluation")

    test_metrics = trainer.validate(
        model=model,
        loader=test_loader,
        desc="Test",
    )

    logger.info(
        "Test metrics: "
        "f1=%.4f, "
        "precision=%.4f, "
        "recall=%.4f, "
        "miou=%.4f",
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["miou"],
    )

    metrics_path = (
        trainer.log_dir / "test_metrics.yaml"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            test_metrics,
            file,
            sort_keys=False,
        )

    logger.info(
        "Saved test metrics to %s",
        metrics_path,
    )

    if trainer.writer is not None:
        trainer.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load only a pretrained U-Net encoder, "
            "freeze it, and train a fresh decoder."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    logger.info(
        "Starting encoder-only feature extraction experiment"
    )

    logger.info(
        "Config: %s",
        config_path,
    )

    logger.info(
        "Source checkpoint: %s",
        checkpoint_path,
    )

    trainer = UNetTrainer(
        config_path=str(config_path),
    )

    run_training(
        trainer=trainer,
        checkpoint_path=checkpoint_path,
        learning_rate=args.learning_rate,
    )

    logger.info(
        "Encoder-only feature extraction experiment complete"
    )


if __name__ == "__main__":
    main()