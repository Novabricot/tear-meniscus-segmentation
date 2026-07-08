import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler

from train_unet import UNetTrainer


logger = logging.getLogger(__name__)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    logger.info(
        "Loaded pretrained checkpoint: %s",
        checkpoint_path,
    )


def freeze_encoder(
    model: torch.nn.Module,
) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    logger.info(
        "Encoder frozen | trainable parameters=%d | "
        "frozen parameters=%d",
        trainable_parameters,
        frozen_parameters,
    )


def build_decoder_optimizer(
    trainer: UNetTrainer,
    model: torch.nn.Module,
    learning_rate: float,
):
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


def run_training(
    trainer: UNetTrainer,
    checkpoint_path: Path,
    learning_rate: float,
) -> None:
    train_loader, val_loader, test_loader = (
        trainer.build_dataloaders()
    )

    model = trainer.build_model()

    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=trainer.device,
    )

    freeze_encoder(model)

    optimizer = build_decoder_optimizer(
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
        "Feature extraction learning rate: %s",
        learning_rate,
    )

    for epoch in range(trainer.epochs):
        train_loss = trainer.train_epoch(
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

    load_checkpoint(
        model=model,
        checkpoint_path=best_checkpoint_path,
        device=trainer.device,
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a U-Net decoder using a frozen "
            "pretrained encoder."
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
        "Starting frozen-encoder feature extraction experiment"
    )

    logger.info("Config: %s", config_path)
    logger.info("Checkpoint: %s", checkpoint_path)

    trainer = UNetTrainer(
        config_path=str(config_path),
    )

    run_training(
        trainer=trainer,
        checkpoint_path=checkpoint_path,
        learning_rate=args.learning_rate,
    )

    logger.info(
        "Frozen-encoder experiment complete"
    )


if __name__ == "__main__":
    main()