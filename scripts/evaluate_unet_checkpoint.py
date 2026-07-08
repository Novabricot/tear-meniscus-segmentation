import argparse
import logging
from pathlib import Path

import torch
import yaml

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
        "Loaded checkpoint: %s",
        checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a U-Net checkpoint on the test split "
            "defined by a training config."
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
        "--output",
        required=True,
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

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

    trainer = UNetTrainer(
        config_path=str(config_path),
    )

    _, _, test_loader = trainer.build_dataloaders()

    model = trainer.build_model()

    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=trainer.device,
    )

    logger.info("Running checkpoint test evaluation")

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
        "miou=%.4f, "
        "loss=%.4f",
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["miou"],
        test_metrics["loss"],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
        output_path,
    )

    if trainer.writer is not None:
        trainer.writer.close()


if __name__ == "__main__":
    main()