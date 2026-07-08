import argparse
from pathlib import Path

import torch
import yaml

from train_unet import UNetTrainer


def load_checkpoint(
    trainer: UNetTrainer,
    checkpoint_path: Path,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=trainer.device,
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

    trainer.model.load_state_dict(state_dict)

    print(
        f"Loaded pretrained checkpoint: "
        f"{checkpoint_path}"
    )


def override_learning_rate(
    trainer: UNetTrainer,
    learning_rate: float,
):
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = learning_rate

    print(
        f"Fine-tuning learning rate: "
        f"{learning_rate}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained U-Net on a target center."
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
        default=1e-5,
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

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    print("Target-center fine-tuning")
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    trainer = UNetTrainer(config)

    load_checkpoint(
        trainer=trainer,
        checkpoint_path=checkpoint_path,
    )

    override_learning_rate(
        trainer=trainer,
        learning_rate=args.learning_rate,
    )

    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()