import argparse
import random
import shutil
from pathlib import Path

import yaml


def find_image_mask_pairs(center_dir: Path):
    image_dir = center_dir / "Original"
    mask_dir = center_dir / "Label"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    pairs = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue

        mask_path = mask_dir / image_path.name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {center_dir}")

    return pairs


def split_dataset(
    pairs,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    rng = random.Random(seed)

    pairs = list(pairs)
    rng.shuffle(pairs)

    total = len(pairs)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_pairs = pairs[:train_size]

    val_pairs = pairs[
        train_size : train_size + val_size
    ]

    test_pairs = pairs[
        train_size + val_size :
    ]

    return train_pairs, val_pairs, test_pairs


def copy_pairs(
    pairs,
    output_root: Path,
    split: str,
    center_name: str,
):
    image_output = output_root / split / "images"
    mask_output = output_root / split / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        output_name = f"{center_name}__{image_path.name}"

        shutil.copy2(image_path, image_output / output_name)
        shutil.copy2(mask_path, mask_output / output_name)


def create_config(
    base_config: Path,
    output_config: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    seed: int,
    epochs: int,
):
    with base_config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["data"]["dataset_path"] = str(dataset_path)
    config["data"]["random_seed"] = seed

    config["training"]["epochs"] = epochs
    config["training"]["batch_size"] = 2
    config["training"]["gradient_accumulation_steps"] = 4

    config["logging"]["checkpoint_dir"] = str(checkpoint_dir)
    config["logging"]["log_dir"] = str(log_dir)

    output_config.parent.mkdir(parents=True, exist_ok=True)

    with output_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a target-center infrared split and configs "
            "for feature extraction transfer experiments."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/Open DataSet2",
    )

    parser.add_argument(
        "--target-center",
        required=True,
        choices=[
            "Infrared1",
            "Infrared2",
            "Infrared3",
        ],
    )

    parser.add_argument(
        "--output-root",
        default="data/feature_extraction",
    )

    parser.add_argument(
        "--config-root",
        default="configs/feature_extraction",
    )

    parser.add_argument(
        "--base-config",
        default="configs/within_modality_center/infrared_leave_Infrared3_out.yaml",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    raw_root = Path(args.input)
    target_name = args.target_center
    target_slug = target_name.lower()

    dataset_path = Path(args.output_root) / target_slug

    if dataset_path.exists() and args.force:
        shutil.rmtree(dataset_path)

    pairs = find_image_mask_pairs(raw_root / target_name)

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    copy_pairs(
        train_pairs,
        dataset_path,
        "train",
        target_name,
    )

    copy_pairs(
        val_pairs,
        dataset_path,
        "val",
        target_name,
    )

    copy_pairs(
        test_pairs,
        dataset_path,
        "test",
        target_name,
    )

    experiments = [
        "scratch",
        "encoder_only",
        "frozen_encoder",
        "full_finetuning",
    ]

    for experiment in experiments:
        run_name = f"{target_slug}_{experiment}"

        create_config(
            base_config=Path(args.base_config),
            output_config=Path(args.config_root) / f"{run_name}.yaml",
            dataset_path=dataset_path,
            checkpoint_dir=Path("models/feature_extraction") / run_name,
            log_dir=Path("results/feature_extraction") / run_name,
            seed=args.seed,
            epochs=args.epochs,
        )

    print(f"Created feature extraction split for {target_name}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")
    print(f"Dataset: {dataset_path}")
    print(f"Configs written to: {args.config_root}")
    print()
    print("Created configs:")
    for experiment in experiments:
        run_name = f"{target_slug}_{experiment}"
        print(f"- {Path(args.config_root) / f'{run_name}.yaml'}")


if __name__ == "__main__":
    main()