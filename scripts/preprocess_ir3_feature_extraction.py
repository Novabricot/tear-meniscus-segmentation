import argparse
import random
import shutil
from pathlib import Path

import yaml


def find_image_mask_pairs(center_dir: Path):
    image_dir = center_dir / "Original"
    mask_dir = center_dir / "Label"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory: {image_dir}"
        )

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Missing mask directory: {mask_dir}"
        )

    pairs = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue

        mask_path = mask_dir / image_path.name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found in {center_dir}"
        )

    return pairs


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

        shutil.copy2(
            image_path,
            image_output / output_name,
        )

        shutil.copy2(
            mask_path,
            mask_output / output_name,
        )


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


def create_config(
    base_config: Path,
    output_config: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    seed: int,
):
    with base_config.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    config["data"]["dataset_path"] = str(dataset_path)
    config["data"]["random_seed"] = seed

    config["training"]["epochs"] = 30
    config["training"]["batch_size"] = 2
    config["training"]["gradient_accumulation_steps"] = 4

    config["logging"]["checkpoint_dir"] = str(
        checkpoint_dir
    )

    config["logging"]["log_dir"] = str(log_dir)

    output_config.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_config.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create an Infrared3 train/val/test split "
            "for feature extraction experiments."
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/Open DataSet2",
    )

    parser.add_argument(
        "--output",
        default="data/feature_extraction/infrared3",
    )

    parser.add_argument(
        "--config-output",
        default=(
            "configs/feature_extraction/"
            "infrared3_frozen_encoder.yaml"
        ),
    )

    parser.add_argument(
        "--base-config",
        default="configs/unet_config.yaml",
    )

    parser.add_argument(
        "--target-center",
        default="Infrared3",
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
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    raw_root = Path(args.input)
    output_root = Path(args.output)

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)

    pairs = find_image_mask_pairs(
        raw_root / args.target_center
    )

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    copy_pairs(
        train_pairs,
        output_root,
        "train",
        args.target_center,
    )

    copy_pairs(
        val_pairs,
        output_root,
        "val",
        args.target_center,
    )

    copy_pairs(
        test_pairs,
        output_root,
        "test",
        args.target_center,
    )

    create_config(
        base_config=Path(args.base_config),
        output_config=Path(args.config_output),
        dataset_path=output_root,
        checkpoint_dir=Path(
            "models/feature_extraction/"
            "infrared3_frozen_encoder"
        ),
        log_dir=Path(
            "results/feature_extraction/"
            "infrared3_frozen_encoder"
        ),
        seed=args.seed,
    )

    print("Infrared3 feature extraction split created")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")
    print(f"Dataset: {output_root}")
    print(f"Config:  {args.config_output}")


if __name__ == "__main__":
    main()