import argparse
import math
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


def copy_pairs(pairs, output_root: Path, split: str, center_name: str):
    image_output = output_root / split / "images"
    mask_output = output_root / split / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        output_name = f"{center_name}__{image_path.name}"

        shutil.copy2(image_path, image_output / output_name)
        shutil.copy2(mask_path, mask_output / output_name)


def copy_multi_center_pairs(pairs, output_root: Path, split: str):
    image_output = output_root / split / "images"
    mask_output = output_root / split / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for center_name, image_path, mask_path in pairs:
        output_name = f"{center_name}__{image_path.name}"

        shutil.copy2(image_path, image_output / output_name)
        shutil.copy2(mask_path, mask_output / output_name)


def split_train_val(pairs, val_ratio: float, seed: int):
    rng = random.Random(seed)

    pairs = list(pairs)
    rng.shuffle(pairs)

    val_size = max(1, int(len(pairs) * val_ratio))

    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    return train_pairs, val_pairs


def create_config(
    base_config: Path,
    output_config: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
):
    with base_config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["data"]["dataset_path"] = str(dataset_path)
    config["data"]["random_seed"] = seed

    config["training"]["epochs"] = epochs
    config["training"]["batch_size"] = batch_size
    config["training"]["gradient_accumulation_steps"] = (
        gradient_accumulation_steps
    )

    config["logging"]["checkpoint_dir"] = str(checkpoint_dir)
    config["logging"]["log_dir"] = str(log_dir)

    output_config.parent.mkdir(parents=True, exist_ok=True)

    with output_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create fixed-test target-center fine-tuning datasets."
    )

    parser.add_argument(
        "--input",
        default="data/raw/Open DataSet2",
    )

    parser.add_argument(
        "--output",
        default="data/fixed_target_finetuning",
    )

    parser.add_argument(
        "--config-output",
        default="configs/fixed_target_finetuning",
    )

    parser.add_argument(
        "--base-config",
        default="configs/unet_config.yaml",
    )

    parser.add_argument(
        "--models-root",
        default="models/fixed_target_finetuning",
    )

    parser.add_argument(
        "--results-root",
        default="results/fixed_target_finetuning",
    )

    parser.add_argument(
        "--source-centers",
        nargs="+",
        default=["Infrared2", "Infrared3"],
    )

    parser.add_argument(
        "--target-center",
        default="Infrared1",
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
    )

    parser.add_argument(
        "--target-test-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--source-val-ratio",
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
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    raw_root = Path(args.input)
    output_root = Path(args.output)
    config_root = Path(args.config_output)
    base_config = Path(args.base_config)
    models_root = Path(args.models_root)
    results_root = Path(args.results_root)

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)

    source_pairs = []

    for center_name in args.source_centers:
        center_pairs = find_image_mask_pairs(
            raw_root / center_name
        )

        source_pairs.extend(
            [
                (center_name, image_path, mask_path)
                for image_path, mask_path in center_pairs
            ]
        )

    target_pairs = find_image_mask_pairs(
        raw_root / args.target_center
    )

    rng = random.Random(args.seed)

    target_pairs = list(target_pairs)
    rng.shuffle(target_pairs)

    target_test_size = max(
        1,
        int(len(target_pairs) * args.target_test_ratio),
    )

    target_test_pairs = target_pairs[:target_test_size]
    target_adaptation_pool = target_pairs[target_test_size:]

    source_train_pairs, source_val_pairs = split_train_val(
        source_pairs,
        val_ratio=args.source_val_ratio,
        seed=args.seed,
    )

    source_dataset_root = output_root / "source"

    copy_multi_center_pairs(
        source_train_pairs,
        source_dataset_root,
        "train",
    )

    copy_multi_center_pairs(
        source_val_pairs,
        source_dataset_root,
        "val",
    )

    copy_pairs(
        target_test_pairs,
        source_dataset_root,
        "test",
        args.target_center,
    )

    create_config(
        base_config=base_config,
        output_config=config_root / "source.yaml",
        dataset_path=source_dataset_root,
        checkpoint_dir=models_root / "source",
        log_dir=results_root / "source",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print("\nCreated source dataset")
    print(f"Source train: {len(source_train_pairs)}")
    print(f"Source val:   {len(source_val_pairs)}")
    print(f"Target test:  {len(target_test_pairs)}")

    for fraction in args.fractions:
        percentage = int(round(fraction * 100))

        adaptation_size = max(
            1,
            math.ceil(
                len(target_adaptation_pool) * fraction
            ),
        )

        adaptation_pairs = target_adaptation_pool[
            :adaptation_size
        ]

        if len(adaptation_pairs) < 2:
            raise RuntimeError(
                f"Not enough target samples for {percentage}% fine-tuning."
            )

        adaptation_train_pairs, adaptation_val_pairs = (
            split_train_val(
                adaptation_pairs,
                val_ratio=0.20,
                seed=args.seed,
            )
        )

        dataset_name = f"target_{percentage}pct"
        dataset_root = output_root / dataset_name

        copy_pairs(
            adaptation_train_pairs,
            dataset_root,
            "train",
            args.target_center,
        )

        copy_pairs(
            adaptation_val_pairs,
            dataset_root,
            "val",
            args.target_center,
        )

        copy_pairs(
            target_test_pairs,
            dataset_root,
            "test",
            args.target_center,
        )

        create_config(
            base_config=base_config,
            output_config=config_root / f"{dataset_name}.yaml",
            dataset_path=dataset_root,
            checkpoint_dir=models_root / dataset_name,
            log_dir=results_root / dataset_name,
            seed=args.seed,
            epochs=20,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        print(f"\nCreated {dataset_name}")
        print(
            f"Adaptation train: "
            f"{len(adaptation_train_pairs)}"
        )
        print(
            f"Adaptation val:   "
            f"{len(adaptation_val_pairs)}"
        )
        print(
            f"Fixed target test: "
            f"{len(target_test_pairs)}"
        )

    print("\nFinished.")
    print(
        "All fine-tuning experiments use the same "
        "target-center test set."
    )


if __name__ == "__main__":
    main()