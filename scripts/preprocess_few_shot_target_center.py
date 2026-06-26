import argparse
import math
import random
import shutil
from pathlib import Path

import yaml


def find_image_mask_pairs(center_dir: Path):
    image_dir = center_dir / "Original"
    mask_dir = center_dir / "Label"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing Original/Label in {center_dir}")

    pairs = []

    for img_path in sorted(image_dir.iterdir()):
        if not img_path.is_file():
            continue

        mask_path = mask_dir / img_path.name

        if mask_path.exists():
            pairs.append((img_path, mask_path))

    return pairs


def copy_pairs(pairs, output_root: Path, split: str):
    img_out = output_root / split / "images"
    mask_out = output_root / split / "masks"

    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for center_name, img_path, mask_path in pairs:
        out_name = f"{center_name}__{img_path.name}"

        shutil.copy2(img_path, img_out / out_name)
        shutil.copy2(mask_path, mask_out / out_name)


def split_source_train_val(source_pairs, val_ratio: float, seed: int):
    rng = random.Random(seed)
    source_pairs = list(source_pairs)
    rng.shuffle(source_pairs)

    val_size = int(len(source_pairs) * val_ratio)

    val_pairs = source_pairs[:val_size]
    train_pairs = source_pairs[val_size:]

    return train_pairs, val_pairs


def split_target_few_shot(target_pairs, fraction: float, seed: int):
    rng = random.Random(seed)
    target_pairs = list(target_pairs)
    rng.shuffle(target_pairs)

    n_target_train = max(1, math.ceil(len(target_pairs) * fraction))

    target_train_pairs = target_pairs[:n_target_train]
    target_test_pairs = target_pairs[n_target_train:]

    return target_train_pairs, target_test_pairs


def create_config(
    base_config: Path,
    output_config: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    epochs: int,
    seed: int,
    batch_size: int,
    gradient_accumulation_steps: int,
):
    with base_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["dataset_path"] = str(dataset_path)
    cfg["data"]["random_seed"] = seed

    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps

    cfg["logging"]["checkpoint_dir"] = str(checkpoint_dir)
    cfg["logging"]["log_dir"] = str(log_dir)

    output_config.parent.mkdir(parents=True, exist_ok=True)

    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_few_shot_fold(
    fraction: float,
    raw_root: Path,
    source_centers: list[str],
    target_center: str,
    output_root: Path,
    config_output: Path,
    base_config: Path,
    models_root: Path,
    results_root: Path,
    val_ratio: float,
    epochs: int,
    seed: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    force: bool,
):
    percent = int(round(fraction * 100))
    fold_name = f"{target_center}_target_{percent}pct"
    fold_root = output_root / fold_name

    if fold_root.exists():
        if force:
            shutil.rmtree(fold_root)
        else:
            print(f"Skipping existing fold: {fold_root}")
            return

    source_pairs = []

    for center in source_centers:
        raw_pairs = find_image_mask_pairs(raw_root / center)
        source_pairs.extend(
            [(center, img_path, mask_path) for img_path, mask_path in raw_pairs]
        )

    raw_target_pairs = find_image_mask_pairs(raw_root / target_center)
    target_pairs = [
        (target_center, img_path, mask_path)
        for img_path, mask_path in raw_target_pairs
    ]

    source_train_pairs, source_val_pairs = split_source_train_val(
        source_pairs=source_pairs,
        val_ratio=val_ratio,
        seed=seed,
    )

    target_train_pairs, target_test_pairs = split_target_few_shot(
        target_pairs=target_pairs,
        fraction=fraction,
        seed=seed,
    )

    train_pairs = source_train_pairs + target_train_pairs
    val_pairs = source_val_pairs
    test_pairs = target_test_pairs

    copy_pairs(train_pairs, fold_root, "train")
    copy_pairs(val_pairs, fold_root, "val")
    copy_pairs(test_pairs, fold_root, "test")

    config_path = config_output / f"{fold_name}.yaml"

    create_config(
        base_config=base_config,
        output_config=config_path,
        dataset_path=fold_root,
        checkpoint_dir=models_root / fold_name,
        log_dir=results_root / fold_name,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print(f"\nCreated few-shot fold: {fold_name}")
    print(f"Source centers: {source_centers}")
    print(f"Target center:  {target_center}")
    print(f"Target fraction used for training: {fraction:.2%}")
    print(f"Source train: {len(source_train_pairs)}")
    print(f"Source val:   {len(source_val_pairs)}")
    print(f"Target train: {len(target_train_pairs)}")
    print(f"Target test:  {len(target_test_pairs)}")
    print(f"Final train:  {len(train_pairs)}")
    print(f"Final val:    {len(val_pairs)}")
    print(f"Final test:   {len(test_pairs)}")
    print(f"Config:       {config_path}")
    print(f"Results:      {results_root / fold_name}")
    print(f"Models:       {models_root / fold_name}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create few-shot target-center adaptation splits. "
            "Example: train on IR2+IR3 plus 5/10/20% of IR1, test on remaining IR1."
        )
    )

    parser.add_argument("--input", default="data/raw/Open DataSet2")
    parser.add_argument("--output", default="data/few_shot_center")
    parser.add_argument("--config-output", default="configs/few_shot_center")
    parser.add_argument("--models-root", default="models/few_shot_center")
    parser.add_argument("--results-root", default="results/few_shot_center")
    parser.add_argument("--base-config", default="configs/unet_config.yaml")

    parser.add_argument("--target-center", default="Infrared1")
    parser.add_argument(
        "--source-centers",
        nargs="+",
        default=["Infrared2", "Infrared3"],
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
        help="Target-center fractions added to training.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    raw_root = Path(args.input)
    output_root = Path(args.output)
    config_output = Path(args.config_output)
    base_config = Path(args.base_config)
    models_root = Path(args.models_root)
    results_root = Path(args.results_root)

    print("Few-shot target-center adaptation")
    print(f"Raw root: {raw_root}")
    print(f"Target center: {args.target_center}")
    print(f"Source centers: {args.source_centers}")
    print(f"Fractions: {args.fractions}")

    for fraction in args.fractions:
        build_few_shot_fold(
            fraction=fraction,
            raw_root=raw_root,
            source_centers=args.source_centers,
            target_center=args.target_center,
            output_root=output_root,
            config_output=config_output,
            base_config=base_config,
            models_root=models_root,
            results_root=results_root,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            force=args.force,
        )


if __name__ == "__main__":
    main()