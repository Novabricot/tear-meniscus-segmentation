import argparse
import random
import shutil
from pathlib import Path

import yaml


COLOUR_CENTERS = ["Colour1", "Colour2"]
INFRARED_CENTERS = ["Infrared1", "Infrared2", "Infrared3"]


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
        # Prefix with center name to avoid overwriting files
        # if two centers use the same filenames.
        out_name = f"{center_name}__{img_path.name}"

        shutil.copy2(img_path, img_out / out_name)
        shutil.copy2(mask_path, mask_out / out_name)


def make_train_val_split(pairs, val_ratio: float, seed: int):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    val_size = int(len(pairs) * val_ratio)

    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    return train_pairs, val_pairs


def create_config(
    base_config: Path,
    output_config: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    epochs: int,
    seed: int,
):
    with base_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["dataset_path"] = str(dataset_path)
    cfg["data"]["random_seed"] = seed

    cfg["training"]["epochs"] = epochs

    cfg["logging"]["checkpoint_dir"] = str(checkpoint_dir)
    cfg["logging"]["log_dir"] = str(log_dir)

    output_config.parent.mkdir(parents=True, exist_ok=True)

    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_fold(
    fold_name: str,
    train_centers: list[str],
    test_centers: list[str],
    all_pairs_by_center: dict[str, list],
    output_root: Path,
    config_output: Path,
    base_config: Path,
    models_root: Path,
    results_root: Path,
    val_ratio: float,
    epochs: int,
    seed: int,
    force: bool,
):
    fold_root = output_root / fold_name

    if fold_root.exists():
        if force:
            shutil.rmtree(fold_root)
        else:
            print(f"Skipping existing fold: {fold_root}")
            return

    train_val_pairs = []
    for center in train_centers:
        train_val_pairs.extend(all_pairs_by_center[center])

    test_pairs = []
    for center in test_centers:
        test_pairs.extend(all_pairs_by_center[center])

    train_pairs, val_pairs = make_train_val_split(
        pairs=train_val_pairs,
        val_ratio=val_ratio,
        seed=seed,
    )

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
    )

    print(f"\nCreated fold: {fold_name}")
    print(f"Train centers: {train_centers}")
    print(f"Test centers:  {test_centers}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")
    print(f"Config: {config_path}")
    print(f"Results: {results_root / fold_name}")
    print(f"Models:  {models_root / fold_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create within-modality center-shift splits for tear meniscus segmentation."
    )

    parser.add_argument("--input", default="data/raw/Open DataSet2")
    parser.add_argument("--output", default="data/within_modality_center")
    parser.add_argument("--base-config", default="configs/unet_config.yaml")
    parser.add_argument("--config-output", default="configs/within_modality_center")
    parser.add_argument("--models-root", default="models/within_modality_center")
    parser.add_argument("--results-root", default="results/within_modality_center")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    raw_root = Path(args.input)
    output_root = Path(args.output)
    config_output = Path(args.config_output)
    base_config = Path(args.base_config)
    models_root = Path(args.models_root)
    results_root = Path(args.results_root)

    all_centers = COLOUR_CENTERS + INFRARED_CENTERS
    all_pairs_by_center = {}

    for center in all_centers:
        center_dir = raw_root / center
        raw_pairs = find_image_mask_pairs(center_dir)

        pairs = [(center, img_path, mask_path) for img_path, mask_path in raw_pairs]
        all_pairs_by_center[center] = pairs

        print(f"{center}: {len(pairs)} pairs")

    # Colour-only center shift:
    # Train on Colour1, test on Colour2.
    build_fold(
        fold_name="colour_train_Colour1_test_Colour2",
        train_centers=["Colour1"],
        test_centers=["Colour2"],
        all_pairs_by_center=all_pairs_by_center,
        output_root=output_root,
        config_output=config_output,
        base_config=base_config,
        models_root=models_root,
        results_root=results_root,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        seed=args.seed,
        force=args.force,
    )

    # Colour-only center shift:
    # Train on Colour2, test on Colour1.
    build_fold(
        fold_name="colour_train_Colour2_test_Colour1",
        train_centers=["Colour2"],
        test_centers=["Colour1"],
        all_pairs_by_center=all_pairs_by_center,
        output_root=output_root,
        config_output=config_output,
        base_config=base_config,
        models_root=models_root,
        results_root=results_root,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        seed=args.seed,
        force=args.force,
    )

    # Infrared-only leave-one-center-out:
    # Train on two infrared centers, test on the third.
    for test_center in INFRARED_CENTERS:
        train_centers = [
            center for center in INFRARED_CENTERS if center != test_center
        ]

        build_fold(
            fold_name=f"infrared_leave_{test_center}_out",
            train_centers=train_centers,
            test_centers=[test_center],
            all_pairs_by_center=all_pairs_by_center,
            output_root=output_root,
            config_output=config_output,
            base_config=base_config,
            models_root=models_root,
            results_root=results_root,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            seed=args.seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()