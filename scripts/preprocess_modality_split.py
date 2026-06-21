import argparse
import random
import shutil
from pathlib import Path
import yaml


MODALITIES = {
    "colour": ["Colour1", "Colour2"],
    "infrared": ["Infrared1", "Infrared2", "Infrared3"],
}


def find_pairs(center_dir: Path):
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

    for img_path, mask_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(mask_path, mask_out / mask_path.name)


def create_config(base_config: Path, output_config: Path, dataset_path: Path, epochs: int):
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["dataset_path"] = str(dataset_path)
    cfg["training"]["epochs"] = epochs

    output_config.parent.mkdir(parents=True, exist_ok=True)

    with open(output_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Open DataSet2")
    parser.add_argument("--output", default="data/modality_split")
    parser.add_argument("--base-config", default="configs/unet_config.yaml")
    parser.add_argument("--config-output", default="configs/modality_split")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    raw_root = Path(args.input)
    output_root = Path(args.output)
    config_output = Path(args.config_output)

    for modality, centers in MODALITIES.items():
        modality_root = output_root / modality

        if modality_root.exists():
            if args.force:
                shutil.rmtree(modality_root)
            else:
                print(f"Skipping existing modality split: {modality_root}")
                continue

        all_pairs = []

        print(f"\n=== {modality.upper()} ===")

        for center in centers:
            center_pairs = find_pairs(raw_root / center)
            all_pairs.extend(center_pairs)
            print(f"{center}: {len(center_pairs)} pairs")

        random.shuffle(all_pairs)

        total = len(all_pairs)
        test_size = int(total * args.test_ratio)
        val_size = int(total * args.val_ratio)

        test_pairs = all_pairs[:test_size]
        val_pairs = all_pairs[test_size:test_size + val_size]
        train_pairs = all_pairs[test_size + val_size:]

        copy_pairs(train_pairs, modality_root, "train")
        copy_pairs(val_pairs, modality_root, "val")
        copy_pairs(test_pairs, modality_root, "test")

        config_path = config_output / f"{modality}_config.yaml"

        create_config(
            base_config=Path(args.base_config),
            output_config=config_path,
            dataset_path=modality_root,
            epochs=args.epochs,
        )

        print(f"\nCreated modality split: {modality}")
        print(f"Train: {len(train_pairs)}")
        print(f"Val:   {len(val_pairs)}")
        print(f"Test:  {len(test_pairs)}")
        print(f"Config: {config_path}")


if __name__ == "__main__":
    main()