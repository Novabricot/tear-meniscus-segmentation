import argparse
import json
import logging
import os
from pathlib import Path
import random
from PIL import Image


def find_data_pairs(raw_root: Path):
    pairs = []
    raw_root = raw_root.resolve()

    for original_dir in raw_root.rglob("Original"):
        label_dir = original_dir.parent / "Label"
        if not label_dir.exists():
            continue

        for image_path in sorted(original_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue

            label_path = label_dir / image_path.name
            if label_path.exists():
                pairs.append((image_path, label_path))

    return pairs


def prepare_mask(mask_path: Path, output_path: Path):
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask = mask.point(lambda value: 255 if value > 127 else 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(output_path, format="PNG")


def prepare_image(image_path: Path, output_path: Path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")


def write_split_index(split_index, output_dir: Path):
    metadata_path = output_dir / "splits.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(split_index, f, indent=2)


def build_processed_dataset(raw_root: Path, processed_root: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    raw_root = raw_root.resolve()
    processed_root = processed_root.resolve()
    random.seed(seed)

    pairs = find_data_pairs(raw_root)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {raw_root}")

    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    split_index = {"train": [], "val": [], "test": []}
    for split_name, records in splits.items():
        for image_path, mask_path in records:
            relative_name = image_path.name
            image_output = processed_root / split_name / "images" / relative_name
            mask_output = processed_root / split_name / "masks" / relative_name
            prepare_image(image_path, image_output)
            prepare_mask(mask_path, mask_output)
            split_index[split_name].append(relative_name)

    write_split_index(split_index, processed_root)
    return split_index


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tear meniscus dataset for segmentation training")
    parser.add_argument("--input", type=str, default="data/raw", help="Path to raw dataset root")
    parser.add_argument("--output", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set fraction")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed data")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {input_path}")

    if output_path.exists() and args.force:
        logging.info(f"Removing existing processed folder: {output_path}")
        import shutil

        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Preparing processed dataset from {input_path} to {output_path}")
    split_index = build_processed_dataset(
        raw_root=input_path,
        processed_root=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    logging.info("Processed dataset created successfully")
    logging.info(f"Train samples: {len(split_index['train'])}")
    logging.info(f"Val samples: {len(split_index['val'])}")
    logging.info(f"Test samples: {len(split_index['test'])}")


if __name__ == "__main__":
    main()
