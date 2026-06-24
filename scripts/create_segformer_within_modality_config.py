from pathlib import Path
import yaml


SOURCE_CONFIG_DIR = Path("configs/within_modality_center")
OUTPUT_CONFIG_DIR = Path("configs/within_modality_center_segformer")

RESULTS_ROOT = Path("results/within_modality_center_segformer")
MODELS_ROOT = Path("models/within_modality_center_segformer")


def main():
    if not SOURCE_CONFIG_DIR.exists():
        raise FileNotFoundError(
            f"Missing source config directory: {SOURCE_CONFIG_DIR}\n"
            "You need configs/within_modality_center/ before creating SegFormer configs."
        )

    OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(SOURCE_CONFIG_DIR.glob("*.yaml"))

    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {SOURCE_CONFIG_DIR}")

    for src_path in config_paths:
        fold_name = src_path.stem
        dst_path = OUTPUT_CONFIG_DIR / src_path.name

        with src_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Keep the same dataset split, but write SegFormer outputs elsewhere.
        cfg["logging"]["checkpoint_dir"] = str(MODELS_ROOT / fold_name)
        cfg["logging"]["log_dir"] = str(RESULTS_ROOT / fold_name)

        # Safer memory settings for SegFormer on the shared GPU.
        cfg["training"]["batch_size"] = 2
        cfg["training"]["gradient_accumulation_steps"] = 4

        # Make the config explicit, even if train_segformer.py mostly ignores this section.
        cfg["model"]["name"] = "SegFormer"

        with dst_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Created {dst_path}")


if __name__ == "__main__":
    main()