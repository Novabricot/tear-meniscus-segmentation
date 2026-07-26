#!/usr/bin/env python3
"""
Create repeated-seed configs for IR3 scratch vs encoder-only experiments.

Generated configs:
    configs/repeated_seeds/ir3_scratch_seed44.yaml
    configs/repeated_seeds/ir3_scratch_seed45.yaml
    configs/repeated_seeds/ir3_scratch_seed46.yaml
    configs/repeated_seeds/ir3_encoder_only_seed44.yaml
    configs/repeated_seeds/ir3_encoder_only_seed45.yaml
    configs/repeated_seeds/ir3_encoder_only_seed46.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


SEEDS = [44, 45, 46]

BASE_SCRATCH_CONFIG = Path("configs/feature_extraction/infrared3_scratch.yaml")
BASE_ENCODER_ONLY_CONFIG = Path("configs/feature_extraction/infrared3_encoder_only.yaml")

OUTPUT_CONFIG_DIR = Path("configs/repeated_seeds")


def recursively_update_key(config: Any, key_name: str, new_value: Any) -> None:
    if isinstance(config, dict):
        for key, value in config.items():
            if key == key_name:
                config[key] = new_value
            else:
                recursively_update_key(value, key_name, new_value)
    elif isinstance(config, list):
        for item in config:
            recursively_update_key(item, key_name, new_value)


def recursively_replace_string(config: Any, old: str, new: str) -> Any:
    if isinstance(config, dict):
        return {key: recursively_replace_string(value, old, new) for key, value in config.items()}
    if isinstance(config, list):
        return [recursively_replace_string(item, old, new) for item in config]
    if isinstance(config, str):
        return config.replace(old, new)
    return config


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing base config: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def make_config(base_path: Path, experiment_name: str, seed: int) -> Path:
    config = load_yaml(base_path)
    config = copy.deepcopy(config)

    recursively_update_key(config, "random_seed", seed)
    recursively_update_key(config, "seed", seed)

    if "scratch" in experiment_name:
        config = recursively_replace_string(
            config,
            "results/feature_extraction/infrared3_scratch",
            f"results/repeated_seeds/{experiment_name}",
        )
        config = recursively_replace_string(
            config,
            "models/feature_extraction/infrared3_scratch",
            f"models/repeated_seeds/{experiment_name}",
        )
    else:
        config = recursively_replace_string(
            config,
            "results/feature_extraction/infrared3_encoder_only",
            f"results/repeated_seeds/{experiment_name}",
        )
        config = recursively_replace_string(
            config,
            "models/feature_extraction/infrared3_encoder_only",
            f"models/repeated_seeds/{experiment_name}",
        )

    output_path = OUTPUT_CONFIG_DIR / f"{experiment_name}.yaml"
    save_yaml(output_path, config)
    return output_path


def main() -> None:
    OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    generated = []

    for seed in SEEDS:
        generated.append(make_config(BASE_SCRATCH_CONFIG, f"ir3_scratch_seed{seed}", seed))
        generated.append(make_config(BASE_ENCODER_ONLY_CONFIG, f"ir3_encoder_only_seed{seed}", seed))

    print("Generated configs:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()