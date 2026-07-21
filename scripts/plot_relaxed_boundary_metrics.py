import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


MODEL_FILES = {
    "IR1 full fine-tuning": (
        "results/relaxed_boundary_metrics/"
        "infrared1_full_finetuning/"
        "relaxed_boundary_metrics_summary.csv"
    ),
    "IR2 encoder-only": (
        "results/relaxed_boundary_metrics/"
        "infrared2_encoder_only/"
        "relaxed_boundary_metrics_summary.csv"
    ),
    "IR3 frozen encoder": (
        "results/relaxed_boundary_metrics/"
        "infrared3_frozen_encoder/"
        "relaxed_boundary_metrics_summary.csv"
    ),
}


def read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            converted = {}

            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value

            rows.append(converted)

    return rows


def load_all_results(base_dir: Path) -> Dict[str, List[Dict[str, float]]]:
    results = {}

    for model_name, relative_path in MODEL_FILES.items():
        path = base_dir / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing summary CSV for {model_name}: {path}"
            )

        results[model_name] = read_summary_csv(path)

    return results


def plot_relaxed_f1_by_radius(
    results: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, rows in results.items():
        radii = [
            int(row["radius"])
            for row in rows
        ]

        relaxed_f1 = [
            float(row["global_relaxed_f1"])
            for row in rows
        ]

        plt.plot(
            radii,
            relaxed_f1,
            marker="o",
            linewidth=2,
            label=model_name,
        )

        for x_value, y_value in zip(radii, relaxed_f1):
            plt.text(
                x_value,
                y_value + 0.002,
                f"{y_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xlabel("Boundary tolerance radius (px)")
    plt.ylabel("Global relaxed F1")
    plt.title("Strict vs tolerance-aware F1")
    plt.xticks([0, 1, 2, 3], ["Strict", "1 px", "2 px", "3 px"])
    plt.ylim(0.86, 0.98)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_relaxed_f1_gain(
    results: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    model_names = list(results.keys())

    gain_1px = []
    gain_2px = []
    gain_3px = []

    for model_name in model_names:
        rows = results[model_name]

        by_radius = {
            int(row["radius"]): row
            for row in rows
        }

        strict = float(by_radius[0]["global_relaxed_f1"])

        gain_1px.append(
            float(by_radius[1]["global_relaxed_f1"]) - strict
        )

        gain_2px.append(
            float(by_radius[2]["global_relaxed_f1"]) - strict
        )

        gain_3px.append(
            float(by_radius[3]["global_relaxed_f1"]) - strict
        )

    x_positions = list(range(len(model_names)))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(
        [x - width for x in x_positions],
        gain_1px,
        width,
        label="1 px tolerance",
    )

    plt.bar(
        x_positions,
        gain_2px,
        width,
        label="2 px tolerance",
    )

    plt.bar(
        [x + width for x in x_positions],
        gain_3px,
        width,
        label="3 px tolerance",
    )

    plt.xlabel("Model")
    plt.ylabel("F1 gain vs strict evaluation")
    plt.title("Effect of boundary tolerance on F1")
    plt.xticks(
        x_positions,
        model_names,
        rotation=15,
        ha="right",
    )
    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot relaxed boundary metric results."
    )

    parser.add_argument(
        "--base-dir",
        default=".",
        help="Repository root directory.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/relaxed_boundary_metrics/figures",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    results = load_all_results(base_dir)

    plot_relaxed_f1_by_radius(
        results=results,
        output_path=output_dir / "relaxed_f1_by_radius.png",
    )

    plot_relaxed_f1_gain(
        results=results,
        output_path=output_dir / "relaxed_f1_gain_vs_strict.png",
    )

    print("Saved figures to:", output_dir)


if __name__ == "__main__":
    main()