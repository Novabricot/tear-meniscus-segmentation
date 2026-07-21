import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


MODEL_FILES = {
    "IR1 full fine-tuning": (
        "results/uncertainty_zones/"
        "infrared1_full_finetuning/"
        "uncertainty_zones_summary.csv"
    ),
    "IR2 encoder-only": (
        "results/uncertainty_zones/"
        "infrared2_encoder_only/"
        "uncertainty_zones_summary.csv"
    ),
    "IR3 frozen encoder": (
        "results/uncertainty_zones/"
        "infrared3_frozen_encoder/"
        "uncertainty_zones_summary.csv"
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


def plot_error_percentage_in_uncertain_boundary(
    results: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, rows in results.items():
        radii = [
            int(row["radius"])
            for row in rows
        ]

        percentages = [
            100.0
            * float(row["global_pct_errors_in_uncertain_boundary"])
            for row in rows
        ]

        plt.plot(
            radii,
            percentages,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Uncertain boundary radius (px)")
    plt.ylabel("Errors inside uncertain boundary (%)")
    plt.title("Model errors near uncertain annotation boundaries")
    plt.xticks([1, 2, 3])
    plt.ylim(0, 100)
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


def plot_error_rate_by_zone_radius_1(
    results: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    model_names = list(results.keys())

    certain_positive = []
    uncertain_boundary = []
    certain_negative = []

    for model_name in model_names:
        rows = results[model_name]

        radius_1_rows = [
            row
            for row in rows
            if int(row["radius"]) == 1
        ]

        if len(radius_1_rows) != 1:
            raise ValueError(
                f"Expected exactly one radius=1 row for {model_name}"
            )

        row = radius_1_rows[0]

        certain_positive.append(
            100.0
            * float(row["global_error_rate_certain_positive"])
        )

        uncertain_boundary.append(
            100.0
            * float(row["global_error_rate_uncertain_boundary"])
        )

        certain_negative.append(
            100.0
            * float(row["global_error_rate_certain_negative"])
        )

    x_positions = list(range(len(model_names)))
    width = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(
        [x - width for x in x_positions],
        certain_positive,
        width,
        label="Certain positive",
    )

    plt.bar(
        x_positions,
        uncertain_boundary,
        width,
        label="Uncertain boundary",
    )

    plt.bar(
        [x + width for x in x_positions],
        certain_negative,
        width,
        label="Certain negative",
    )

    plt.xlabel("Model")
    plt.ylabel("Error rate inside region (%)")
    plt.title("Error rate by uncertainty region, radius = 1 px")
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
        description=(
            "Plot uncertainty zone analysis results."
        )
    )

    parser.add_argument(
        "--base-dir",
        default=".",
        help="Repository root directory.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/uncertainty_zones/figures",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    results = load_all_results(base_dir)

    plot_error_percentage_in_uncertain_boundary(
        results=results,
        output_path=(
            output_dir
            / "pct_errors_in_uncertain_boundary.png"
        ),
    )

    plot_error_rate_by_zone_radius_1(
        results=results,
        output_path=(
            output_dir
            / "error_rate_by_zone_radius1.png"
        ),
    )

    print(
        "Saved figures to:",
        output_dir,
    )


if __name__ == "__main__":
    main()