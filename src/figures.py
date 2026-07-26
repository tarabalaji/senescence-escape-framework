from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CONFIG = Path("config/project_config.json")


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_baseline_models(tables: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(tables / "baseline_model_summary.csv")
    data = data.sort_values("roc_auc_mean", ascending=True)

    plt.figure(figsize=(9, 5.5))
    plt.barh(data["model"], data["roc_auc_mean"])
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Mean leave-one-cell-line-out ROC-AUC")
    plt.ylabel("Model")
    plt.title("Predictive performance of baseline feature groups")
    save_figure(figures / "baseline_model_comparison.png", dpi)


def plot_top_interventions(
    tables: Path,
    figures: Path,
    dpi: int,
    top_targets: int,
) -> None:
    data = pd.read_csv(tables / "top_predicted_interventions.csv")
    data = data.head(top_targets).sort_values(
        "mean_predicted_escape_reduction", ascending=True
    )
    labels = data["target"].astype(str) + " (" + data["intervention"].astype(str) + ")"

    plt.figure(figsize=(9, 6.5))
    plt.barh(labels, data["mean_predicted_escape_reduction"])
    plt.xlabel("Mean predicted reduction in escape probability")
    plt.ylabel("Simulated intervention")
    plt.title("Top computationally predicted interventions")
    save_figure(figures / "top_predicted_interventions.png", dpi)


def plot_target_priorities(
    tables: Path,
    figures: Path,
    dpi: int,
    top_targets: int,
) -> None:
    data = pd.read_csv(tables / "target_priority_rankings.csv")
    data = data.head(top_targets).sort_values("priority_score", ascending=True)

    plt.figure(figsize=(8.5, 6))
    plt.barh(data["gene"], data["priority_score"])
    plt.xlabel("Integrated target-priority score")
    plt.ylabel("Gene")
    plt.title("Top targets supported across computational evidence")
    save_figure(figures / "target_priority_ranking.png", dpi)


def plot_sensitivity(tables: Path, figures: Path, dpi: int) -> None:
    path = tables / "perturbation_sensitivity_stability.csv"
    if not path.exists():
        return

    data = pd.read_csv(path)
    for rap_fraction, group in data.groupby("rap_effect_fraction"):
        group = group.sort_values("perturbation_strength")
        plt.plot(
            group["perturbation_strength"],
            group["top_k_overlap_with_reference"],
            marker="o",
            label=f"RAP fraction={rap_fraction}",
        )

    plt.xlabel("Perturbation strength")
    plt.ylabel("Top-k overlap with reference ranking")
    plt.ylim(0, 1.05)
    plt.title("Robustness of intervention ranking")
    plt.legend()
    save_figure(figures / "perturbation_sensitivity.png", dpi)


def plot_bootstrap_stability(tables: Path, figures: Path, dpi: int) -> None:
    path = tables / "perturbation_bootstrap_stability.csv"
    if not path.exists():
        return

    data = pd.read_csv(path).head(15).sort_values("top_k_frequency", ascending=True)

    plt.figure(figsize=(8.5, 6))
    plt.barh(data["target"], data["top_k_frequency"])
    plt.xlabel("Fraction of bootstrap samples appearing in top k")
    plt.ylabel("Target")
    plt.xlim(0, 1.0)
    plt.title("Bootstrap stability of prioritized interventions")
    save_figure(figures / "perturbation_bootstrap_stability.png", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate science-fair-ready figures from saved results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tables = Path(config["paths"]["tables"])
    figures = Path(config["paths"]["figures"])
    dpi = int(config["figures"]["dpi"])
    top_targets = int(config["figures"]["top_targets"])

    required = [
        tables / "baseline_model_summary.csv",
        tables / "top_predicted_interventions.csv",
        tables / "target_priority_rankings.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required figure inputs are missing: {missing}")

    plot_baseline_models(tables, figures, dpi)
    plot_top_interventions(tables, figures, dpi, top_targets)
    plot_target_priorities(tables, figures, dpi, top_targets)
    plot_sensitivity(tables, figures, dpi)
    plot_bootstrap_stability(tables, figures, dpi)

    print(f"Saved research-fair figures to {figures}")


if __name__ == "__main__":
    main()
