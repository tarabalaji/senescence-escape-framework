from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_BASELINE_METRICS_PATH = Path("results/tables/baseline_model_metrics.csv")
DEFAULT_BASELINE_PREDICTIONS_PATH = Path(
    "results/tables/baseline_model_predictions.csv"
)
DEFAULT_ESCAPE_MARKERS_PATH = Path("results/tables/escape_prone_conserved_markers.csv")
DEFAULT_ESCAPE_FEATURES_PATH = Path(
    "results/tables/escape_prone_conserved_features.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")

DEFAULT_REFERENCE_MODEL = "full_model"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_PERMUTATION_ITERATIONS = 1000


def validate_baseline_tables(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    required_metric_columns = {
        "model",
        "held_out_cell_line",
        "roc_auc",
        "balanced_accuracy",
        "f1",
    }
    missing_metrics = required_metric_columns.difference(metrics.columns)

    if missing_metrics:
        raise ValueError(
            f"Baseline metrics are missing columns: {sorted(missing_metrics)}"
        )

    required_prediction_columns = {
        "cell_id",
        "model",
        "held_out_cell_line",
        "transition_label",
        "transition_probability",
    }
    missing_predictions = required_prediction_columns.difference(predictions.columns)

    if missing_predictions:
        raise ValueError(
            f"Baseline predictions are missing columns: {sorted(missing_predictions)}"
        )

    probabilities = pd.to_numeric(
        predictions["transition_probability"],
        errors="coerce",
    )
    labels = pd.to_numeric(
        predictions["transition_label"],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise ValueError("Transition probabilities must be numeric")

    if not probabilities.between(0, 1).all():
        raise ValueError("Transition probabilities must be between 0 and 1")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Transition labels must contain only 0 and 1")


def summarize_model_performance(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
    ]

    available_metrics = [
        column for column in metric_columns if column in metrics.columns
    ]

    grouped = metrics.groupby("model")[available_metrics].agg(
        ["mean", "std", "min", "max"]
    )

    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]

    summary = grouped.reset_index()

    if "roc_auc_mean" in summary.columns:
        summary = summary.sort_values(
            "roc_auc_mean",
            ascending=False,
        )

    return summary.reset_index(drop=True)


def calculate_ablation_results(
    metrics: pd.DataFrame,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
) -> pd.DataFrame:
    reference = metrics.loc[metrics["model"].astype(str).eq(reference_model)].copy()

    if reference.empty:
        raise ValueError(f"Reference model not found: {reference_model}")

    metric_columns = [
        "roc_auc",
        "balanced_accuracy",
        "f1",
        "average_precision",
    ]
    available_metrics = [
        column for column in metric_columns if column in metrics.columns
    ]

    merged = metrics.merge(
        reference[
            [
                "held_out_cell_line",
                *available_metrics,
            ]
        ],
        on="held_out_cell_line",
        suffixes=("", "_reference"),
    )

    merged = merged.loc[~merged["model"].astype(str).eq(reference_model)].copy()

    for metric in available_metrics:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_reference"]

    aggregation = {
        f"delta_{metric}": ["mean", "std", "min", "max"] for metric in available_metrics
    }

    ablation = merged.groupby("model").agg(aggregation)

    ablation.columns = [
        f"{column}_{statistic}" for column, statistic in ablation.columns
    ]

    ablation = ablation.reset_index()
    ablation.insert(1, "reference_model", reference_model)

    if "delta_roc_auc_mean" in ablation.columns:
        ablation = ablation.sort_values(
            "delta_roc_auc_mean",
            ascending=False,
        )

    return ablation.reset_index(drop=True)


def bootstrap_model_auc(
    predictions: pd.DataFrame,
    model_name: str,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    selected = predictions.loc[predictions["model"].astype(str).eq(model_name)].copy()

    if selected.empty:
        raise ValueError(f"Model not found: {model_name}")

    rng = np.random.default_rng(random_state)
    rows = []

    for held_out, fold in selected.groupby(
        "held_out_cell_line",
        observed=True,
    ):
        labels = fold["transition_label"].to_numpy(dtype=int)
        probabilities = fold["transition_probability"].to_numpy(dtype=float)

        if np.unique(labels).size < 2:
            continue

        aucs = []

        for _ in range(iterations):
            indices = rng.integers(
                0,
                len(fold),
                size=len(fold),
            )
            sampled_labels = labels[indices]

            if np.unique(sampled_labels).size < 2:
                continue

            aucs.append(
                roc_auc_score(
                    sampled_labels,
                    probabilities[indices],
                )
            )

        if not aucs:
            continue

        rows.append(
            {
                "model": model_name,
                "held_out_cell_line": str(held_out),
                "bootstrap_iterations_requested": iterations,
                "bootstrap_iterations_valid": len(aucs),
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_standard_deviation": float(np.std(aucs, ddof=1)),
                "roc_auc_ci_lower": float(np.quantile(aucs, 0.025)),
                "roc_auc_ci_upper": float(np.quantile(aucs, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def permutation_test_auc(
    predictions: pd.DataFrame,
    model_name: str,
    iterations: int = DEFAULT_PERMUTATION_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    selected = predictions.loc[predictions["model"].astype(str).eq(model_name)].copy()

    if selected.empty:
        raise ValueError(f"Model not found: {model_name}")

    rng = np.random.default_rng(random_state)
    rows = []

    for held_out, fold in selected.groupby(
        "held_out_cell_line",
        observed=True,
    ):
        labels = fold["transition_label"].to_numpy(dtype=int)
        probabilities = fold["transition_probability"].to_numpy(dtype=float)

        if np.unique(labels).size < 2:
            continue

        observed_auc = float(roc_auc_score(labels, probabilities))

        null_aucs = np.empty(iterations, dtype=float)

        for index in range(iterations):
            permuted_labels = rng.permutation(labels)
            null_aucs[index] = roc_auc_score(
                permuted_labels,
                probabilities,
            )

        p_value = float((1 + np.sum(null_aucs >= observed_auc)) / (iterations + 1))

        rows.append(
            {
                "model": model_name,
                "held_out_cell_line": str(held_out),
                "observed_roc_auc": observed_auc,
                "null_roc_auc_mean": float(np.mean(null_aucs)),
                "null_roc_auc_standard_deviation": float(np.std(null_aucs, ddof=1)),
                "permutation_p_value": p_value,
                "iterations": iterations,
            }
        )

    return pd.DataFrame(rows)


def compare_models_paired(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    first = predictions.loc[
        predictions["model"].astype(str).eq(model_a),
        [
            "cell_id",
            "held_out_cell_line",
            "transition_label",
            "transition_probability",
        ],
    ].rename(
        columns={
            "transition_probability": "probability_a",
        }
    )

    second = predictions.loc[
        predictions["model"].astype(str).eq(model_b),
        [
            "cell_id",
            "held_out_cell_line",
            "transition_label",
            "transition_probability",
        ],
    ].rename(
        columns={
            "transition_probability": "probability_b",
        }
    )

    if first.empty:
        raise ValueError(f"Model not found: {model_a}")
    if second.empty:
        raise ValueError(f"Model not found: {model_b}")

    merged = first.merge(
        second,
        on=[
            "cell_id",
            "held_out_cell_line",
            "transition_label",
        ],
        how="inner",
    )

    rows = []

    for held_out, fold in merged.groupby(
        "held_out_cell_line",
        observed=True,
    ):
        labels = fold["transition_label"].to_numpy(dtype=int)

        if np.unique(labels).size < 2:
            continue

        auc_a = float(
            roc_auc_score(
                labels,
                fold["probability_a"],
            )
        )
        auc_b = float(
            roc_auc_score(
                labels,
                fold["probability_b"],
            )
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "held_out_cell_line": str(held_out),
                "roc_auc_a": auc_a,
                "roc_auc_b": auc_b,
                "delta_roc_auc_a_minus_b": auc_a - auc_b,
                "n_cells": len(fold),
            }
        )

    return pd.DataFrame(rows)


def summarize_conservation(
    conserved_markers: pd.DataFrame | None,
    conserved_features: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    if conserved_markers is not None:
        rows.append(
            {
                "validation_component": "conserved_gene_markers",
                "count": len(conserved_markers),
            }
        )

    if conserved_features is not None:
        rows.append(
            {
                "validation_component": "conserved_features",
                "count": len(conserved_features),
            }
        )

    return pd.DataFrame(rows)


def create_validation_summary(
    model_summary: pd.DataFrame,
    ablation_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    permutation_results: pd.DataFrame,
    conservation_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not model_summary.empty:
        best_model = model_summary.iloc[0]

        rows.append(
            {
                "category": "model_performance",
                "finding": "best_mean_roc_auc_model",
                "value": str(best_model["model"]),
                "numeric_value": float(best_model["roc_auc_mean"]),
            }
        )

    if not ablation_results.empty:
        best_ablation = ablation_results.iloc[0]

        rows.append(
            {
                "category": "ablation",
                "finding": "best_model_relative_to_reference",
                "value": str(best_ablation["model"]),
                "numeric_value": float(best_ablation["delta_roc_auc_mean"]),
            }
        )

    if not bootstrap_results.empty:
        rows.append(
            {
                "category": "uncertainty",
                "finding": "mean_bootstrap_ci_width",
                "value": "roc_auc_95_percent_interval",
                "numeric_value": float(
                    (
                        bootstrap_results["roc_auc_ci_upper"]
                        - bootstrap_results["roc_auc_ci_lower"]
                    ).mean()
                ),
            }
        )

    if not permutation_results.empty:
        rows.append(
            {
                "category": "significance",
                "finding": "maximum_permutation_p_value",
                "value": str(permutation_results["model"].iloc[0]),
                "numeric_value": float(
                    permutation_results["permutation_p_value"].max()
                ),
            }
        )

    for row in conservation_summary.itertuples(index=False):
        rows.append(
            {
                "category": "cross_cell_line_conservation",
                "finding": row.validation_component,
                "value": row.validation_component,
                "numeric_value": float(row.count),
            }
        )

    return pd.DataFrame(rows)


def run_validation(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_model: str = DEFAULT_REFERENCE_MODEL,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    permutation_iterations: int = DEFAULT_PERMUTATION_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    conserved_markers: pd.DataFrame | None = None,
    conserved_features: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_baseline_tables(metrics, predictions)

    model_summary = summarize_model_performance(metrics)

    ablation_results = calculate_ablation_results(
        metrics,
        reference_model=reference_model,
    )

    bootstrap_results = bootstrap_model_auc(
        predictions,
        model_name=reference_model,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )

    permutation_results = permutation_test_auc(
        predictions,
        model_name=reference_model,
        iterations=permutation_iterations,
        random_state=random_state,
    )

    conservation_summary = summarize_conservation(
        conserved_markers,
        conserved_features,
    )

    validation_summary = create_validation_summary(
        model_summary,
        ablation_results,
        bootstrap_results,
        permutation_results,
        conservation_summary,
    )

    return (
        model_summary,
        ablation_results,
        bootstrap_results,
        permutation_results,
        conservation_summary,
        validation_summary,
    )


def save_results(
    model_summary: pd.DataFrame,
    ablation_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    permutation_results: pd.DataFrame,
    conservation_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_summary.to_csv(
        output_directory / "validation_model_summary.csv",
        index=False,
    )

    ablation_results.to_csv(
        output_directory / "validation_ablation_results.csv",
        index=False,
    )

    bootstrap_results.to_csv(
        output_directory / "validation_bootstrap_results.csv",
        index=False,
    )

    permutation_results.to_csv(
        output_directory / "validation_permutation_results.csv",
        index=False,
    )

    conservation_summary.to_csv(
        output_directory / "validation_conservation_summary.csv",
        index=False,
    )

    validation_summary.to_csv(
        output_directory / "validation_summary.csv",
        index=False,
    )


def _read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SenEscape model performance, uncertainty, "
            "significance, ablations, and conserved findings."
        )
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_BASELINE_METRICS_PATH,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS_PATH,
    )

    parser.add_argument(
        "--conserved-markers",
        type=Path,
        default=DEFAULT_ESCAPE_MARKERS_PATH,
    )

    parser.add_argument(
        "--conserved-features",
        type=Path,
        default=DEFAULT_ESCAPE_FEATURES_PATH,
    )

    parser.add_argument(
        "--reference-model",
        type=str,
        default=DEFAULT_REFERENCE_MODEL,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )

    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=DEFAULT_PERMUTATION_ITERATIONS,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    args = parser.parse_args()

    for path in [args.metrics, args.predictions]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    results = run_validation(
        pd.read_csv(args.metrics),
        pd.read_csv(args.predictions),
        reference_model=args.reference_model,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        conserved_markers=_read_optional_table(args.conserved_markers),
        conserved_features=_read_optional_table(args.conserved_features),
    )

    save_results(
        *results,
        output_directory=args.output_directory,
    )

    model_summary = results[0]
    permutation_results = results[3]

    print(f"Validation completed for {len(model_summary)} models")

    if not model_summary.empty:
        best = model_summary.iloc[0]

        print(
            f"Best model by mean ROC-AUC: {best['model']} ({best['roc_auc_mean']:.3f})"
        )

    if not permutation_results.empty:
        maximum_p_value = permutation_results["permutation_p_value"].max()

        print(f"Maximum reference-model permutation p-value: {maximum_p_value:.4f}")

    print(f"Saved validation outputs to {args.output_directory}")


if __name__ == "__main__":
    main()
