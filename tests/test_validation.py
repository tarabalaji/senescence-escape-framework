from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    bootstrap_model_auc,
    calculate_ablation_results,
    compare_models_paired,
    create_validation_summary,
    permutation_test_auc,
    run_validation,
    save_results,
    summarize_conservation,
    summarize_model_performance,
    validate_baseline_tables,
)


@pytest.fixture
def validation_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [
                "full_model",
                "full_model",
                "rap_only",
                "rap_only",
                "rap_plus_pathways",
                "rap_plus_pathways",
            ],
            "held_out_cell_line": [
                "MCF7",
                "T47D",
                "MCF7",
                "T47D",
                "MCF7",
                "T47D",
            ],
            "roc_auc": [
                0.80,
                0.82,
                0.75,
                0.77,
                0.84,
                0.85,
            ],
            "balanced_accuracy": [
                0.70,
                0.72,
                0.66,
                0.68,
                0.75,
                0.76,
            ],
            "precision": [
                0.71,
                0.73,
                0.67,
                0.69,
                0.76,
                0.77,
            ],
            "recall": [
                0.69,
                0.71,
                0.65,
                0.67,
                0.74,
                0.75,
            ],
            "f1": [
                0.70,
                0.72,
                0.66,
                0.68,
                0.75,
                0.76,
            ],
            "average_precision": [
                0.79,
                0.81,
                0.74,
                0.76,
                0.83,
                0.84,
            ],
        }
    )


@pytest.fixture
def validation_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []

    for model, noise in [
        ("full_model", 0.18),
        ("rap_only", 0.25),
        ("rap_plus_pathways", 0.12),
    ]:
        for cell_line in ["MCF7", "T47D"]:
            for index in range(40):
                label = int(index >= 20)
                probability = np.clip(
                    label + rng.normal(0, noise),
                    0,
                    1,
                )

                rows.append(
                    {
                        "cell_id": (f"{cell_line}_{index}"),
                        "model": model,
                        "held_out_cell_line": cell_line,
                        "transition_label": label,
                        "transition_probability": probability,
                    }
                )

    return pd.DataFrame(rows)


def test_validate_baseline_tables(
    validation_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    validate_baseline_tables(
        validation_metrics,
        validation_predictions,
    )


def test_validate_baseline_tables_missing_column(
    validation_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    invalid = validation_metrics.drop(columns=["roc_auc"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_baseline_tables(
            invalid,
            validation_predictions,
        )


def test_summarize_model_performance(
    validation_metrics: pd.DataFrame,
) -> None:
    summary = summarize_model_performance(validation_metrics)

    assert summary.iloc[0]["model"] == ("rap_plus_pathways")

    assert summary.loc[
        summary["model"].eq("full_model"),
        "roc_auc_mean",
    ].iloc[0] == pytest.approx(0.81)


def test_calculate_ablation_results(
    validation_metrics: pd.DataFrame,
) -> None:
    results = calculate_ablation_results(
        validation_metrics,
        reference_model="full_model",
    )

    pathway_delta = results.loc[
        results["model"].eq("rap_plus_pathways"),
        "delta_roc_auc_mean",
    ].iloc[0]

    rap_delta = results.loc[
        results["model"].eq("rap_only"),
        "delta_roc_auc_mean",
    ].iloc[0]

    assert pathway_delta > 0
    assert rap_delta < 0


def test_bootstrap_model_auc(
    validation_predictions: pd.DataFrame,
) -> None:
    results = bootstrap_model_auc(
        validation_predictions,
        model_name="full_model",
        iterations=100,
    )

    assert len(results) == 2

    assert (
        results["roc_auc_ci_lower"]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (
        results["roc_auc_ci_upper"]
        .between(
            0,
            1,
        )
        .all()
    )


def test_permutation_test_auc(
    validation_predictions: pd.DataFrame,
) -> None:
    results = permutation_test_auc(
        validation_predictions,
        model_name="full_model",
        iterations=100,
    )

    assert len(results) == 2

    assert (
        results["permutation_p_value"]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (results["observed_roc_auc"] > results["null_roc_auc_mean"]).all()


def test_compare_models_paired(
    validation_predictions: pd.DataFrame,
) -> None:
    results = compare_models_paired(
        validation_predictions,
        model_a="rap_plus_pathways",
        model_b="rap_only",
    )

    assert len(results) == 2

    assert (results["delta_roc_auc_a_minus_b"] >= 0).all()


def test_summarize_conservation() -> None:
    markers = pd.DataFrame({"gene": ["A", "B", "C"]})
    features = pd.DataFrame({"feature": ["X", "Y"]})

    summary = summarize_conservation(
        markers,
        features,
    )

    counts = dict(
        zip(
            summary["validation_component"],
            summary["count"],
        )
    )

    assert counts["conserved_gene_markers"] == 3
    assert counts["conserved_features"] == 2


def test_create_validation_summary(
    validation_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    model_summary = summarize_model_performance(validation_metrics)

    ablation_results = calculate_ablation_results(validation_metrics)

    bootstrap_results = bootstrap_model_auc(
        validation_predictions,
        "full_model",
        iterations=50,
    )

    permutation_results = permutation_test_auc(
        validation_predictions,
        "full_model",
        iterations=50,
    )

    conservation_summary = summarize_conservation(
        pd.DataFrame({"gene": ["A"]}),
        pd.DataFrame({"feature": ["X"]}),
    )

    summary = create_validation_summary(
        model_summary,
        ablation_results,
        bootstrap_results,
        permutation_results,
        conservation_summary,
    )

    assert not summary.empty

    assert "best_mean_roc_auc_model" in set(summary["finding"])


def test_run_validation(
    validation_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    results = run_validation(
        validation_metrics,
        validation_predictions,
        bootstrap_iterations=50,
        permutation_iterations=50,
        conserved_markers=pd.DataFrame({"gene": ["A", "B"]}),
        conserved_features=pd.DataFrame({"feature": ["X"]}),
    )

    assert len(results) == 6

    model_summary = results[0]
    validation_summary = results[-1]

    assert len(model_summary) == 3
    assert not validation_summary.empty


def test_save_results(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame({"value": [1]})

    save_results(
        table,
        table,
        table,
        table,
        table,
        table,
        tmp_path,
    )

    expected_files = [
        "validation_model_summary.csv",
        "validation_ablation_results.csv",
        "validation_bootstrap_results.csv",
        "validation_permutation_results.csv",
        "validation_conservation_summary.csv",
        "validation_summary.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
