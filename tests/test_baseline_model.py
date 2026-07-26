from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.baseline_model import (
    evaluate_logistic_baseline,
    evaluate_majority_baseline,
    fit_baseline_coefficients,
    identify_feature_groups,
    run_baseline_analysis,
    save_results,
    summarize_baseline_metrics,
    validate_inputs,
)


@pytest.fixture
def baseline_adata() -> ad.AnnData:
    obs_rows = []

    for cell_line in ["MCF7", "T47D"]:
        for index in range(40):
            label = int(index >= 20)

            obs_rows.append(
                {
                    "cell_line": cell_line,
                    "condition": "REPOP" if label else "TIS",
                }
            )

    obs = pd.DataFrame(
        obs_rows,
        index=[f"cell_{index}" for index in range(len(obs_rows))],
    )

    return ad.AnnData(
        X=np.zeros((len(obs), 2)),
        obs=obs,
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )


@pytest.fixture
def baseline_feature_table(
    baseline_adata: ad.AnnData,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    labels = (
        baseline_adata.obs["condition"].astype(str).eq("REPOP").astype(int).to_numpy()
    )

    signal = labels + rng.normal(0, 0.10, len(labels))

    return pd.DataFrame(
        {
            "cell_id": baseline_adata.obs_names.astype(str),
            "rap_score": signal,
            "pathway__e2f_targets": (signal + rng.normal(0, 0.05, len(labels))),
            "pathway__g2_m_checkpoint": (signal + rng.normal(0, 0.05, len(labels))),
            "regulator__E2F1": (signal + rng.normal(0, 0.05, len(labels))),
            "regulator__FOS": ((1 - labels) + rng.normal(0, 0.10, len(labels))),
            "transition_label": labels,
        }
    )


def test_validate_inputs(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    indexed = validate_inputs(
        baseline_adata,
        baseline_feature_table,
    )

    assert indexed.index.equals(pd.Index(baseline_adata.obs_names.astype(str)))

    assert "transition_label" in indexed.columns


def test_validate_inputs_missing_column(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    invalid = baseline_feature_table.drop(columns=["transition_label"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_inputs(
            baseline_adata,
            invalid,
        )


def test_validate_inputs_duplicate_cell_ids(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    invalid = baseline_feature_table.copy()

    invalid.loc[1, "cell_id"] = invalid.loc[0, "cell_id"]

    with pytest.raises(
        ValueError,
        match="duplicate cell IDs",
    ):
        validate_inputs(
            baseline_adata,
            invalid,
        )


def test_identify_feature_groups(
    baseline_feature_table: pd.DataFrame,
) -> None:
    indexed = baseline_feature_table.set_index("cell_id")

    groups = identify_feature_groups(indexed)

    assert set(groups) == {
        "rap_only",
        "pathways_only",
        "regulators_only",
        "rap_plus_pathways",
        "rap_plus_regulators",
        "pathways_plus_regulators",
        "full_model",
    }

    assert groups["rap_only"] == ["rap_score"]

    assert len(groups["pathways_only"]) == 2
    assert len(groups["regulators_only"]) == 2
    assert len(groups["full_model"]) == 5


def test_evaluate_majority_baseline(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    indexed = baseline_feature_table.set_index("cell_id")

    labels = indexed["transition_label"]

    cell_lines = pd.Series(
        baseline_adata.obs["cell_line"].to_numpy(),
        index=baseline_adata.obs_names.astype(str),
    )

    predictions, metrics = evaluate_majority_baseline(
        labels,
        cell_lines,
    )

    assert len(metrics) == 2
    assert len(predictions) == baseline_adata.n_obs
    assert set(metrics["model"]) == {"majority_class"}
    assert (
        predictions["transition_probability"]
        .between(
            0,
            1,
        )
        .all()
    )


def test_evaluate_logistic_baseline(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    indexed = baseline_feature_table.set_index("cell_id")

    features = indexed[["rap_score"]]
    labels = indexed["transition_label"]

    cell_lines = pd.Series(
        baseline_adata.obs["cell_line"].to_numpy(),
        index=baseline_adata.obs_names.astype(str),
    )

    predictions, metrics = evaluate_logistic_baseline(
        features,
        labels,
        cell_lines,
        model_name="rap_only",
    )

    assert len(metrics) == 2
    assert len(predictions) == baseline_adata.n_obs
    assert (metrics["roc_auc"] > 0.90).all()


def test_fit_baseline_coefficients(
    baseline_feature_table: pd.DataFrame,
) -> None:
    indexed = baseline_feature_table.set_index("cell_id")

    features = indexed[
        [
            "rap_score",
            "pathway__e2f_targets",
        ]
    ]

    coefficients = fit_baseline_coefficients(
        features,
        indexed["transition_label"],
        model_name="test_model",
    )

    assert len(coefficients) == 2
    assert set(coefficients["model"]) == {"test_model"}
    assert coefficients["absolute_coefficient"].is_monotonic_decreasing


def test_summarize_baseline_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_a",
                "model_b",
                "model_b",
            ],
            "n_features": [1, 1, 2, 2],
            "accuracy": [0.8, 0.9, 0.6, 0.7],
            "balanced_accuracy": [0.8, 0.9, 0.6, 0.7],
            "precision": [0.8, 0.9, 0.6, 0.7],
            "recall": [0.8, 0.9, 0.6, 0.7],
            "f1": [0.8, 0.9, 0.6, 0.7],
            "roc_auc": [0.9, 0.95, 0.7, 0.75],
            "average_precision": [0.9, 0.95, 0.7, 0.75],
        }
    )

    summary = summarize_baseline_metrics(metrics)

    assert list(summary["model"]) == [
        "model_a",
        "model_b",
    ]

    assert summary.loc[
        summary["model"].eq("model_a"),
        "roc_auc_mean",
    ].iloc[0] == pytest.approx(0.925)


def test_run_baseline_analysis(
    baseline_adata: ad.AnnData,
    baseline_feature_table: pd.DataFrame,
) -> None:
    (
        metrics,
        predictions,
        coefficients,
        summary,
    ) = run_baseline_analysis(
        baseline_adata,
        baseline_feature_table,
    )

    expected_models = {
        "majority_class",
        "rap_only",
        "pathways_only",
        "regulators_only",
        "rap_plus_pathways",
        "rap_plus_regulators",
        "pathways_plus_regulators",
        "full_model",
    }

    assert set(metrics["model"]) == expected_models
    assert set(predictions["model"]) == expected_models
    assert set(summary["model"]) == expected_models

    assert "majority_class" not in set(coefficients["model"])

    assert len(metrics) == len(expected_models) * 2

    full_auc = summary.loc[
        summary["model"].eq("full_model"),
        "roc_auc_mean",
    ].iloc[0]

    assert full_auc > 0.90


def test_save_results(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame({"value": [1]})

    save_results(
        table,
        table,
        table,
        table,
        tmp_path,
    )

    expected_files = [
        "baseline_model_metrics.csv",
        "baseline_model_predictions.csv",
        "baseline_model_coefficients.csv",
        "baseline_model_summary.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
