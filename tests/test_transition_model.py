from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.transition_model import (
    build_transition_features,
    calculate_pathway_activity,
    calculate_regulator_features,
    create_prediction_summary,
    create_transition_labels,
    evaluate_predictions,
    fit_final_model,
    parse_gene_list,
    run_leave_one_cell_line_out,
    run_transition_model,
    save_results,
    select_pathway_gene_sets,
    select_regulators,
    standardize_gene_expression,
    validate_adata,
)


@pytest.fixture
def transition_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)

    genes = [
        "E2F1",
        "MYC",
        "TP53",
        "JUN",
        "FOS",
        "MKI67",
        "CCNB1",
        "CDK1",
    ]

    observations = []

    expression_rows = []

    for cell_line in [
        "MCF7",
        "T47D",
    ]:
        for state in [
            "CTR",
            "TIS",
            "REPOP",
        ]:
            for cell_index in range(10):
                observations.append(
                    {
                        "cell_line": cell_line,
                        "state": state,
                        "rap_score": {
                            "CTR": 0.2,
                            "TIS": -1.0,
                            "REPOP": 1.0,
                        }[state]
                        + rng.normal(0, 0.1),
                    }
                )

                base = rng.normal(
                    1.0,
                    0.1,
                    len(genes),
                )

                if state == "TIS":
                    base[
                        [
                            0,
                            1,
                            5,
                            6,
                            7,
                        ]
                    ] -= 0.8

                if state == "REPOP":
                    base[
                        [
                            0,
                            1,
                            5,
                            6,
                            7,
                        ]
                    ] += 1.0

                expression_rows.append(base)

    obs = pd.DataFrame(
        observations,
        index=[f"cell_{index}" for index in range(len(observations))],
    )

    var = pd.DataFrame(index=genes)

    return ad.AnnData(
        X=np.asarray(expression_rows),
        obs=obs,
        var=var,
    )


@pytest.fixture
def pathway_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "TIS_vs_CTR",
            ],
            "pathway": [
                "E2F Targets",
                "G2-M Checkpoint",
                "Weak Pathway",
                "Senescence",
            ],
            "normalized_enrichment_score": [
                1.8,
                1.6,
                0.4,
                -1.5,
            ],
            "false_discovery_rate": [
                0.001,
                0.01,
                0.5,
                0.01,
            ],
            "leading_edge_genes": [
                "E2F1;MKI67;CDK1",
                "CCNB1;CDK1;MKI67",
                "TP53;JUN",
                "TP53;FOS",
            ],
        }
    )


@pytest.fixture
def regulator_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regulator": [
                "E2F1",
                "MYC",
                "TP53",
                "JUN",
            ],
            "priority_score": [
                0.9,
                0.8,
                0.4,
                0.2,
            ],
        }
    )


def test_validate_adata_accepts_valid_data(
    transition_adata: ad.AnnData,
) -> None:
    validate_adata(transition_adata)


def test_validate_adata_rejects_missing_columns(
    transition_adata: ad.AnnData,
) -> None:
    invalid = transition_adata.copy()

    invalid.obs = invalid.obs.drop(columns=["rap_score"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_adata(invalid)


def test_parse_gene_list() -> None:
    genes = parse_gene_list("E2F1; MYC|TP53, JUN")

    assert genes == [
        "E2F1",
        "MYC",
        "TP53",
        "JUN",
    ]


def test_select_pathway_gene_sets(
    pathway_results: pd.DataFrame,
) -> None:
    gene_sets = select_pathway_gene_sets(
        pathway_results,
        maximum_pathways=2,
    )

    assert set(gene_sets) == {
        "E2F Targets",
        "G2-M Checkpoint",
    }

    assert gene_sets["E2F Targets"] == [
        "E2F1",
        "MKI67",
        "CDK1",
    ]


def test_select_regulators(
    regulator_results: pd.DataFrame,
) -> None:
    regulators = select_regulators(
        regulator_results,
        maximum_regulators=2,
    )

    assert regulators == [
        "E2F1",
        "MYC",
    ]


def test_standardize_gene_expression() -> None:
    matrix = np.array(
        [
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ]
    )

    standardized = standardize_gene_expression(matrix)

    assert standardized.shape == matrix.shape

    assert np.isclose(
        standardized[:, 0].mean(),
        0.0,
    )

    assert np.allclose(
        standardized[:, 1],
        0.0,
    )


def test_calculate_pathway_activity(
    transition_adata: ad.AnnData,
) -> None:
    gene_sets = {
        "E2F Targets": [
            "E2F1",
            "MKI67",
            "CDK1",
        ],
        "Missing Pathway": [
            "NOT_A_GENE",
        ],
    }

    activity = calculate_pathway_activity(
        transition_adata,
        gene_sets,
        minimum_genes=2,
    )

    assert list(activity.columns) == ["pathway__e2f_targets"]

    assert len(activity) == (transition_adata.n_obs)


def test_calculate_regulator_features(
    transition_adata: ad.AnnData,
) -> None:
    features = calculate_regulator_features(
        transition_adata,
        [
            "E2F1",
            "MYC",
            "NOT_A_GENE",
        ],
    )

    assert set(features.columns) == {
        "regulator__E2F1",
        "regulator__MYC",
    }


def test_build_transition_features(
    transition_adata: ad.AnnData,
) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={
            "E2F Targets": [
                "E2F1",
                "MKI67",
                "CDK1",
            ],
        },
        regulators=[
            "E2F1",
            "MYC",
        ],
    )

    assert set(features.columns) == {
        "rap_score",
        "pathway__e2f_targets",
        "regulator__E2F1",
        "regulator__MYC",
    }

    assert not features.isna().any().any()


def test_create_transition_labels(
    transition_adata: ad.AnnData,
) -> None:
    labels = create_transition_labels(transition_adata)

    assert labels.loc[transition_adata.obs["state"] == "TIS"].eq(0).all()

    assert labels.loc[transition_adata.obs["state"] == "REPOP"].eq(1).all()

    assert labels.loc[transition_adata.obs["state"] == "CTR"].isna().all()


def test_evaluate_predictions() -> None:
    true_labels = np.array([0, 0, 1, 1])

    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_predictions(
        true_labels,
        probabilities,
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_run_leave_one_cell_line_out(
    transition_adata: ad.AnnData,
) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={
            "E2F Targets": [
                "E2F1",
                "MKI67",
                "CDK1",
            ],
        },
        regulators=[
            "E2F1",
            "MYC",
        ],
    )

    labels = create_transition_labels(transition_adata)

    probabilities, metrics = run_leave_one_cell_line_out(
        features,
        labels,
        transition_adata.obs["cell_line"],
    )

    training_mask = labels.notna()

    assert probabilities.loc[training_mask].notna().all()

    assert probabilities.loc[~training_mask].isna().all()

    assert set(metrics["held_out_cell_line"]) == {
        "MCF7",
        "T47D",
    }

    assert (metrics["roc_auc"] >= 0.5).all()


def test_fit_final_model(
    transition_adata: ad.AnnData,
) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={
            "E2F Targets": [
                "E2F1",
                "MKI67",
                "CDK1",
            ],
        },
        regulators=[
            "E2F1",
            "MYC",
        ],
    )

    labels = create_transition_labels(transition_adata)

    model, probabilities, coefficients = fit_final_model(
        features,
        labels,
    )

    assert model is not None

    assert len(probabilities) == (transition_adata.n_obs)

    assert len(coefficients) == (features.shape[1])

    assert set(coefficients.columns) == {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }


def test_run_transition_model(
    transition_adata: ad.AnnData,
    pathway_results: pd.DataFrame,
    regulator_results: pd.DataFrame,
) -> None:
    (
        scored,
        metrics,
        coefficients,
        summary,
        feature_table,
    ) = run_transition_model(
        transition_adata,
        pathway_results,
        regulator_results,
        maximum_pathways=2,
        maximum_regulators=3,
    )

    assert "transition_probability" in scored.obs.columns

    assert "transition_probability_oof" in scored.obs.columns

    assert len(metrics) == 2

    assert not coefficients.empty

    assert set(summary["state"]) == {
        "CTR",
        "TIS",
        "REPOP",
    }

    assert len(feature_table) == (transition_adata.n_obs)


def test_create_prediction_summary(
    transition_adata: ad.AnnData,
) -> None:
    scored = transition_adata.copy()

    scored.obs["transition_probability"] = np.linspace(
        0,
        1,
        scored.n_obs,
    )

    summary = create_prediction_summary(scored)

    assert len(summary) == 6

    assert set(summary.columns) == {
        "cell_line",
        "state",
        "cells",
        "mean_transition_probability",
        "median_transition_probability",
        "standard_deviation",
    }


def test_save_results(
    tmp_path: Path,
    transition_adata: ad.AnnData,
) -> None:
    metrics = pd.DataFrame(
        {
            "held_out_cell_line": [
                "MCF7",
            ],
            "roc_auc": [
                0.9,
            ],
        }
    )

    coefficients = pd.DataFrame(
        {
            "feature": [
                "rap_score",
            ],
            "coefficient": [
                1.0,
            ],
            "absolute_coefficient": [
                1.0,
            ],
        }
    )

    summary = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
            ],
            "state": [
                "TIS",
            ],
        }
    )

    features = pd.DataFrame(
        {
            "cell_id": [
                "cell_1",
            ],
            "rap_score": [
                0.2,
            ],
        }
    )

    output_adata = tmp_path / "transition_scored.h5ad"

    save_results(
        transition_adata,
        metrics,
        coefficients,
        summary,
        features,
        output_adata,
        tmp_path,
    )

    assert output_adata.exists()

    assert (tmp_path / "transition_model_metrics.csv").exists()

    assert (tmp_path / "transition_model_coefficients.csv").exists()

    assert (tmp_path / "transition_probability_summary.csv").exists()

    assert (tmp_path / "transition_model_features.csv").exists()
