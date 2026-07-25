from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.trajectory import (
    add_features_to_adata,
    assign_trajectory_bins,
    calculate_feature_trajectory_correlations,
    calculate_gene_trajectory_correlations,
    calculate_stage_markers,
    classify_trajectory_stage,
    create_condition_distribution_summary,
    create_conserved_feature_summary,
    create_conserved_gene_summary,
    create_escape_pseudotime,
    extract_transition_features,
    get_expression_matrix,
    load_transition_feature_table,
    run_trajectory_analysis,
    sanitize_feature_name,
    save_results,
    select_variable_genes,
    summarize_trajectory_bins,
    validate_trajectory_adata,
)


@pytest.fixture
def trajectory_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)

    genes = [
        "E2F1",
        "MYC",
        "MKI67",
        "CCNB1",
        "CDK1",
        "TP53",
    ]

    observation_rows = []
    expression_rows = []

    for cell_line in [
        "MCF7",
        "T47D",
    ]:
        for condition in [
            "TIS",
            "CTR",
            "REPOP",
        ]:
            for index in range(20):
                base_probability = {
                    "TIS": 0.10,
                    "CTR": 0.55,
                    "REPOP": 0.90,
                }[condition]

                probability = np.clip(
                    base_probability
                    + rng.normal(
                        0.0,
                        0.04,
                    ),
                    0.0,
                    1.0,
                )

                observation_rows.append(
                    {
                        "cell_line": cell_line,
                        "condition": condition,
                        "transition_probability": (probability),
                        "repopulation_associated_potential": (probability * 2 - 1),
                        "pathway__e2f_targets": (
                            probability
                            + rng.normal(
                                0.0,
                                0.05,
                            )
                        ),
                        "regulator__E2F1": (
                            probability
                            + rng.normal(
                                0.0,
                                0.05,
                            )
                        ),
                    }
                )

                expression_rows.append(
                    [
                        probability + rng.normal(0, 0.05),
                        probability + rng.normal(0, 0.05),
                        probability + rng.normal(0, 0.05),
                        probability + rng.normal(0, 0.05),
                        probability + rng.normal(0, 0.05),
                        1.0 - probability + rng.normal(0, 0.05),
                    ]
                )

    obs = pd.DataFrame(
        observation_rows,
        index=[f"cell_{index}" for index in range(len(observation_rows))],
    )

    var = pd.DataFrame(index=genes)

    return ad.AnnData(
        X=np.asarray(expression_rows),
        obs=obs,
        var=var,
    )


def test_validate_trajectory_adata(
    trajectory_adata: ad.AnnData,
) -> None:
    validate_trajectory_adata(trajectory_adata)


def test_validate_trajectory_adata_rejects_missing_column(
    trajectory_adata: ad.AnnData,
) -> None:
    invalid = trajectory_adata.copy()

    invalid.obs = invalid.obs.drop(columns=["transition_probability"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_trajectory_adata(invalid)


def test_create_escape_pseudotime(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    assert "escape_pseudotime" in scored.obs.columns

    assert np.isclose(
        scored.obs["escape_pseudotime"].min(),
        0.0,
    )

    assert np.isclose(
        scored.obs["escape_pseudotime"].max(),
        1.0,
    )


def test_assign_trajectory_bins(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    binned = assign_trajectory_bins(
        scored,
        number_of_bins=6,
    )

    assert binned.obs["trajectory_bin"].notna().all()

    assert set(binned.obs["trajectory_stage"]) == {
        "early",
        "intermediate",
        "late",
    }


def test_classify_trajectory_stage() -> None:
    assert (
        classify_trajectory_stage(
            1,
            9,
        )
        == "early"
    )

    assert (
        classify_trajectory_stage(
            5,
            9,
        )
        == "intermediate"
    )

    assert (
        classify_trajectory_stage(
            9,
            9,
        )
        == "late"
    )


def test_sanitize_feature_name() -> None:
    assert sanitize_feature_name("G2-M Checkpoint") == "g2_m_checkpoint"


def test_get_expression_matrix(
    trajectory_adata: ad.AnnData,
) -> None:
    matrix, genes = get_expression_matrix(
        trajectory_adata,
        [
            "E2F1",
            "MYC",
            "NOT_A_GENE",
        ],
    )

    assert matrix.shape == (
        trajectory_adata.n_obs,
        2,
    )

    assert genes == [
        "E2F1",
        "MYC",
    ]


def test_extract_transition_features(
    trajectory_adata: ad.AnnData,
) -> None:
    features = extract_transition_features(trajectory_adata)

    assert set(features.columns) == {
        "pathway__e2f_targets",
        "regulator__E2F1",
        "repopulation_associated_potential",
    }


def test_add_features_to_adata(
    trajectory_adata: ad.AnnData,
) -> None:
    features = pd.DataFrame(
        {
            "pathway__myc_targets": np.linspace(
                0,
                1,
                trajectory_adata.n_obs,
            )
        },
        index=trajectory_adata.obs_names,
    )

    enriched = add_features_to_adata(
        trajectory_adata,
        features,
    )

    assert "pathway__myc_targets" in enriched.obs.columns


def test_load_transition_feature_table(
    tmp_path: Path,
    trajectory_adata: ad.AnnData,
) -> None:
    table = pd.DataFrame(
        {
            "cell_id": (trajectory_adata.obs_names),
            "rap_score": np.linspace(
                -1,
                1,
                trajectory_adata.n_obs,
            ),
            "pathway__e2f_targets": np.linspace(
                0,
                1,
                trajectory_adata.n_obs,
            ),
            "state": trajectory_adata.obs["condition"].to_numpy(),
        }
    )

    path = tmp_path / "transition_features.csv"

    table.to_csv(
        path,
        index=False,
    )

    loaded = load_transition_feature_table(
        path,
        trajectory_adata,
    )

    assert set(loaded.columns) == {
        "rap_score",
        "pathway__e2f_targets",
    }

    assert len(loaded) == (trajectory_adata.n_obs)


def test_summarize_trajectory_bins(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    scored = assign_trajectory_bins(
        scored,
        number_of_bins=5,
    )

    summary = summarize_trajectory_bins(
        scored,
        feature_columns=[
            "pathway__e2f_targets",
        ],
        minimum_cells_per_bin=1,
    )

    assert not summary.empty

    assert "mean__pathway__e2f_targets" in summary.columns

    assert (summary["cells"] > 0).all()


def test_calculate_feature_trajectory_correlations(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    results = calculate_feature_trajectory_correlations(
        scored,
        feature_columns=[
            "pathway__e2f_targets",
            "regulator__E2F1",
        ],
    )

    assert len(results) == 4

    assert (results["absolute_correlation"] > 0.5).all()


def test_create_conserved_feature_summary() -> None:
    correlations = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "T47D",
                "MCF7",
                "T47D",
            ],
            "feature": [
                "feature_a",
                "feature_a",
                "feature_b",
                "feature_b",
            ],
            "spearman_correlation": [
                0.8,
                0.7,
                0.8,
                -0.7,
            ],
            "absolute_correlation": [
                0.8,
                0.7,
                0.8,
                0.7,
            ],
            "p_value": [
                0.001,
                0.002,
                0.001,
                0.002,
            ],
        }
    )

    conserved = create_conserved_feature_summary(
        correlations,
        minimum_absolute_correlation=0.5,
    )

    assert list(conserved["feature"]) == ["feature_a"]


def test_select_variable_genes(
    trajectory_adata: ad.AnnData,
) -> None:
    genes = select_variable_genes(
        trajectory_adata,
        maximum_genes=3,
    )

    assert len(genes) == 3

    assert set(genes).issubset(set(trajectory_adata.var_names))


def test_calculate_gene_trajectory_correlations(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    results = calculate_gene_trajectory_correlations(
        scored,
        genes=[
            "E2F1",
            "TP53",
        ],
    )

    assert len(results) == 4

    e2f_results = results.loc[results["gene"] == "E2F1"]

    tp53_results = results.loc[results["gene"] == "TP53"]

    assert (e2f_results["spearman_correlation"] > 0).all()

    assert (tp53_results["spearman_correlation"] < 0).all()


def test_create_conserved_gene_summary() -> None:
    correlations = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "T47D",
            ],
            "gene": [
                "E2F1",
                "E2F1",
            ],
            "cells": [
                100,
                100,
            ],
            "spearman_correlation": [
                0.8,
                0.7,
            ],
            "absolute_correlation": [
                0.8,
                0.7,
            ],
            "p_value": [
                0.001,
                0.002,
            ],
            "direction": [
                "increasing",
                "increasing",
            ],
        }
    )

    conserved = create_conserved_gene_summary(
        correlations,
        minimum_absolute_correlation=0.5,
    )

    assert list(conserved["gene"]) == ["E2F1"]


def test_calculate_stage_markers(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    scored = assign_trajectory_bins(
        scored,
        number_of_bins=6,
    )

    markers = calculate_stage_markers(
        scored,
        feature_columns=[
            "pathway__e2f_targets",
        ],
    )

    assert len(markers) == 2

    assert (markers["peak_stage"] == "late").all()


def test_create_condition_distribution_summary(
    trajectory_adata: ad.AnnData,
) -> None:
    scored = create_escape_pseudotime(trajectory_adata)

    summary = create_condition_distribution_summary(scored)

    assert len(summary) == 6

    assert set(summary["condition"]) == {
        "TIS",
        "CTR",
        "REPOP",
    }


def test_run_trajectory_analysis(
    trajectory_adata: ad.AnnData,
) -> None:
    (
        scored,
        trajectory_bins,
        feature_correlations,
        conserved_features,
        gene_correlations,
        conserved_genes,
        stage_markers,
        condition_summary,
    ) = run_trajectory_analysis(
        trajectory_adata,
        number_of_bins=6,
        minimum_cells_per_bin=1,
        maximum_genes=4,
        minimum_absolute_correlation=0.2,
    )

    assert "escape_pseudotime" in scored.obs.columns

    assert "trajectory_bin" in scored.obs.columns

    assert not trajectory_bins.empty

    assert not feature_correlations.empty

    assert not conserved_features.empty

    assert not gene_correlations.empty

    assert not conserved_genes.empty

    assert not stage_markers.empty

    assert len(condition_summary) == 6


def test_save_results(
    tmp_path: Path,
    trajectory_adata: ad.AnnData,
) -> None:
    table = pd.DataFrame(
        {
            "value": [
                1,
            ]
        }
    )

    output_adata = tmp_path / "trajectory_scored.h5ad"

    save_results(
        trajectory_adata,
        table,
        table,
        table,
        table,
        table,
        table,
        table,
        output_adata,
        tmp_path,
    )

    assert output_adata.exists()

    expected_files = [
        "trajectory_bin_summary.csv",
        "trajectory_feature_correlations.csv",
        "trajectory_conserved_features.csv",
        "trajectory_gene_correlations.csv",
        "trajectory_conserved_genes.csv",
        "trajectory_stage_markers.csv",
        "trajectory_condition_summary.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
