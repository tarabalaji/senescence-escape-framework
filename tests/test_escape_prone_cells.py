from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.escape_prone_cells import (
    calculate_gene_markers,
    classify_escape_prone_cells,
    compare_escape_features,
    create_conserved_feature_summary,
    create_conserved_marker_summary,
    extract_obs_features,
    run_escape_prone_analysis,
    save_results,
    summarize_escape_prone_cells,
    validate_adata,
)


@pytest.fixture
def escape_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)

    obs_rows = []
    expression_rows = []

    genes = [
        "MKI67",
        "TOP2A",
        "CLU",
        "IFI6",
    ]

    for cell_line in ["MCF7", "T47D"]:
        for index in range(100):
            probability = index / 100

            obs_rows.append(
                {
                    "cell_line": cell_line,
                    "condition": "TIS",
                    "transition_probability": probability,
                    "escape_pseudotime": probability,
                    "repopulation_associated_potential": (probability * 2 - 1),
                    "pathway__e2f_targets": (probability + rng.normal(0, 0.02)),
                    "regulator__E2F1": (probability + rng.normal(0, 0.02)),
                }
            )

            expression_rows.append(
                [
                    probability + rng.normal(0, 0.02),
                    probability + rng.normal(0, 0.02),
                    1 - probability + rng.normal(0, 0.02),
                    1 - probability + rng.normal(0, 0.02),
                ]
            )

        for index in range(20):
            obs_rows.append(
                {
                    "cell_line": cell_line,
                    "condition": "REPOP",
                    "transition_probability": 0.95,
                    "escape_pseudotime": 0.95,
                    "repopulation_associated_potential": 0.90,
                    "pathway__e2f_targets": 0.95,
                    "regulator__E2F1": 0.95,
                }
            )

            expression_rows.append([1.0, 1.0, 0.0, 0.0])

    obs = pd.DataFrame(
        obs_rows,
        index=[f"cell_{index}" for index in range(len(obs_rows))],
    )

    return ad.AnnData(
        X=np.asarray(expression_rows),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )


def test_validate_adata(
    escape_adata: ad.AnnData,
) -> None:
    validate_adata(escape_adata)


def test_validate_adata_missing_column(
    escape_adata: ad.AnnData,
) -> None:
    invalid = escape_adata.copy()

    invalid.obs = invalid.obs.drop(columns=["transition_probability"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_adata(invalid)


def test_classify_escape_prone_cells(
    escape_adata: ad.AnnData,
) -> None:
    scored, thresholds = classify_escape_prone_cells(
        escape_adata,
        escape_quantile=0.90,
    )

    counts = scored.obs["escape_prone_status"].astype(str).value_counts()

    assert counts["escape_prone_tis"] == 20
    assert counts["stable_tis"] == 180
    assert len(thresholds) == 2


def test_summarize_escape_prone_cells(
    escape_adata: ad.AnnData,
) -> None:
    scored, _ = classify_escape_prone_cells(escape_adata)

    summary = summarize_escape_prone_cells(scored)

    assert len(summary) == 4

    escape_mean = summary.loc[
        summary["status"] == "escape_prone_tis",
        "mean_transition_probability",
    ].mean()

    stable_mean = summary.loc[
        summary["status"] == "stable_tis",
        "mean_transition_probability",
    ].mean()

    assert escape_mean > stable_mean


def test_extract_obs_features(
    escape_adata: ad.AnnData,
) -> None:
    features = extract_obs_features(escape_adata)

    assert set(features) == {
        "pathway__e2f_targets",
        "regulator__E2F1",
        "repopulation_associated_potential",
    }


def test_compare_escape_features(
    escape_adata: ad.AnnData,
) -> None:
    scored, _ = classify_escape_prone_cells(
        escape_adata,
        escape_quantile=0.80,
    )

    results = compare_escape_features(
        scored,
        feature_columns=[
            "pathway__e2f_targets",
            "regulator__E2F1",
        ],
        minimum_cells=10,
    )

    assert len(results) == 4

    assert (results["mean_difference"] > 0).all()

    assert (results["adjusted_p_value"] < 0.05).all()


def test_calculate_gene_markers(
    escape_adata: ad.AnnData,
) -> None:
    scored, _ = classify_escape_prone_cells(
        escape_adata,
        escape_quantile=0.80,
    )

    markers = calculate_gene_markers(
        scored,
        minimum_cells=10,
        log_fold_change_threshold=0.10,
        adjusted_p_value_threshold=0.05,
    )

    assert not markers.empty

    increasing = set(
        markers.loc[
            markers["direction"] == "higher_in_escape_prone",
            "gene",
        ]
    )

    decreasing = set(
        markers.loc[
            markers["direction"] == "lower_in_escape_prone",
            "gene",
        ]
    )

    assert "MKI67" in increasing
    assert "TOP2A" in increasing
    assert "CLU" in decreasing
    assert "IFI6" in decreasing


def test_create_conserved_marker_summary() -> None:
    markers = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "T47D",
                "MCF7",
                "T47D",
            ],
            "gene": [
                "MKI67",
                "MKI67",
                "GENE2",
                "GENE2",
            ],
            "log_fold_change": [
                1.0,
                0.8,
                1.0,
                -0.8,
            ],
            "absolute_log_fold_change": [
                1.0,
                0.8,
                1.0,
                0.8,
            ],
            "adjusted_p_value": [
                0.001,
                0.002,
                0.001,
                0.002,
            ],
        }
    )

    conserved = create_conserved_marker_summary(markers)

    assert list(conserved["gene"]) == ["MKI67"]


def test_create_conserved_feature_summary() -> None:
    results = pd.DataFrame(
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
            "mean_difference": [
                1.0,
                0.8,
                1.0,
                -0.8,
            ],
            "absolute_mean_difference": [
                1.0,
                0.8,
                1.0,
                0.8,
            ],
            "adjusted_p_value": [
                0.001,
                0.002,
                0.001,
                0.002,
            ],
        }
    )

    conserved = create_conserved_feature_summary(results)

    assert list(conserved["feature"]) == ["feature_a"]


def test_run_escape_prone_analysis(
    escape_adata: ad.AnnData,
) -> None:
    results = run_escape_prone_analysis(
        escape_adata,
        escape_quantile=0.80,
        minimum_cells=10,
        log_fold_change_threshold=0.10,
    )

    (
        scored,
        thresholds,
        summary,
        feature_results,
        conserved_features,
        gene_markers,
        conserved_markers,
    ) = results

    assert "escape_prone_status" in scored.obs.columns

    assert len(thresholds) == 2
    assert not summary.empty
    assert not feature_results.empty
    assert not conserved_features.empty
    assert not gene_markers.empty
    assert not conserved_markers.empty


def test_save_results(
    tmp_path: Path,
    escape_adata: ad.AnnData,
) -> None:
    table = pd.DataFrame({"value": [1]})

    output_adata = tmp_path / "escape_prone_scored.h5ad"

    save_results(
        escape_adata,
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
        "escape_prone_thresholds.csv",
        "escape_prone_summary.csv",
        "escape_prone_feature_comparisons.csv",
        "escape_prone_conserved_features.csv",
        "escape_prone_gene_markers.csv",
        "escape_prone_conserved_markers.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
