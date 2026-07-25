from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.target_prioritization import (
    calculate_priority_scores,
    create_ablation_summary,
    create_candidate_summary,
    create_evidence_matrix,
    merge_evidence_tables,
    normalize_zero_to_one,
    prepare_differential_expression_evidence,
    prepare_escape_marker_evidence,
    prepare_regulatory_evidence,
    prepare_signature_evidence,
    prepare_trajectory_evidence,
    prepare_transition_model_evidence,
    run_ablation_analysis,
    run_target_prioritization,
    save_results,
    validate_weights,
)


@pytest.fixture
def markers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": [
                "TOP2A",
                "MKI67",
                "E2F1",
                "CLU",
            ],
            "mean_log_fold_change": [
                0.70,
                0.53,
                0.40,
                -0.55,
            ],
        }
    )


@pytest.fixture
def trajectory_genes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": [
                "TOP2A",
                "MKI67",
                "E2F1",
                "CLU",
            ],
            "mean_spearman_correlation": [
                0.65,
                0.60,
                0.55,
                -0.58,
            ],
        }
    )


@pytest.fixture
def differential_expression() -> pd.DataFrame:
    rows = []

    effects = {
        "TOP2A": 1.8,
        "MKI67": 1.5,
        "E2F1": 1.2,
        "CLU": -1.0,
        "INCONSISTENT": 1.0,
    }

    for cell_line in [
        "MCF7",
        "T47D",
    ]:
        for gene, effect in effects.items():
            if gene == "INCONSISTENT" and cell_line == "T47D":
                effect = -effect

            rows.append(
                {
                    "cell_line": cell_line,
                    "comparison": "REPOP_vs_TIS",
                    "gene": gene,
                    "log2_fold_change": effect,
                    "adjusted_p_value": 0.001,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def regulators() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transcription_factor": [
                "E2F1",
                "MYC",
                "JUN",
            ],
            "cell_lines_supported": [
                2,
                2,
                1,
            ],
            "mean_regulatory_score": [
                0.80,
                0.50,
                0.20,
            ],
        }
    )


@pytest.fixture
def transition_coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "rap_score",
                "regulator__E2F1",
                "regulator__MYC",
                "pathway__e2f_targets",
            ],
            "coefficient": [
                8.0,
                0.8,
                -0.3,
                1.2,
            ],
            "absolute_coefficient": [
                8.0,
                0.8,
                0.3,
                1.2,
            ],
        }
    )


@pytest.fixture
def signature() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": [
                "TOP2A",
                "MKI67",
                "E2F1",
                "CLU",
            ],
            "direction": [
                "up",
                "up",
                "up",
                "down",
            ],
        }
    )


def test_validate_weights() -> None:
    validate_weights(
        {
            "source_a": 0.5,
            "source_b": 0.5,
        }
    )


def test_validate_weights_rejects_negative() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        validate_weights(
            {
                "source_a": -1.0,
            }
        )


def test_normalize_zero_to_one() -> None:
    values = pd.Series(
        [
            2.0,
            4.0,
            6.0,
        ]
    )

    normalized = normalize_zero_to_one(values)

    np.testing.assert_allclose(
        normalized,
        [
            0.0,
            0.5,
            1.0,
        ],
    )


def test_prepare_escape_marker_evidence(
    markers: pd.DataFrame,
) -> None:
    prepared = prepare_escape_marker_evidence(markers)

    assert set(prepared.columns) == {
        "gene",
        "escape_marker_effect",
        "escape_marker_score",
        "escape_marker_direction",
    }

    assert prepared["escape_marker_score"].between(0, 1).all()


def test_prepare_trajectory_evidence(
    trajectory_genes: pd.DataFrame,
) -> None:
    prepared = prepare_trajectory_evidence(trajectory_genes)

    assert set(prepared["gene"]) == {
        "TOP2A",
        "MKI67",
        "E2F1",
        "CLU",
    }

    assert prepared["trajectory_score"].between(0, 1).all()


def test_prepare_differential_expression_evidence(
    differential_expression: pd.DataFrame,
) -> None:
    prepared = prepare_differential_expression_evidence(differential_expression)

    assert "INCONSISTENT" not in set(prepared["gene"])

    assert set(prepared["de_cell_lines_supported"]) == {
        2,
    }

    assert prepared["differential_expression_score"].between(0, 1).all()


def test_prepare_regulatory_evidence(
    regulators: pd.DataFrame,
) -> None:
    prepared = prepare_regulatory_evidence(regulators)

    e2f1 = prepared.loc[prepared["gene"] == "E2F1"].iloc[0]

    assert e2f1["regulatory_score"] == 1.0


def test_prepare_transition_model_evidence(
    transition_coefficients: pd.DataFrame,
) -> None:
    prepared = prepare_transition_model_evidence(transition_coefficients)

    assert set(prepared["gene"]) == {
        "E2F1",
        "MYC",
    }

    assert set(prepared["transition_model_direction"]) == {
        "supports_repop_prediction",
        "opposes_repop_prediction",
    }


def test_prepare_signature_evidence(
    signature: pd.DataFrame,
) -> None:
    prepared = prepare_signature_evidence(signature)

    assert (prepared["signature_score"] == 1.0).all()


def test_merge_evidence_tables() -> None:
    first = pd.DataFrame(
        {
            "gene": [
                "A",
                "B",
            ],
            "score_a": [
                1.0,
                0.5,
            ],
        }
    )

    second = pd.DataFrame(
        {
            "gene": [
                "B",
                "C",
            ],
            "score_b": [
                0.8,
                0.4,
            ],
        }
    )

    merged = merge_evidence_tables(
        [
            first,
            second,
        ]
    )

    assert set(merged["gene"]) == {
        "A",
        "B",
        "C",
    }


def test_calculate_priority_scores() -> None:
    evidence = pd.DataFrame(
        {
            "gene": [
                "A",
                "B",
                "C",
            ],
            "source_a_score": [
                1.0,
                0.5,
                np.nan,
            ],
            "source_b_score": [
                1.0,
                0.2,
                1.0,
            ],
        }
    )

    rankings = calculate_priority_scores(
        evidence,
        weights={
            "source_a_score": 0.5,
            "source_b_score": 0.5,
        },
        minimum_evidence_sources=1,
    )

    assert rankings.iloc[0]["gene"] == "A"

    assert rankings.iloc[0]["priority_score"] == 1.0


def test_create_evidence_matrix() -> None:
    rankings = pd.DataFrame(
        {
            "rank": [
                1,
            ],
            "gene": [
                "E2F1",
            ],
            "priority_score": [
                0.9,
            ],
            "evidence_source_count": [
                2,
            ],
            "source_a_score": [
                1.0,
            ],
            "source_b_score": [
                np.nan,
            ],
        }
    )

    matrix = create_evidence_matrix(
        rankings,
        [
            "source_a_score",
            "source_b_score",
        ],
    )

    assert bool(matrix["has_source_a"].iloc[0])

    assert not bool(matrix["has_source_b"].iloc[0])


def test_run_ablation_analysis() -> None:
    evidence = pd.DataFrame(
        {
            "gene": [
                "A",
                "B",
                "C",
            ],
            "source_a_score": [
                1.0,
                0.6,
                0.2,
            ],
            "source_b_score": [
                0.9,
                0.5,
                0.4,
            ],
        }
    )

    stability, ablated = run_ablation_analysis(
        evidence,
        weights={
            "source_a_score": 0.5,
            "source_b_score": 0.5,
        },
        minimum_evidence_sources=1,
    )

    assert not stability.empty

    assert set(ablated) == {
        "source_a_score",
        "source_b_score",
    }

    assert stability["rank_stability_score"].between(0, 1).all()


def test_create_ablation_summary() -> None:
    full = pd.DataFrame(
        {
            "gene": [
                "A",
                "B",
                "C",
            ],
            "rank": [
                1,
                2,
                3,
            ],
        }
    )

    ablated = {
        "source_a_score": pd.DataFrame(
            {
                "gene": [
                    "A",
                    "C",
                    "D",
                ],
                "rank": [
                    1,
                    2,
                    3,
                ],
            }
        )
    }

    summary = create_ablation_summary(
        full,
        ablated,
        top_k=2,
    )

    assert summary["top_k_overlap"].iloc[0] == 1


def test_run_target_prioritization(
    markers: pd.DataFrame,
    trajectory_genes: pd.DataFrame,
    differential_expression: pd.DataFrame,
    regulators: pd.DataFrame,
    transition_coefficients: pd.DataFrame,
    signature: pd.DataFrame,
) -> None:
    (
        rankings,
        evidence_matrix,
        stability,
        ablation_summary,
        candidate_summary,
    ) = run_target_prioritization(
        markers,
        trajectory_genes,
        differential_expression,
        regulators,
        transition_coefficients,
        signature,
        maximum_targets=20,
    )

    assert not rankings.empty
    assert not evidence_matrix.empty
    assert not stability.empty
    assert not ablation_summary.empty
    assert not candidate_summary.empty

    assert "E2F1" in set(rankings["gene"])

    e2f1 = rankings.loc[rankings["gene"] == "E2F1"].iloc[0]

    assert e2f1["evidence_source_count"] >= 5


def test_create_candidate_summary() -> None:
    rankings = pd.DataFrame(
        {
            "rank": [
                1,
            ],
            "gene": [
                "E2F1",
            ],
            "priority_score": [
                0.9,
            ],
            "evidence_source_count": [
                5,
            ],
            "evidence_sources": [
                "a;b;c",
            ],
            "regulatory_effect": [
                0.8,
            ],
        }
    )

    summary = create_candidate_summary(rankings)

    assert "regulatory_effect" in (summary.columns)


def test_save_results(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame(
        {
            "gene": [
                "E2F1",
            ]
        }
    )

    save_results(
        table,
        table,
        table,
        table,
        table,
        tmp_path,
    )

    expected_files = [
        "target_priority_rankings.csv",
        "target_evidence_matrix.csv",
        "target_rank_stability.csv",
        "target_ablation_summary.csv",
        "target_candidate_summary.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
