from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.target_prioritization import (
    calculate_priority_scores,
    create_ablation_summary,
    merge_evidence_tables,
    prepare_differential_expression_evidence,
    prepare_escape_marker_evidence,
    prepare_signature_evidence,
    prepare_trajectory_evidence,
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

    for cell_line in ["MCF7", "T47D"]:
        for gene, base_effect in effects.items():
            effect = base_effect

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
            ],
            "mean_regulatory_score": [
                0.80,
                0.50,
            ],
        }
    )


@pytest.fixture
def regulatory_edges() -> pd.DataFrame:
    rows = []

    for cell_line in ["MCF7", "T47D"]:
        rows.extend(
            [
                {
                    "cell_line": cell_line,
                    "transcription_factor": "E2F1",
                    "target_gene": "TOP2A",
                    "absolute_tf_target_correlation": 0.80,
                    "rap_consistent": True,
                },
                {
                    "cell_line": cell_line,
                    "transcription_factor": "E2F1",
                    "target_gene": "MKI67",
                    "absolute_tf_target_correlation": 0.70,
                    "rap_consistent": True,
                },
                {
                    "cell_line": cell_line,
                    "transcription_factor": "MYC",
                    "target_gene": "TOP2A",
                    "absolute_tf_target_correlation": 0.60,
                    "rap_consistent": True,
                },
            ]
        )

    return pd.DataFrame(rows)


@pytest.fixture
def transition_coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "rap_score",
                "regulator__E2F1",
                "regulator__MYC",
            ],
            "coefficient": [
                8.0,
                0.8,
                -0.3,
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
            "source_a_score": 0.5,
            "source_b_score": 0.5,
        }
    )


def test_validate_weights_rejects_negative() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        validate_weights({"source_a_score": -1.0})


def test_prepare_escape_marker_evidence(
    markers: pd.DataFrame,
) -> None:
    evidence = prepare_escape_marker_evidence(markers)

    assert "escape_marker_score" in evidence.columns
    assert evidence["gene"].str.isupper().all()


def test_prepare_trajectory_evidence(
    trajectory_genes: pd.DataFrame,
) -> None:
    evidence = prepare_trajectory_evidence(trajectory_genes)

    clu = evidence.loc[evidence["gene"] == "CLU"].iloc[0]

    assert clu["trajectory_direction"] == "decreasing"


def test_prepare_differential_expression_excludes_inconsistent(
    differential_expression: pd.DataFrame,
) -> None:
    evidence = prepare_differential_expression_evidence(differential_expression)

    assert "INCONSISTENT" not in set(evidence["gene"])
    assert "TOP2A" in set(evidence["gene"])


def test_prepare_signature_evidence(
    signature: pd.DataFrame,
) -> None:
    evidence = prepare_signature_evidence(signature)

    assert evidence["signature_score"].eq(1.0).all()


def test_merge_evidence_tables() -> None:
    left = pd.DataFrame(
        {
            "gene": ["A"],
            "a_score": [1.0],
        }
    )
    right = pd.DataFrame(
        {
            "gene": ["B"],
            "b_score": [1.0],
        }
    )

    merged = merge_evidence_tables([left, right])

    assert set(merged["gene"]) == {"A", "B"}


def test_calculate_priority_scores() -> None:
    evidence = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "source_a_score": [1.0, 0.5, np.nan],
            "source_b_score": [1.0, 0.2, 1.0],
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
    assert rankings.iloc[0]["priority_score"] == pytest.approx(1.0)


def test_create_ablation_summary() -> None:
    full = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "rank": [1, 2, 3],
        }
    )
    ablated = {
        "source_a_score": pd.DataFrame(
            {
                "gene": ["A", "C", "D"],
                "rank": [1, 2, 3],
            }
        )
    }

    summary = create_ablation_summary(
        full,
        ablated,
        top_k=2,
    )

    assert summary.iloc[0]["top_k_overlap"] == 1


def test_complete_target_prioritization(
    markers: pd.DataFrame,
    trajectory_genes: pd.DataFrame,
    differential_expression: pd.DataFrame,
    regulators: pd.DataFrame,
    regulatory_edges: pd.DataFrame,
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
        markers=markers,
        trajectory_genes=trajectory_genes,
        differential_expression=(differential_expression),
        regulators=regulators,
        regulatory_edges=regulatory_edges,
        transition_coefficients=(transition_coefficients),
        signature=signature,
        maximum_targets=20,
    )

    assert not rankings.empty
    assert not evidence_matrix.empty
    assert not stability.empty
    assert not ablation_summary.empty
    assert not candidate_summary.empty

    top2a = rankings.loc[rankings["gene"] == "TOP2A"].iloc[0]

    assert pd.notna(top2a["regulatory_score"])
    assert pd.notna(top2a["transition_model_score"])
    assert top2a["evidence_source_count"] >= 5
    assert "E2F1" in top2a["upstream_regulators"]


def test_save_results(tmp_path: Path) -> None:
    table = pd.DataFrame({"gene": ["TOP2A"]})

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
