from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.perturbation import (
    calculate_confidence_score,
    create_gene_pathway_map,
    determine_intervention,
    infer_escape_association,
    parse_gene_list,
    prepare_model_data,
    rank_interventions,
    run_perturbation_analysis,
    save_results,
    simulate_target_perturbation,
    validate_inputs,
)


@pytest.fixture
def feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = np.array(
        [0] * 30 + [1] * 30,
        dtype=int,
    )
    signal = labels + rng.normal(
        0,
        0.1,
        len(labels),
    )

    return pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(len(labels))],
            "rap_score": signal,
            "pathway__e2f_targets": (signal + rng.normal(0, 0.05, len(labels))),
            "pathway__g2_m_checkpoint": (signal + rng.normal(0, 0.05, len(labels))),
            "regulator__E2F1": signal,
            "transition_label": labels,
        }
    )


@pytest.fixture
def target_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "gene": [
                "E2F1",
                "CCNB1",
                "CDKN1A",
            ],
            "priority_score": [
                0.90,
                0.80,
                0.70,
            ],
            "evidence_source_count": [
                5,
                4,
                3,
            ],
            "differential_expression_effect": [
                1.5,
                1.2,
                -1.0,
            ],
        }
    )


@pytest.fixture
def pathway_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
            ],
            "pathway": [
                "E2F Targets",
                "G2 M Checkpoint",
            ],
            "leading_edge_genes": [
                "E2F1;CCNB1",
                "CCNB1|CDKN1A",
            ],
            "normalized_enrichment_score": [
                2.0,
                1.5,
            ],
            "false_discovery_rate": [
                0.01,
                0.02,
            ],
        }
    )


def test_parse_gene_list() -> None:
    assert parse_gene_list("E2F1; CCNB1|E2F1") == [
        "E2F1",
        "CCNB1",
    ]


def test_validate_inputs(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
) -> None:
    validate_inputs(
        feature_table,
        target_rankings,
        pathway_results,
    )


def test_validate_inputs_missing_pathways(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
) -> None:
    invalid = feature_table.drop(
        columns=[
            "pathway__e2f_targets",
            "pathway__g2_m_checkpoint",
        ]
    )

    with pytest.raises(
        ValueError,
        match="no pathway features",
    ):
        validate_inputs(
            invalid,
            target_rankings,
            pathway_results,
        )


def test_prepare_model_data(
    feature_table: pd.DataFrame,
) -> None:
    features, labels = prepare_model_data(feature_table)

    assert list(features.columns) == [
        "rap_score",
        "pathway__e2f_targets",
        "pathway__g2_m_checkpoint",
    ]
    assert labels.nunique() == 2


def test_infer_escape_association() -> None:
    row = pd.Series(
        {
            "gene": "E2F1",
            "differential_expression_effect": 1.5,
        }
    )

    sign, source = infer_escape_association(row)

    assert sign == 1.0
    assert source == ("differential_expression_effect")


def test_determine_intervention() -> None:
    positive = pd.Series(
        {
            "differential_expression_effect": 1.0,
        }
    )
    negative = pd.Series(
        {
            "differential_expression_effect": -1.0,
        }
    )

    assert determine_intervention(positive)[0] == "inhibit"

    assert determine_intervention(negative)[0] == "activate"


def test_create_gene_pathway_map(
    feature_table: pd.DataFrame,
    pathway_results: pd.DataFrame,
) -> None:
    features, _ = prepare_model_data(feature_table)

    mapping = create_gene_pathway_map(
        pathway_results,
        features.columns.tolist(),
    )

    assert len(mapping["E2F1"]) == 1
    assert len(mapping["CCNB1"]) == 2
    assert mapping["E2F1"][0]["feature"] == ("pathway__e2f_targets")


def test_calculate_confidence_score() -> None:
    score = calculate_confidence_score(
        priority_score=0.8,
        pathway_count=2,
        maximum_pathway_count=4,
        evidence_source_count=5,
    )

    assert 0 <= score <= 1


def test_simulate_target_perturbation(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
) -> None:
    from src.perturbation import create_model

    features, labels = prepare_model_data(feature_table)
    model = create_model()
    model.fit(features, labels)

    mapping = create_gene_pathway_map(
        pathway_results,
        features.columns.tolist(),
    )

    target = target_rankings.iloc[0]

    predictions, summary = simulate_target_perturbation(
        features,
        model,
        target,
        mapping["E2F1"],
    )

    assert len(predictions) == len(features)
    assert summary["target"] == "E2F1"
    assert summary["intervention"] == "inhibit"
    assert summary["mean_predicted_escape_reduction"] > 0


def test_rank_interventions() -> None:
    summary = pd.DataFrame(
        {
            "target": ["A", "B"],
            "priority_score": [0.8, 0.9],
            "evidence_source_count": [4, 5],
            "affected_pathway_count": [1, 2],
            "mean_predicted_escape_reduction": [
                0.05,
                0.10,
            ],
        }
    )

    ranked = rank_interventions(summary)

    assert ranked.iloc[0]["target"] == "B"
    assert list(ranked["perturbation_rank"]) == [1, 2]


def test_run_perturbation_analysis(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
) -> None:
    predictions, summary, rankings = run_perturbation_analysis(
        feature_table,
        target_rankings,
        pathway_results,
        maximum_targets=3,
    )

    assert len(summary) == 3
    assert len(rankings) == 3
    assert len(predictions) == (len(feature_table) * 3)
    assert rankings.iloc[0]["perturbation_rank"] == 1
    assert set(rankings["intervention"]) == {
        "inhibit",
        "activate",
    }


def test_save_results(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame({"value": [1]})

    save_results(
        table,
        table,
        table,
        tmp_path,
    )

    expected_files = [
        "perturbation_predictions.csv",
        "perturbation_summary.csv",
        "top_predicted_interventions.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()
