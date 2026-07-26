from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regulatory_propagation import (
    min_max_scale,
    normalize_gene,
    parse_boolean,
    prepare_direct_regulator_evidence,
    prepare_direct_transition_evidence,
    prepare_network_regulatory_evidence,
    prepare_network_transition_evidence,
    require_columns,
    summarize_regulatory_edges,
)


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
def edges() -> pd.DataFrame:
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


def test_normalize_gene() -> None:
    assert normalize_gene(" top2a ") == "TOP2A"
    assert normalize_gene(np.nan) == ""


def test_min_max_scale() -> None:
    result = min_max_scale(pd.Series([2.0, 4.0, 6.0]))

    np.testing.assert_allclose(
        result,
        [0.0, 0.5, 1.0],
    )


def test_min_max_scale_constant_positive() -> None:
    result = min_max_scale(pd.Series([3.0, 3.0]))

    np.testing.assert_allclose(
        result,
        [1.0, 1.0],
    )


def test_parse_boolean() -> None:
    result = parse_boolean(
        pd.Series(
            [
                True,
                False,
                "yes",
                "0",
            ]
        )
    )

    np.testing.assert_allclose(
        result,
        [1.0, 0.0, 1.0, 0.0],
    )


def test_require_columns_rejects_missing() -> None:
    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        require_columns(
            pd.DataFrame({"a": [1]}),
            {"a", "b"},
            "Example",
        )


def test_prepare_direct_regulator_evidence(
    regulators: pd.DataFrame,
) -> None:
    evidence = prepare_direct_regulator_evidence(regulators)

    assert set(evidence["gene"]) == {
        "E2F1",
        "MYC",
    }
    assert set(evidence["regulatory_evidence_type"]) == {"direct"}


def test_prepare_direct_transition_evidence(
    transition_coefficients: pd.DataFrame,
) -> None:
    evidence = prepare_direct_transition_evidence(transition_coefficients)

    assert set(evidence["gene"]) == {
        "E2F1",
        "MYC",
    }
    assert "rap_score" not in set(evidence["gene"])


def test_summarize_regulatory_edges(
    edges: pd.DataFrame,
) -> None:
    summarized = summarize_regulatory_edges(edges)

    row = summarized.loc[
        (summarized["transcription_factor"] == "E2F1")
        & (summarized["target_gene"] == "TOP2A")
    ].iloc[0]

    assert row["cell_lines_supported"] == 2
    assert row["rap_consistency_rate"] == 1.0
    assert 0.0 <= row["edge_support_score"] <= 1.0


def test_regulatory_evidence_propagates(
    regulators: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    evidence = prepare_network_regulatory_evidence(
        regulators,
        edges,
    )

    top2a = evidence.loc[evidence["gene"] == "TOP2A"].iloc[0]

    assert top2a["regulatory_score"] > 0
    assert top2a["regulatory_evidence_type"] == "propagated"
    assert set(top2a["upstream_regulators"].split(";")) == {"E2F1", "MYC"}


def test_direct_regulator_is_retained(
    regulators: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    evidence = prepare_network_regulatory_evidence(
        regulators,
        edges,
    )

    e2f1 = evidence.loc[evidence["gene"] == "E2F1"].iloc[0]

    assert e2f1["regulatory_score"] >= 0
    assert e2f1["regulatory_evidence_type"] in {
        "direct",
        "direct_and_propagated",
    }


def test_transition_evidence_propagates(
    transition_coefficients: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    evidence = prepare_network_transition_evidence(
        transition_coefficients,
        edges,
    )

    top2a = evidence.loc[evidence["gene"] == "TOP2A"].iloc[0]

    assert top2a["transition_model_score"] > 0
    assert top2a["transition_evidence_type"] == "propagated"
    assert "E2F1" in top2a["transition_upstream_regulators"]
