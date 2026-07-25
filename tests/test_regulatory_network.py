from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.regulatory_network import (
    calculate_spearman_correlation,
    create_conserved_regulator_summary,
    infer_regulatory_edges,
    load_transcription_factors,
    rank_regulators_by_rap,
    residualize_within_samples,
    run_regulatory_network,
    save_results,
    summarize_regulators,
    validate_dataset,
    validate_signature,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    genes = [
        "TF_POSITIVE",
        "TF_NEGATIVE",
        "TF_NOISE",
        "UP_TARGET_1",
        "UP_TARGET_2",
        "DOWN_TARGET_1",
        "DOWN_TARGET_2",
    ]

    rows = []
    cell_lines = []
    conditions = []
    sample_ids = []
    rap_scores = []

    random_generator = np.random.default_rng(42)

    for cell_line in ["MCF7", "T47D"]:
        for condition_index, condition in enumerate(["CTR", "TIS", "REPOP"]):
            for cell_index in range(15):
                if condition == "CTR":
                    rap = 0.3
                elif condition == "TIS":
                    rap = -1.0
                else:
                    rap = 1.0

                rap += random_generator.normal(
                    0,
                    0.15,
                )

                tf_positive = rap + random_generator.normal(0, 0.1) + 3

                tf_negative = -rap + random_generator.normal(0, 0.1) + 3

                tf_noise = random_generator.normal(
                    3,
                    1,
                )

                expression = [
                    tf_positive,
                    tf_negative,
                    tf_noise,
                    rap + random_generator.normal(0, 0.1) + 3,
                    rap + random_generator.normal(0, 0.1) + 3,
                    -rap + random_generator.normal(0, 0.1) + 3,
                    -rap + random_generator.normal(0, 0.1) + 3,
                ]

                rows.append(expression)
                cell_lines.append(cell_line)
                conditions.append(condition)
                sample_ids.append(f"{cell_line}-{condition}-{cell_index % 2 + 1}")
                rap_scores.append(rap)

    expression_matrix = np.maximum(
        np.asarray(rows, dtype=np.float32),
        0,
    )

    obs = pd.DataFrame(
        {
            "cell_line": cell_lines,
            "condition": conditions,
            "sample_id": sample_ids,
            "repopulation_associated_potential": (rap_scores),
        },
        index=[f"CELL_{index}" for index in range(len(rows))],
    )

    dataset = ad.AnnData(
        X=expression_matrix,
        obs=obs,
        var=pd.DataFrame(index=genes),
    )

    dataset.raw = dataset.copy()

    return dataset


@pytest.fixture
def example_signature() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": [
                "UP_TARGET_1",
                "UP_TARGET_2",
                "DOWN_TARGET_1",
                "DOWN_TARGET_2",
            ],
            "direction": [
                "up",
                "up",
                "down",
                "down",
            ],
        }
    )


def test_validate_dataset_accepts_valid_data(
    example_dataset: ad.AnnData,
) -> None:
    validate_dataset(example_dataset)


def test_validate_dataset_rejects_missing_score(
    example_dataset: ad.AnnData,
) -> None:
    del example_dataset.obs["repopulation_associated_potential"]

    with pytest.raises(
        ValueError,
        match="does not contain score column",
    ):
        validate_dataset(example_dataset)


def test_validate_signature_rejects_invalid_direction() -> None:
    signature = pd.DataFrame(
        {
            "gene": ["A"],
            "direction": ["invalid"],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be 'up' or 'down'",
    ):
        validate_signature(signature)


def test_load_transcription_factors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tf_list.txt"

    path.write_text(
        "TF_POSITIVE\nTF_NEGATIVE\n# comment\nTF_POSITIVE\n",
        encoding="utf-8",
    )

    transcription_factors = load_transcription_factors(path)

    assert transcription_factors == [
        "TF_POSITIVE",
        "TF_NEGATIVE",
    ]


def test_residualize_within_samples() -> None:
    values = np.array([1.0, 3.0, 10.0, 14.0])

    sample_ids = np.array(["A", "A", "B", "B"])

    residuals = residualize_within_samples(
        values,
        sample_ids,
    )

    np.testing.assert_allclose(
        residuals,
        [-1.0, 1.0, -2.0, 2.0],
    )


def test_calculate_spearman_correlation() -> None:
    first = np.array([1, 2, 3, 4, 5])
    second = np.array([2, 4, 6, 8, 10])

    correlation, p_value = calculate_spearman_correlation(
        first,
        second,
    )

    assert np.isclose(correlation, 1.0)
    assert p_value < 0.05


def test_constant_vector_returns_zero_correlation() -> None:
    first = np.ones(10)
    second = np.arange(10)

    correlation, p_value = calculate_spearman_correlation(
        first,
        second,
    )

    assert correlation == 0.0
    assert p_value == 1.0


def test_rank_regulators_by_rap(
    example_dataset: ad.AnnData,
) -> None:
    regulators = rank_regulators_by_rap(
        example_dataset,
        [
            "TF_POSITIVE",
            "TF_NEGATIVE",
            "TF_NOISE",
        ],
        cell_line="MCF7",
        top_regulators=3,
    )

    assert len(regulators) == 3

    positive = regulators.loc[regulators["transcription_factor"] == "TF_POSITIVE"].iloc[
        0
    ]

    negative = regulators.loc[regulators["transcription_factor"] == "TF_NEGATIVE"].iloc[
        0
    ]

    assert positive["rap_correlation"] > 0
    assert negative["rap_correlation"] < 0


def test_infer_regulatory_edges(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
) -> None:
    regulators = pd.DataFrame(
        {
            "cell_line": ["MCF7", "MCF7"],
            "transcription_factor": [
                "TF_POSITIVE",
                "TF_NEGATIVE",
            ],
            "rap_correlation": [0.9, -0.9],
            "absolute_rap_correlation": [0.9, 0.9],
            "rap_p_value": [0.001, 0.001],
            "cell_count": [45, 45],
        }
    )

    edges = infer_regulatory_edges(
        example_dataset,
        regulators,
        example_signature,
        minimum_absolute_correlation=0.2,
    )

    assert not edges.empty

    assert {
        "TF_POSITIVE",
        "TF_NEGATIVE",
    }.issubset(set(edges["transcription_factor"]))


def test_summarize_regulators() -> None:
    regulators = pd.DataFrame(
        {
            "cell_line": ["MCF7"],
            "transcription_factor": ["TF_POSITIVE"],
            "rap_correlation": [0.8],
            "absolute_rap_correlation": [0.8],
            "rap_p_value": [0.001],
            "cell_count": [100],
        }
    )

    edges = pd.DataFrame(
        {
            "cell_line": ["MCF7"] * 3,
            "transcription_factor": ["TF_POSITIVE"] * 3,
            "target_gene": ["A", "B", "C"],
            "target_direction": [
                "up",
                "up",
                "down",
            ],
            "tf_target_correlation": [
                0.7,
                0.6,
                -0.5,
            ],
            "absolute_tf_target_correlation": [
                0.7,
                0.6,
                0.5,
            ],
            "tf_target_p_value": [
                0.001,
                0.002,
                0.003,
            ],
            "rap_correlation": [0.8] * 3,
            "rap_consistent": [
                True,
                True,
                True,
            ],
        }
    )

    summary = summarize_regulators(
        regulators,
        edges,
        minimum_targets=3,
    )

    assert len(summary) == 1
    assert summary["target_count"].iloc[0] == 3
    assert summary["consistency_fraction"].iloc[0] == 1.0
    assert summary["regulatory_score"].iloc[0] > 0


def test_create_conserved_regulator_summary() -> None:
    summary = pd.DataFrame(
        {
            "cell_line": ["MCF7", "T47D"],
            "transcription_factor": [
                "TF_POSITIVE",
                "TF_POSITIVE",
            ],
            "regulatory_score": [0.5, 0.7],
            "rap_correlation": [0.8, 0.9],
        }
    )

    conserved = create_conserved_regulator_summary(summary)

    assert len(conserved) == 1

    assert conserved["cell_lines_supported"].iloc[0] == 2

    assert np.isclose(
        conserved["mean_regulatory_score"].iloc[0],
        0.6,
    )


def test_run_regulatory_network(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
) -> None:
    (
        regulators,
        edges,
        regulator_summary,
        conserved_summary,
    ) = run_regulatory_network(
        example_dataset,
        example_signature,
        [
            "TF_POSITIVE",
            "TF_NEGATIVE",
            "TF_NOISE",
        ],
        top_regulators=3,
        minimum_absolute_correlation=0.2,
        minimum_targets=2,
    )

    assert not regulators.empty
    assert not edges.empty
    assert not regulator_summary.empty
    assert not conserved_summary.empty

    assert set(regulators["cell_line"]) == {
        "MCF7",
        "T47D",
    }


def test_save_results(
    tmp_path: Path,
) -> None:
    regulators = pd.DataFrame(
        {
            "transcription_factor": ["TF1"],
        }
    )

    edges = pd.DataFrame(
        {
            "transcription_factor": ["TF1"],
            "target_gene": ["GENE1"],
        }
    )

    regulator_summary = pd.DataFrame(
        {
            "transcription_factor": ["TF1"],
            "regulatory_score": [0.5],
        }
    )

    conserved_summary = pd.DataFrame(
        {
            "transcription_factor": ["TF1"],
            "cell_lines_supported": [2],
        }
    )

    save_results(
        regulators,
        edges,
        regulator_summary,
        conserved_summary,
        tmp_path,
    )

    assert (tmp_path / "rap_regulator_correlations.csv").exists()

    assert (tmp_path / "regulatory_network_edges.csv").exists()

    assert (tmp_path / "regulator_summary.csv").exists()

    assert (tmp_path / "conserved_regulators.csv").exists()
