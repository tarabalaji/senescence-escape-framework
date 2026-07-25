from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.escape_index import (
    add_escape_index,
    calculate_signature_score,
    create_score_summary,
    save_outputs,
    select_conserved_signature,
    standardize_genes,
    standardize_score_within_cell_line,
    validate_dataset,
    validate_differential_expression,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    genes = [
        "ESCAPE_UP_1",
        "ESCAPE_UP_2",
        "ESCAPE_DOWN_1",
        "ESCAPE_DOWN_2",
        "NOISE",
    ]

    expression_rows = []
    cell_lines = []
    conditions = []
    sample_ids = []

    random_generator = np.random.default_rng(42)

    for cell_line in ["MCF7", "T47D"]:
        for condition in ["CTR", "TIS", "REPOP"]:
            for cell_index in range(10):
                expression = random_generator.normal(
                    loc=2.0,
                    scale=0.2,
                    size=len(genes),
                )

                if condition == "TIS":
                    expression[2] += 3
                    expression[3] += 3

                if condition == "REPOP":
                    expression[0] += 5
                    expression[1] += 5

                expression_rows.append(expression)
                cell_lines.append(cell_line)
                conditions.append(condition)
                sample_ids.append(f"{cell_line}-{condition}-{cell_index % 2 + 1}")

    expression_matrix = np.asarray(
        expression_rows,
        dtype=np.float32,
    )

    obs = pd.DataFrame(
        {
            "cell_line": cell_lines,
            "condition": conditions,
            "sample_id": sample_ids,
            "replicate": [str(index % 2 + 1) for index in range(len(cell_lines))],
        },
        index=[f"CELL_{index}" for index in range(len(cell_lines))],
    )

    var = pd.DataFrame(
        index=genes,
    )

    dataset = ad.AnnData(
        X=expression_matrix.copy(),
        obs=obs,
        var=var,
    )

    dataset.raw = dataset.copy()

    return dataset


@pytest.fixture
def differential_expression_results() -> pd.DataFrame:
    rows = []

    gene_effects = {
        "ESCAPE_UP_1": 2.5,
        "ESCAPE_UP_2": 1.8,
        "ESCAPE_DOWN_1": -2.2,
        "ESCAPE_DOWN_2": -1.5,
        "NONCONSERVED": 2.0,
        "WEAK_GENE": 0.1,
    }

    for cell_line in ["MCF7", "T47D"]:
        for gene, effect in gene_effects.items():
            adjusted_p_value = 0.001

            if gene == "NONCONSERVED" and cell_line == "T47D":
                adjusted_p_value = 0.5

            rows.append(
                {
                    "cell_line": cell_line,
                    "comparison": "REPOP_vs_TIS",
                    "gene": gene,
                    "log2_fold_change": effect,
                    "adjusted_p_value": adjusted_p_value,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def example_signature() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": [
                "ESCAPE_UP_1",
                "ESCAPE_UP_2",
                "ESCAPE_DOWN_1",
                "ESCAPE_DOWN_2",
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


def test_validate_dataset_rejects_missing_raw(
    example_dataset: ad.AnnData,
) -> None:
    example_dataset.raw = None

    with pytest.raises(
        ValueError,
        match="normalized expression",
    ):
        validate_dataset(example_dataset)


def test_validate_de_rejects_missing_columns() -> None:
    results = pd.DataFrame(
        {
            "gene": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_differential_expression(results)


def test_select_conserved_signature(
    differential_expression_results: pd.DataFrame,
) -> None:
    signature = select_conserved_signature(
        differential_expression_results,
        minimum_absolute_log2_fold_change=0.5,
        top_genes_per_direction=10,
    )

    selected_genes = set(signature["gene"])

    assert selected_genes == {
        "ESCAPE_UP_1",
        "ESCAPE_UP_2",
        "ESCAPE_DOWN_1",
        "ESCAPE_DOWN_2",
    }

    assert set(signature["direction"]) == {
        "up",
        "down",
    }

    assert (signature["cell_lines_supported"] == 2).all()


def test_nonconserved_gene_can_be_selected_with_lower_requirement(
    differential_expression_results: pd.DataFrame,
) -> None:
    signature = select_conserved_signature(
        differential_expression_results,
        minimum_absolute_log2_fold_change=0.5,
        top_genes_per_direction=10,
        minimum_cell_lines=1,
    )

    assert "NONCONSERVED" in set(signature["gene"])


def test_signature_respects_top_gene_limit(
    differential_expression_results: pd.DataFrame,
) -> None:
    signature = select_conserved_signature(
        differential_expression_results,
        minimum_absolute_log2_fold_change=0.5,
        top_genes_per_direction=1,
    )

    assert (signature["direction"] == "up").sum() == 1

    assert (signature["direction"] == "down").sum() == 1


def test_standardize_genes() -> None:
    expression = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    standardized = standardize_genes(expression)

    np.testing.assert_allclose(
        standardized.mean(axis=0),
        np.zeros(2),
        atol=1e-7,
    )

    np.testing.assert_allclose(
        standardized.std(axis=0),
        np.ones(2),
        atol=1e-7,
    )


def test_calculate_signature_score(
    example_dataset: ad.AnnData,
) -> None:
    scores = calculate_signature_score(
        example_dataset,
        up_genes=[
            "ESCAPE_UP_1",
            "ESCAPE_UP_2",
        ],
        down_genes=[
            "ESCAPE_DOWN_1",
            "ESCAPE_DOWN_2",
        ],
    )

    assert scores.shape == (example_dataset.n_obs,)

    conditions = example_dataset.obs["condition"].astype(str).to_numpy()

    repop_mean = scores[conditions == "REPOP"].mean()

    tis_mean = scores[conditions == "TIS"].mean()

    assert repop_mean > tis_mean


def test_standardize_score_within_cell_line(
    example_dataset: ad.AnnData,
) -> None:
    raw_scores = np.arange(
        example_dataset.n_obs,
        dtype=float,
    )

    standardized = standardize_score_within_cell_line(
        example_dataset,
        raw_scores,
    )

    cell_lines = example_dataset.obs["cell_line"].astype(str).to_numpy()

    for cell_line in ["MCF7", "T47D"]:
        group_scores = standardized[cell_lines == cell_line]

        assert np.isclose(
            group_scores.mean(),
            0.0,
        )

        assert np.isclose(
            group_scores.std(ddof=0),
            1.0,
        )


def test_add_escape_index(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
) -> None:
    scored = add_escape_index(
        example_dataset,
        example_signature,
    )

    assert "repopulation_associated_potential" in scored.obs.columns

    assert "repopulation_associated_potential_raw" in scored.obs.columns

    assert "escape_index" in scored.uns

    condition_means = scored.obs.groupby(
        "condition",
        observed=True,
    )["repopulation_associated_potential"].mean()

    assert condition_means["REPOP"] > condition_means["TIS"]


def test_add_escape_index_does_not_modify_original(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
) -> None:
    add_escape_index(
        example_dataset,
        example_signature,
    )

    assert "repopulation_associated_potential" not in example_dataset.obs.columns


def test_create_score_summary(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
) -> None:
    scored = add_escape_index(
        example_dataset,
        example_signature,
    )

    summary = create_score_summary(scored)

    expected_columns = {
        "cell_line",
        "condition",
        "sample_id",
        "cell_count",
        "mean_score",
        "median_score",
        "standard_deviation",
        "minimum_score",
        "maximum_score",
    }

    assert set(summary.columns) == expected_columns

    assert summary["cell_count"].sum() == (example_dataset.n_obs)


def test_missing_up_genes_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one upregulated",
    ):
        calculate_signature_score(
            example_dataset,
            up_genes=[],
            down_genes=["ESCAPE_DOWN_1"],
        )


def test_save_outputs(
    example_dataset: ad.AnnData,
    example_signature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    scored = add_escape_index(
        example_dataset,
        example_signature,
    )

    summary = create_score_summary(scored)

    dataset_path = tmp_path / "combined_scored.h5ad"
    signature_path = tmp_path / "signature.csv"
    summary_path = tmp_path / "summary.csv"

    save_outputs(
        scored,
        example_signature,
        summary,
        dataset_path,
        signature_path,
        summary_path,
    )

    assert dataset_path.exists()
    assert signature_path.exists()
    assert summary_path.exists()

    loaded = ad.read_h5ad(dataset_path)

    assert "repopulation_associated_potential" in loaded.obs.columns
