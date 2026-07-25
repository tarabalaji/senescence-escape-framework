from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.differential_expression import (
    create_gene_summary,
    filter_significant_genes,
    prepare_cell_line_dataset,
    run_comparison,
    run_differential_expression,
    save_results,
    validate_dataset,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    random_generator = np.random.default_rng(42)

    cell_lines = []
    conditions = []
    sample_ids = []
    expression_rows = []

    genes = [
        "GENE_TIS",
        "GENE_REPOP",
        "GENE_CONTROL",
        "GENE_SHARED",
        "GENE_NOISE_1",
        "GENE_NOISE_2",
    ]

    for cell_line in ["MCF7", "T47D"]:
        for condition in ["CTR", "TIS", "REPOP"]:
            for cell_index in range(12):
                expression = random_generator.poisson(
                    lam=2,
                    size=len(genes),
                ).astype(float)

                if condition == "CTR":
                    expression[2] += 10

                if condition == "TIS":
                    expression[0] += 12
                    expression[3] += 5

                if condition == "REPOP":
                    expression[1] += 12
                    expression[3] += 8

                expression_rows.append(expression)
                cell_lines.append(cell_line)
                conditions.append(condition)
                sample_ids.append(f"{cell_line}-{condition}-{cell_index % 2 + 1}")

    counts = np.asarray(
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
        X=counts.copy(),
        obs=obs,
        var=var,
    )

    normalized = dataset.copy()

    cell_totals = normalized.X.sum(
        axis=1,
        keepdims=True,
    )

    normalized.X = np.log1p(normalized.X / cell_totals * 10_000)

    dataset.X = normalized.X.copy()
    dataset.raw = normalized.copy()

    return dataset


def test_validate_dataset_accepts_valid_data(
    example_dataset: ad.AnnData,
) -> None:
    validate_dataset(example_dataset)


def test_validate_dataset_rejects_missing_metadata() -> None:
    dataset = ad.AnnData(
        X=np.ones((5, 3)),
        obs=pd.DataFrame(
            {
                "condition": ["CTR"] * 5,
            }
        ),
    )

    dataset.raw = dataset.copy()

    with pytest.raises(
        ValueError,
        match="missing required metadata columns",
    ):
        validate_dataset(dataset)


def test_validate_dataset_rejects_missing_raw(
    example_dataset: ad.AnnData,
) -> None:
    example_dataset.raw = None

    with pytest.raises(
        ValueError,
        match="does not contain normalized expression",
    ):
        validate_dataset(example_dataset)


def test_prepare_cell_line_dataset(
    example_dataset: ad.AnnData,
) -> None:
    subset = prepare_cell_line_dataset(
        example_dataset,
        "MCF7",
    )

    assert subset.n_obs == 36
    assert set(subset.obs["cell_line"].astype(str)) == {"MCF7"}


def test_prepare_missing_cell_line_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    with pytest.raises(
        ValueError,
        match="No cells were found",
    ):
        prepare_cell_line_dataset(
            example_dataset,
            "INVALID",
        )


def test_run_comparison(
    example_dataset: ad.AnnData,
) -> None:
    mcf7_data = prepare_cell_line_dataset(
        example_dataset,
        "MCF7",
    )

    results = run_comparison(
        mcf7_data,
        group="TIS",
        reference="CTR",
    )

    assert not results.empty

    expected_columns = {
        "gene",
        "score",
        "log2_fold_change",
        "p_value",
        "adjusted_p_value",
        "fraction_expressed_group",
        "fraction_expressed_reference",
        "group",
        "reference",
        "comparison",
        "group_cell_count",
        "reference_cell_count",
    }

    assert set(results.columns) == expected_columns
    assert set(results["group"]) == {"TIS"}
    assert set(results["reference"]) == {"CTR"}
    assert results["group_cell_count"].iloc[0] == 12
    assert results["reference_cell_count"].iloc[0] == 12


def test_tis_gene_is_upregulated(
    example_dataset: ad.AnnData,
) -> None:
    mcf7_data = prepare_cell_line_dataset(
        example_dataset,
        "MCF7",
    )

    results = run_comparison(
        mcf7_data,
        group="TIS",
        reference="CTR",
    )

    tis_gene = results.loc[results["gene"] == "GENE_TIS"].iloc[0]

    assert tis_gene["log2_fold_change"] > 0


def test_repopulation_gene_is_upregulated(
    example_dataset: ad.AnnData,
) -> None:
    t47d_data = prepare_cell_line_dataset(
        example_dataset,
        "T47D",
    )

    results = run_comparison(
        t47d_data,
        group="REPOP",
        reference="TIS",
    )

    repop_gene = results.loc[results["gene"] == "GENE_REPOP"].iloc[0]

    assert repop_gene["log2_fold_change"] > 0


def test_run_differential_expression(
    example_dataset: ad.AnnData,
) -> None:
    results = run_differential_expression(example_dataset)

    assert set(results["cell_line"]) == {
        "MCF7",
        "T47D",
    }

    assert set(results["comparison"]) == {
        "TIS_vs_CTR",
        "REPOP_vs_TIS",
        "REPOP_vs_CTR",
    }

    expected_rows = 2 * 3 * example_dataset.n_vars

    assert len(results) == expected_rows


def test_invalid_comparison_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    mcf7_data = prepare_cell_line_dataset(
        example_dataset,
        "MCF7",
    )

    with pytest.raises(
        ValueError,
        match="conditions not found",
    ):
        run_comparison(
            mcf7_data,
            group="INVALID",
            reference="CTR",
        )


def test_filter_significant_genes() -> None:
    results = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "adjusted_p_value": [
                0.01,
                0.04,
                0.10,
                0.001,
            ],
            "log2_fold_change": [
                1.0,
                0.1,
                2.0,
                -0.8,
            ],
        }
    )

    significant = filter_significant_genes(
        results,
        adjusted_p_value_threshold=0.05,
        minimum_absolute_log2_fold_change=0.25,
    )

    assert significant["gene"].tolist() == [
        "A",
        "D",
    ]


def test_filter_significant_genes_rejects_invalid_threshold() -> None:
    results = pd.DataFrame(
        {
            "adjusted_p_value": [0.01],
            "log2_fold_change": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        filter_significant_genes(
            results,
            adjusted_p_value_threshold=0,
        )


def test_create_gene_summary() -> None:
    results = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "MCF7",
                "MCF7",
                "T47D",
            ],
            "comparison": [
                "TIS_vs_CTR",
                "TIS_vs_CTR",
                "REPOP_vs_TIS",
                "TIS_vs_CTR",
            ],
            "gene": ["A", "B", "C", "D"],
            "log2_fold_change": [
                1.0,
                -1.0,
                0.5,
                2.0,
            ],
            "adjusted_p_value": [
                0.01,
                0.02,
                0.03,
                0.01,
            ],
        }
    )

    summary = create_gene_summary(results)

    mcf7_tis = summary[
        (summary["cell_line"] == "MCF7") & (summary["comparison"] == "TIS_vs_CTR")
    ].iloc[0]

    assert mcf7_tis["significant_genes"] == 2
    assert mcf7_tis["upregulated_genes"] == 1
    assert mcf7_tis["downregulated_genes"] == 1


def test_save_results(
    tmp_path: Path,
) -> None:
    all_results = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "adjusted_p_value": [0.01, 0.2],
            "log2_fold_change": [1.0, 0.1],
        }
    )

    significant_results = all_results.iloc[[0]].copy()

    summary = pd.DataFrame(
        {
            "cell_line": ["MCF7"],
            "comparison": ["TIS_vs_CTR"],
            "significant_genes": [1],
            "upregulated_genes": [1],
            "downregulated_genes": [0],
        }
    )

    save_results(
        all_results,
        significant_results,
        summary,
        tmp_path,
    )

    assert (tmp_path / "differential_expression_all.csv").exists()

    assert (tmp_path / "differential_expression_significant.csv").exists()

    assert (tmp_path / "differential_expression_summary.csv").exists()
