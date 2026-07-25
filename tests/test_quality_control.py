from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.quality_control import (
    QCThresholds,
    add_qc_flags,
    calculate_qc_metrics,
    create_qc_summary,
    filter_cells,
    identify_mitochondrial_genes,
    save_qc_outputs,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    counts = sparse.csr_matrix(
        np.array(
            [
                [100, 50, 0, 0],
                [10, 5, 0, 0],
                [100, 100, 100, 100],
                [20, 10, 10, 60],
                [50, 50, 50, 0],
                [500, 500, 500, 500],
            ],
            dtype=np.int32,
        )
    )

    obs = pd.DataFrame(
        {
            "sample_id": [
                "MCF7-ctr-1",
                "MCF7-ctr-1",
                "MCF7-tis-1",
                "T47D-ctr-1",
                "T47D-tis-1",
                "T47D-repop-1",
            ],
            "cell_line": [
                "MCF7",
                "MCF7",
                "MCF7",
                "T47D",
                "T47D",
                "T47D",
            ],
            "condition": [
                "CTR",
                "CTR",
                "TIS",
                "CTR",
                "TIS",
                "REPOP",
            ],
            "replicate": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
        },
        index=[f"CELL_{index}" for index in range(6)],
    )

    var = pd.DataFrame(
        {
            "gene_symbol": [
                "GENE1",
                "GENE2",
                "MT-CO1",
                "MT-ND1",
            ]
        },
        index=[
            "GENE1",
            "GENE2",
            "MT-CO1",
            "MT-ND1",
        ],
    )

    return ad.AnnData(
        X=counts,
        obs=obs,
        var=var,
    )


@pytest.fixture
def test_thresholds() -> dict[str, QCThresholds]:
    return {
        "MCF7": QCThresholds(
            min_genes=2,
            min_counts=100,
            max_counts=1000,
            max_mito_pct=25.0,
        ),
        "T47D": QCThresholds(
            min_genes=3,
            min_counts=100,
            max_counts=1000,
            max_mito_pct=40.0,
        ),
    }


def test_identify_mitochondrial_genes(
    example_dataset: ad.AnnData,
) -> None:
    mitochondrial_mask = identify_mitochondrial_genes(example_dataset)

    np.testing.assert_array_equal(
        mitochondrial_mask,
        np.array([False, False, True, True]),
    )


def test_calculate_qc_metrics(
    example_dataset: ad.AnnData,
) -> None:
    qc_data = calculate_qc_metrics(example_dataset)

    assert "total_counts" in qc_data.obs.columns
    assert "n_genes_by_counts" in qc_data.obs.columns
    assert "mitochondrial_counts" in qc_data.obs.columns
    assert "pct_counts_mito" in qc_data.obs.columns
    assert "is_mitochondrial" in qc_data.var.columns

    np.testing.assert_array_equal(
        qc_data.obs["total_counts"].to_numpy(),
        np.array([150, 15, 400, 100, 150, 2000]),
    )

    np.testing.assert_array_equal(
        qc_data.obs["n_genes_by_counts"].to_numpy(),
        np.array([2, 2, 4, 4, 3, 4]),
    )


def test_calculate_mitochondrial_percentage(
    example_dataset: ad.AnnData,
) -> None:
    qc_data = calculate_qc_metrics(example_dataset)

    expected_percentages = np.array([0.0, 0.0, 50.0, 70.0, 33.333333, 50.0])

    np.testing.assert_allclose(
        qc_data.obs["pct_counts_mito"].to_numpy(),
        expected_percentages,
        rtol=1e-5,
    )


def test_calculate_qc_metrics_does_not_modify_original(
    example_dataset: ad.AnnData,
) -> None:
    calculate_qc_metrics(example_dataset)

    assert "total_counts" not in example_dataset.obs.columns
    assert "is_mitochondrial" not in example_dataset.var.columns


def test_add_qc_flags(
    example_dataset: ad.AnnData,
    test_thresholds: dict[str, QCThresholds],
) -> None:
    flagged_data = add_qc_flags(
        example_dataset,
        thresholds=test_thresholds,
    )

    expected_passes = [
        True,
        False,
        False,
        False,
        True,
        False,
    ]

    assert flagged_data.obs["passes_qc"].tolist() == expected_passes

    assert bool(
        flagged_data.obs.loc[
            "CELL_1",
            "fails_min_counts",
        ]
    )

    assert bool(
        flagged_data.obs.loc[
            "CELL_2",
            "fails_mito",
        ]
    )

    assert bool(
        flagged_data.obs.loc[
            "CELL_3",
            "fails_mito",
        ]
    )

    assert bool(
        flagged_data.obs.loc[
            "CELL_5",
            "fails_max_counts",
        ]
    )


def test_filter_cells(
    example_dataset: ad.AnnData,
    test_thresholds: dict[str, QCThresholds],
) -> None:
    filtered_data = filter_cells(
        example_dataset,
        thresholds=test_thresholds,
    )

    assert filtered_data.n_obs == 2
    assert filtered_data.obs_names.tolist() == [
        "CELL_0",
        "CELL_4",
    ]

    assert filtered_data.uns["cells_before_qc"] == 6
    assert filtered_data.uns["cells_after_qc"] == 2
    assert filtered_data.uns["cells_removed_qc"] == 4


def test_create_qc_summary(
    example_dataset: ad.AnnData,
    test_thresholds: dict[str, QCThresholds],
) -> None:
    flagged_data = add_qc_flags(
        example_dataset,
        thresholds=test_thresholds,
    )

    summary = create_qc_summary(flagged_data)

    assert not summary.empty

    expected_columns = {
        "cell_line",
        "condition",
        "sample_id",
        "cells_before",
        "cells_after",
        "cells_removed",
        "retention_pct",
        "failed_min_genes",
        "failed_min_counts",
        "failed_max_counts",
        "failed_mito",
    }

    assert set(summary.columns) == expected_columns

    assert summary["cells_before"].sum() == 6
    assert summary["cells_after"].sum() == 2
    assert summary["cells_removed"].sum() == 4


def test_missing_cell_line_threshold_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    incomplete_thresholds = {
        "MCF7": QCThresholds(
            min_genes=2,
            min_counts=100,
            max_counts=1000,
            max_mito_pct=25,
        )
    }

    with pytest.raises(
        ValueError,
        match="No quality-control thresholds",
    ):
        add_qc_flags(
            example_dataset,
            thresholds=incomplete_thresholds,
        )


def test_invalid_thresholds_raise_error(
    example_dataset: ad.AnnData,
) -> None:
    invalid_thresholds = {
        "MCF7": QCThresholds(
            min_genes=2,
            min_counts=1000,
            max_counts=100,
            max_mito_pct=25,
        ),
        "T47D": QCThresholds(
            min_genes=2,
            min_counts=100,
            max_counts=1000,
            max_mito_pct=25,
        ),
    }

    with pytest.raises(
        ValueError,
        match="max_counts must be greater",
    ):
        add_qc_flags(
            example_dataset,
            thresholds=invalid_thresholds,
        )


def test_dataset_missing_metadata_raises_error() -> None:
    dataset = ad.AnnData(
        X=sparse.csr_matrix(np.array([[1, 2], [3, 4]])),
        obs=pd.DataFrame(index=["CELL_1", "CELL_2"]),
        var=pd.DataFrame(index=["GENE1", "MT-CO1"]),
    )

    with pytest.raises(
        ValueError,
        match="missing required metadata columns",
    ):
        calculate_qc_metrics(dataset)


def test_zero_count_cell_has_zero_mito_percentage() -> None:
    dataset = ad.AnnData(
        X=sparse.csr_matrix(np.array([[0, 0]])),
        obs=pd.DataFrame(
            {
                "sample_id": ["sample-1"],
                "cell_line": ["MCF7"],
                "condition": ["CTR"],
                "replicate": ["1"],
            },
            index=["CELL_1"],
        ),
        var=pd.DataFrame(index=["GENE1", "MT-CO1"]),
    )

    qc_data = calculate_qc_metrics(dataset)

    assert (
        qc_data.obs.loc[
            "CELL_1",
            "pct_counts_mito",
        ]
        == 0.0
    )


def test_save_qc_outputs(
    example_dataset: ad.AnnData,
    test_thresholds: dict[str, QCThresholds],
    tmp_path: Path,
) -> None:
    flagged_data = add_qc_flags(
        example_dataset,
        thresholds=test_thresholds,
    )

    filtered_data = flagged_data[flagged_data.obs["passes_qc"]].copy()

    summary = create_qc_summary(flagged_data)

    output_path = tmp_path / "combined_qc.h5ad"
    summary_path = tmp_path / "qc_summary.csv"

    save_qc_outputs(
        filtered_data,
        summary,
        output_path,
        summary_path,
    )

    assert output_path.exists()
    assert summary_path.exists()

    saved_summary = pd.read_csv(summary_path)

    assert saved_summary["cells_before"].sum() == 6
    assert saved_summary["cells_after"].sum() == 2
