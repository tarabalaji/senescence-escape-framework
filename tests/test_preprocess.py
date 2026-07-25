from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.preprocess import (
    normalize_and_log_transform,
    preprocess_dataset,
    preserve_raw_counts,
    save_dataset,
    scale_expression,
    select_highly_variable_genes,
    validate_dataset,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    counts = sparse.csr_matrix(
        np.array(
            [
                [10, 0, 5, 0, 1, 3],
                [4, 2, 0, 1, 0, 5],
                [0, 8, 3, 2, 1, 0],
                [6, 1, 4, 0, 2, 2],
                [2, 4, 1, 5, 0, 1],
                [1, 3, 6, 2, 4, 0],
            ],
            dtype=np.float32,
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
            "replicate": ["1"] * 6,
        },
        index=[f"CELL_{index}" for index in range(6)],
    )

    var = pd.DataFrame(index=[f"GENE_{index}" for index in range(6)])

    return ad.AnnData(
        X=counts,
        obs=obs,
        var=var,
    )


def test_validate_dataset_accepts_valid_data(
    example_dataset: ad.AnnData,
) -> None:
    validate_dataset(example_dataset)


def test_validate_dataset_rejects_empty_cells() -> None:
    dataset = ad.AnnData(
        X=np.empty((0, 3)),
        var=pd.DataFrame(index=["GENE_1", "GENE_2", "GENE_3"]),
    )

    with pytest.raises(
        ValueError,
        match="no cells",
    ):
        validate_dataset(dataset)


def test_validate_dataset_rejects_empty_genes() -> None:
    dataset = ad.AnnData(
        X=np.empty((3, 0)),
        obs=pd.DataFrame(index=["CELL_1", "CELL_2", "CELL_3"]),
    )

    with pytest.raises(
        ValueError,
        match="no genes",
    ):
        validate_dataset(dataset)


def test_validate_dataset_rejects_negative_values() -> None:
    dataset = ad.AnnData(
        X=np.array(
            [
                [1, -1],
                [2, 3],
            ],
            dtype=float,
        ),
    )

    with pytest.raises(
        ValueError,
        match="negative values",
    ):
        validate_dataset(dataset)


def test_preserve_raw_counts(
    example_dataset: ad.AnnData,
) -> None:
    original_counts = example_dataset.X.copy()

    preserve_raw_counts(example_dataset)

    assert "counts" in example_dataset.layers

    np.testing.assert_array_equal(
        example_dataset.layers["counts"].toarray(),
        original_counts.toarray(),
    )


def test_preserve_raw_counts_rejects_existing_layer(
    example_dataset: ad.AnnData,
) -> None:
    preserve_raw_counts(example_dataset)

    with pytest.raises(
        ValueError,
        match="would be overwritten",
    ):
        preserve_raw_counts(example_dataset)


def test_normalize_and_log_transform(
    example_dataset: ad.AnnData,
) -> None:
    normalize_and_log_transform(
        example_dataset,
        target_sum=100,
    )

    restored_counts = np.expm1(
        example_dataset.X.toarray()
        if sparse.issparse(example_dataset.X)
        else example_dataset.X
    )

    row_sums = restored_counts.sum(axis=1)

    np.testing.assert_allclose(
        row_sums,
        np.full(example_dataset.n_obs, 100.0),
        rtol=1e-5,
    )


def test_normalization_rejects_invalid_target_sum(
    example_dataset: ad.AnnData,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        normalize_and_log_transform(
            example_dataset,
            target_sum=0,
        )


def test_select_highly_variable_genes(
    example_dataset: ad.AnnData,
) -> None:
    normalize_and_log_transform(example_dataset)

    select_highly_variable_genes(
        example_dataset,
        n_top_genes=3,
    )

    assert "highly_variable" in example_dataset.var.columns
    assert int(example_dataset.var["highly_variable"].sum()) == 3


def test_highly_variable_gene_count_is_capped(
    example_dataset: ad.AnnData,
) -> None:
    normalize_and_log_transform(example_dataset)

    select_highly_variable_genes(
        example_dataset,
        n_top_genes=100,
    )

    assert int(example_dataset.var["highly_variable"].sum()) <= example_dataset.n_vars


def test_scale_expression(
    example_dataset: ad.AnnData,
) -> None:
    normalize_and_log_transform(example_dataset)

    scale_expression(
        example_dataset,
        max_value=5,
    )

    matrix = np.asarray(example_dataset.X)

    assert np.isfinite(matrix).all()
    assert np.abs(matrix).max() <= 5


def test_preprocess_dataset(
    example_dataset: ad.AnnData,
) -> None:
    original_counts = example_dataset.X.copy()

    processed = preprocess_dataset(
        example_dataset,
        target_sum=100,
        n_top_genes=3,
        scale_max_value=5,
    )

    assert "counts" in processed.layers
    assert processed.raw is not None
    assert "highly_variable" in processed.var.columns
    assert "preprocessing" in processed.uns

    np.testing.assert_array_equal(
        processed.layers["counts"].toarray(),
        original_counts.toarray(),
    )

    assert int(processed.var["highly_variable"].sum()) == 3
    assert processed.n_obs == example_dataset.n_obs
    assert processed.n_vars == example_dataset.n_vars


def test_preprocess_does_not_modify_original(
    example_dataset: ad.AnnData,
) -> None:
    preprocess_dataset(
        example_dataset,
        n_top_genes=3,
    )

    assert "counts" not in example_dataset.layers
    assert "highly_variable" not in example_dataset.var.columns
    assert example_dataset.raw is None


def test_subset_highly_variable_genes(
    example_dataset: ad.AnnData,
) -> None:
    processed = preprocess_dataset(
        example_dataset,
        n_top_genes=3,
        subset_highly_variable=True,
    )

    assert processed.n_vars == 3
    assert processed.var["highly_variable"].all()


def test_save_dataset(
    example_dataset: ad.AnnData,
    tmp_path: Path,
) -> None:
    processed = preprocess_dataset(
        example_dataset,
        n_top_genes=3,
    )

    output_path = tmp_path / "preprocessed.h5ad"

    save_dataset(processed, output_path)

    assert output_path.exists()

    loaded = ad.read_h5ad(output_path)

    assert loaded.n_obs == processed.n_obs
    assert loaded.n_vars == processed.n_vars
    assert "counts" in loaded.layers
