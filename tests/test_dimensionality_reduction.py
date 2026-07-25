from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.dimensionality_reduction import (
    create_analysis_matrix,
    run_dimensionality_reduction,
    run_neighbors,
    run_pca,
    run_umap,
    save_dataset,
    transfer_results,
    validate_dataset,
)


@pytest.fixture
def example_dataset() -> ad.AnnData:
    random_generator = np.random.default_rng(42)

    expression = random_generator.normal(
        loc=2.0,
        scale=1.0,
        size=(30, 10),
    ).astype(np.float32)

    expression = np.maximum(expression, 0)

    obs = pd.DataFrame(
        {
            "sample_id": [f"SAMPLE_{index % 6}" for index in range(30)],
            "cell_line": ["MCF7" if index < 15 else "T47D" for index in range(30)],
            "condition": [["CTR", "TIS", "REPOP"][index % 3] for index in range(30)],
            "replicate": [str(index % 2 + 1) for index in range(30)],
        },
        index=[f"CELL_{index}" for index in range(30)],
    )

    var = pd.DataFrame(
        {
            "highly_variable": [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ]
        },
        index=[f"GENE_{index}" for index in range(10)],
    )

    return ad.AnnData(
        X=expression,
        obs=obs,
        var=var,
    )


def test_validate_dataset_accepts_valid_data(
    example_dataset: ad.AnnData,
) -> None:
    validate_dataset(example_dataset)


def test_validate_dataset_rejects_missing_hvg_column() -> None:
    dataset = ad.AnnData(
        X=np.ones((5, 4)),
        var=pd.DataFrame(index=[f"GENE_{index}" for index in range(4)]),
    )

    with pytest.raises(
        ValueError,
        match="highly variable gene annotations",
    ):
        validate_dataset(dataset)


def test_validate_dataset_rejects_no_selected_genes(
    example_dataset: ad.AnnData,
) -> None:
    example_dataset.var["highly_variable"] = False

    with pytest.raises(
        ValueError,
        match="No highly variable genes",
    ):
        validate_dataset(example_dataset)


def test_create_analysis_matrix(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(
        example_dataset,
        scale_max_value=5,
    )

    assert analysis_data.n_obs == 30
    assert analysis_data.n_vars == 5
    assert np.isfinite(analysis_data.X).all()
    assert np.abs(analysis_data.X).max() <= 5


def test_create_analysis_matrix_does_not_modify_original(
    example_dataset: ad.AnnData,
) -> None:
    original_expression = example_dataset.X.copy()

    create_analysis_matrix(example_dataset)

    np.testing.assert_array_equal(
        example_dataset.X,
        original_expression,
    )


def test_run_pca(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)

    run_pca(
        analysis_data,
        n_components=3,
    )

    assert "X_pca" in analysis_data.obsm
    assert analysis_data.obsm["X_pca"].shape == (30, 3)
    assert "PCs" in analysis_data.varm
    assert "pca" in analysis_data.uns


def test_run_pca_caps_component_count(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)

    run_pca(
        analysis_data,
        n_components=100,
    )

    assert analysis_data.obsm["X_pca"].shape[1] == 4


def test_run_neighbors_requires_pca(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)

    with pytest.raises(
        ValueError,
        match="PCA must be calculated",
    ):
        run_neighbors(analysis_data)


def test_run_neighbors(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)
    run_neighbors(analysis_data, n_neighbors=5)

    assert "neighbors" in analysis_data.uns
    assert "distances" in analysis_data.obsp
    assert "connectivities" in analysis_data.obsp


def test_run_umap_requires_neighbors(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)

    with pytest.raises(
        ValueError,
        match="Neighbors must be calculated",
    ):
        run_umap(analysis_data)


def test_run_umap(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)
    run_neighbors(analysis_data, n_neighbors=5)
    run_umap(analysis_data, random_state=42)

    assert "X_umap" in analysis_data.obsm
    assert analysis_data.obsm["X_umap"].shape == (30, 2)


def test_transfer_results(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)
    run_neighbors(analysis_data, n_neighbors=5)
    run_umap(analysis_data, random_state=42)

    result = transfer_results(
        example_dataset,
        analysis_data,
    )

    assert result.n_vars == example_dataset.n_vars
    assert "X_pca" in result.obsm
    assert "X_umap" in result.obsm
    assert "distances" in result.obsp
    assert "connectivities" in result.obsp
    assert result.varm["PCs"].shape == (10, 3)

    non_hvg_loadings = result.varm["PCs"][~result.var["highly_variable"].to_numpy()]

    assert np.isnan(non_hvg_loadings).all()


def test_run_dimensionality_reduction(
    example_dataset: ad.AnnData,
) -> None:
    result = run_dimensionality_reduction(
        example_dataset,
        n_components=3,
        n_neighbors=5,
        min_dist=0.5,
        random_state=42,
    )

    assert result.n_obs == example_dataset.n_obs
    assert result.n_vars == example_dataset.n_vars
    assert result.obsm["X_pca"].shape == (30, 3)
    assert result.obsm["X_umap"].shape == (30, 2)

    assert result.uns["dimensionality_reduction"]["highly_variable_genes_used"] == 5


def test_invalid_component_count_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        run_pca(
            analysis_data,
            n_components=0,
        )


def test_invalid_neighbor_count_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)

    with pytest.raises(
        ValueError,
        match="greater than one",
    ):
        run_neighbors(
            analysis_data,
            n_neighbors=1,
        )


def test_invalid_min_dist_raises_error(
    example_dataset: ad.AnnData,
) -> None:
    analysis_data = create_analysis_matrix(example_dataset)
    run_pca(analysis_data, n_components=3)
    run_neighbors(analysis_data, n_neighbors=5)

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        run_umap(
            analysis_data,
            min_dist=1.5,
        )


def test_save_dataset(
    example_dataset: ad.AnnData,
    tmp_path: Path,
) -> None:
    result = run_dimensionality_reduction(
        example_dataset,
        n_components=3,
        n_neighbors=5,
        random_state=42,
    )

    output_path = tmp_path / "combined_reduced.h5ad"

    save_dataset(result, output_path)

    assert output_path.exists()

    loaded = ad.read_h5ad(output_path)

    assert "X_pca" in loaded.obsm
    assert "X_umap" in loaded.obsm
