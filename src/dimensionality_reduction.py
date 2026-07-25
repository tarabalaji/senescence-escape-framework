from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

DEFAULT_N_COMPONENTS = 50
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.5
DEFAULT_RANDOM_STATE = 42
DEFAULT_SCALE_MAX_VALUE = 10.0


def validate_dataset(dataset: ad.AnnData) -> None:
    if dataset.n_obs == 0:
        raise ValueError("Dataset contains no cells")

    if dataset.n_vars == 0:
        raise ValueError("Dataset contains no genes")

    if "highly_variable" not in dataset.var.columns:
        raise ValueError("Dataset does not contain highly variable gene annotations")

    if not dataset.var["highly_variable"].any():
        raise ValueError("No highly variable genes were selected")


def create_analysis_matrix(
    dataset: ad.AnnData,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
) -> ad.AnnData:
    if scale_max_value <= 0:
        raise ValueError("scale_max_value must be greater than zero")

    validate_dataset(dataset)

    highly_variable_mask = (
        dataset.var["highly_variable"].fillna(False).astype(bool).to_numpy()
    )

    analysis_data = dataset[:, highly_variable_mask].copy()

    sc.pp.scale(
        analysis_data,
        zero_center=True,
        max_value=scale_max_value,
    )

    return analysis_data


def run_pca(
    analysis_data: ad.AnnData,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    if n_components <= 0:
        raise ValueError("n_components must be greater than zero")

    maximum_components = min(
        analysis_data.n_obs - 1,
        analysis_data.n_vars - 1,
    )

    if maximum_components < 1:
        raise ValueError("Dataset is too small to perform PCA")

    components_to_use = min(
        n_components,
        maximum_components,
    )

    sc.tl.pca(
        analysis_data,
        n_comps=components_to_use,
        svd_solver="arpack",
        random_state=random_state,
    )

    analysis_data.uns["pca_parameters"] = {
        "requested_n_components": n_components,
        "used_n_components": components_to_use,
        "random_state": random_state,
    }


def run_neighbors(
    analysis_data: ad.AnnData,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> None:
    if "X_pca" not in analysis_data.obsm:
        raise ValueError("PCA must be calculated before computing neighbors")

    if n_neighbors <= 1:
        raise ValueError("n_neighbors must be greater than one")

    if analysis_data.n_obs < 3:
        raise ValueError("At least three cells are required for neighbors")

    neighbors_to_use = min(
        n_neighbors,
        analysis_data.n_obs - 1,
    )

    pca_components = analysis_data.obsm["X_pca"].shape[1]

    sc.pp.neighbors(
        analysis_data,
        n_neighbors=neighbors_to_use,
        n_pcs=pca_components,
    )

    analysis_data.uns["neighbor_parameters"] = {
        "requested_n_neighbors": n_neighbors,
        "used_n_neighbors": neighbors_to_use,
        "n_pcs": pca_components,
    }


def run_umap(
    analysis_data: ad.AnnData,
    min_dist: float = DEFAULT_MIN_DIST,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    if "neighbors" not in analysis_data.uns:
        raise ValueError("Neighbors must be calculated before UMAP")

    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must be between zero and one")

    sc.tl.umap(
        analysis_data,
        min_dist=min_dist,
        random_state=random_state,
    )

    analysis_data.uns["umap_parameters"] = {
        "min_dist": min_dist,
        "random_state": random_state,
    }


def transfer_results(
    dataset: ad.AnnData,
    analysis_data: ad.AnnData,
) -> ad.AnnData:
    if not dataset.obs_names.equals(analysis_data.obs_names):
        raise ValueError("Cell identifiers do not match between datasets")

    if "X_pca" not in analysis_data.obsm:
        raise ValueError("Analysis data does not contain PCA results")

    if "X_umap" not in analysis_data.obsm:
        raise ValueError("Analysis data does not contain UMAP results")

    result = dataset.copy()

    result.obsm["X_pca"] = analysis_data.obsm["X_pca"].copy()
    result.obsm["X_umap"] = analysis_data.obsm["X_umap"].copy()

    result.uns["pca"] = analysis_data.uns["pca"].copy()
    result.uns["neighbors"] = analysis_data.uns["neighbors"].copy()
    result.uns["umap"] = analysis_data.uns["umap"].copy()

    result.uns["pca_parameters"] = analysis_data.uns["pca_parameters"].copy()

    result.uns["neighbor_parameters"] = analysis_data.uns["neighbor_parameters"].copy()

    result.uns["umap_parameters"] = analysis_data.uns["umap_parameters"].copy()

    result.obsp["distances"] = analysis_data.obsp["distances"].copy()

    result.obsp["connectivities"] = analysis_data.obsp["connectivities"].copy()

    result.varm["PCs"] = np.full(
        (result.n_vars, analysis_data.varm["PCs"].shape[1]),
        np.nan,
        dtype=np.float32,
    )

    highly_variable_mask = (
        result.var["highly_variable"].fillna(False).astype(bool).to_numpy()
    )

    result.varm["PCs"][highly_variable_mask] = analysis_data.varm["PCs"]

    result.uns["dimensionality_reduction"] = {
        "highly_variable_genes_used": analysis_data.n_vars,
        "cells_used": analysis_data.n_obs,
    }

    return result


def run_dimensionality_reduction(
    dataset: ad.AnnData,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    random_state: int = DEFAULT_RANDOM_STATE,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
) -> ad.AnnData:
    analysis_data = create_analysis_matrix(
        dataset,
        scale_max_value=scale_max_value,
    )

    run_pca(
        analysis_data,
        n_components=n_components,
        random_state=random_state,
    )

    run_neighbors(
        analysis_data,
        n_neighbors=n_neighbors,
    )

    run_umap(
        analysis_data,
        min_dist=min_dist,
        random_state=random_state,
    )

    return transfer_results(
        dataset,
        analysis_data,
    )


def save_dataset(
    dataset: ad.AnnData,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_h5ad(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA, nearest-neighbor graph construction, "
            "and UMAP on the preprocessed GSE280381 dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_preprocessed.h5ad"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/combined_reduced.h5ad"),
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=DEFAULT_MIN_DIST,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    dataset = ad.read_h5ad(args.input)

    reduced = run_dimensionality_reduction(
        dataset,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    save_dataset(reduced, args.output)

    print(f"Cells: {reduced.n_obs:,}")
    print(f"Genes: {reduced.n_vars:,}")
    print(f"PCA components: {reduced.obsm['X_pca'].shape[1]}")
    print(f"UMAP dimensions: {reduced.obsm['X_umap'].shape[1]}")
    print(f"Saved reduced dataset to {args.output}")


if __name__ == "__main__":
    main()
