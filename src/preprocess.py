from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

DEFAULT_TARGET_SUM = 10_000
DEFAULT_N_TOP_GENES = 2_000
DEFAULT_SCALE_MAX_VALUE = 10.0


def validate_dataset(dataset: ad.AnnData) -> None:
    if dataset.n_obs == 0:
        raise ValueError("Dataset contains no cells")

    if dataset.n_vars == 0:
        raise ValueError("Dataset contains no genes")

    if dataset.X is None:
        raise ValueError("Dataset does not contain an expression matrix")

    if np.any(
        np.asarray(dataset.X.data if sparse.issparse(dataset.X) else dataset.X) < 0
    ):
        raise ValueError("Expression matrix contains negative values")


def preserve_raw_counts(
    dataset: ad.AnnData,
    layer_name: str = "counts",
) -> None:
    if layer_name in dataset.layers:
        raise ValueError(f"Layer already exists and would be overwritten: {layer_name}")

    dataset.layers[layer_name] = dataset.X.copy()


def normalize_and_log_transform(
    dataset: ad.AnnData,
    target_sum: float = DEFAULT_TARGET_SUM,
) -> None:
    if target_sum <= 0:
        raise ValueError("target_sum must be greater than zero")

    sc.pp.normalize_total(
        dataset,
        target_sum=target_sum,
    )

    sc.pp.log1p(dataset)

    dataset.uns["normalization"] = {
        "method": "total-count normalization",
        "target_sum": target_sum,
        "log_transform": "log1p",
    }


def select_highly_variable_genes(
    dataset: ad.AnnData,
    n_top_genes: int = DEFAULT_N_TOP_GENES,
) -> None:
    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be greater than zero")

    genes_to_select = min(n_top_genes, dataset.n_vars)

    sc.pp.highly_variable_genes(
        dataset,
        n_top_genes=genes_to_select,
        flavor="seurat",
        inplace=True,
    )

    scores = (
        dataset.var["dispersions_norm"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-np.inf)
    )

    selected_genes = scores.nlargest(
        genes_to_select,
        keep="first",
    ).index

    dataset.var["highly_variable"] = dataset.var_names.isin(selected_genes)

    dataset.uns["highly_variable_gene_selection"] = {
        "method": "seurat",
        "requested_n_top_genes": n_top_genes,
        "selected_n_top_genes": int(dataset.var["highly_variable"].sum()),
    }


def scale_expression(
    dataset: ad.AnnData,
    max_value: float = DEFAULT_SCALE_MAX_VALUE,
) -> None:
    if max_value <= 0:
        raise ValueError("max_value must be greater than zero")

    sc.pp.scale(
        dataset,
        zero_center=True,
        max_value=max_value,
    )

    dataset.uns["scaling"] = {
        "zero_center": True,
        "max_value": max_value,
    }


def preprocess_dataset(
    dataset: ad.AnnData,
    target_sum: float = DEFAULT_TARGET_SUM,
    n_top_genes: int = DEFAULT_N_TOP_GENES,
    scale_max_value: float = DEFAULT_SCALE_MAX_VALUE,
    subset_highly_variable: bool = False,
) -> ad.AnnData:
    validate_dataset(dataset)

    processed = dataset.copy()

    preserve_raw_counts(processed)

    normalize_and_log_transform(
        processed,
        target_sum=target_sum,
    )

    processed.raw = processed.copy()

    select_highly_variable_genes(
        processed,
        n_top_genes=n_top_genes,
    )

    if subset_highly_variable:
        processed = processed[
            :,
            processed.var["highly_variable"],
        ].copy()

        scale_expression(
            processed,
            max_value=scale_max_value,
        )
    else:
        processed.uns["scaling"] = {
            "applied": False,
            "reason": (
                "Scaling skipped to avoid densifying the full gene-expression matrix"
            ),
            "max_value": scale_max_value,
        }

    processed.uns["preprocessing"] = {
        "target_sum": target_sum,
        "n_top_genes": n_top_genes,
        "scale_max_value": scale_max_value,
        "subset_highly_variable": subset_highly_variable,
        "cells": processed.n_obs,
        "genes": processed.n_vars,
    }

    return processed


def save_dataset(
    dataset: ad.AnnData,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_h5ad(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the quality-controlled GSE280381 dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_qc.h5ad"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/combined_preprocessed.h5ad"),
    )

    parser.add_argument(
        "--target-sum",
        type=float,
        default=DEFAULT_TARGET_SUM,
    )

    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=DEFAULT_N_TOP_GENES,
    )

    parser.add_argument(
        "--scale-max-value",
        type=float,
        default=DEFAULT_SCALE_MAX_VALUE,
    )

    parser.add_argument(
        "--subset-highly-variable",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    dataset = ad.read_h5ad(args.input)

    processed = preprocess_dataset(
        dataset,
        target_sum=args.target_sum,
        n_top_genes=args.n_top_genes,
        scale_max_value=args.scale_max_value,
        subset_highly_variable=args.subset_highly_variable,
    )

    save_dataset(processed, args.output)

    highly_variable_count = int(processed.var["highly_variable"].sum())

    print(f"Cells: {processed.n_obs:,}")
    print(f"Genes: {processed.n_vars:,}")
    print(f"Highly variable genes: {highly_variable_count:,}")
    print(f"Saved preprocessed dataset to {args.output}")


if __name__ == "__main__":
    main()
