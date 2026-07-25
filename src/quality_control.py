from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class QCThresholds:
    min_genes: int = 500
    min_counts: int = 1000
    max_counts: int = 30000
    max_mito_pct: float = 10.0


DEFAULT_THRESHOLDS = {
    "MCF7": QCThresholds(),
    "T47D": QCThresholds(),
}

REQUIRED_OBS_COLUMNS = {
    "sample_id",
    "cell_line",
    "condition",
    "replicate",
}


def validate_dataset(dataset: ad.AnnData) -> None:
    missing_columns = REQUIRED_OBS_COLUMNS.difference(dataset.obs.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required metadata columns: {sorted(missing_columns)}"
        )

    if dataset.n_obs == 0:
        raise ValueError("Dataset contains no cells")

    if dataset.n_vars == 0:
        raise ValueError("Dataset contains no genes")


def identify_mitochondrial_genes(
    dataset: ad.AnnData,
) -> np.ndarray:
    gene_names = dataset.var_names.astype(str)

    mitochondrial_mask = np.asarray(
        gene_names.str.upper().str.startswith("MT-"),
        dtype=bool,
    )

    if not mitochondrial_mask.any() and "gene_symbol" in dataset.var.columns:
        gene_symbols = dataset.var["gene_symbol"].astype(str)

        mitochondrial_mask = np.asarray(
            gene_symbols.str.upper().str.startswith("MT-"),
            dtype=bool,
        )

    return mitochondrial_mask


def row_sums(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel()

    return np.asarray(matrix).sum(axis=1)


def detected_gene_counts(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray((matrix > 0).sum(axis=1)).ravel()

    return np.count_nonzero(np.asarray(matrix) > 0, axis=1)


def calculate_qc_metrics(
    dataset: ad.AnnData,
    copy: bool = True,
) -> ad.AnnData:
    validate_dataset(dataset)

    qc_data = dataset.copy() if copy else dataset

    total_counts = row_sums(qc_data.X)
    genes_detected = detected_gene_counts(qc_data.X)

    mitochondrial_mask = identify_mitochondrial_genes(qc_data)
    qc_data.var["is_mitochondrial"] = mitochondrial_mask

    if mitochondrial_mask.any():
        mitochondrial_counts = row_sums(qc_data.X[:, mitochondrial_mask])
    else:
        mitochondrial_counts = np.zeros(
            qc_data.n_obs,
            dtype=float,
        )

    mitochondrial_pct = (
        np.divide(
            mitochondrial_counts,
            total_counts,
            out=np.zeros_like(
                mitochondrial_counts,
                dtype=float,
            ),
            where=total_counts > 0,
        )
        * 100
    )

    qc_data.obs["total_counts"] = total_counts
    qc_data.obs["n_genes_by_counts"] = genes_detected
    qc_data.obs["mitochondrial_counts"] = mitochondrial_counts
    qc_data.obs["pct_counts_mito"] = mitochondrial_pct

    return qc_data


def validate_thresholds(
    thresholds: dict[str, QCThresholds],
    observed_cell_lines: set[str],
) -> None:
    missing_cell_lines = observed_cell_lines.difference(thresholds)

    if missing_cell_lines:
        raise ValueError(
            "No quality-control thresholds were supplied for: "
            f"{sorted(missing_cell_lines)}"
        )

    for cell_line, values in thresholds.items():
        if values.min_genes < 0:
            raise ValueError(f"{cell_line} min_genes cannot be negative")

        if values.min_counts < 0:
            raise ValueError(f"{cell_line} min_counts cannot be negative")

        if values.max_counts <= values.min_counts:
            raise ValueError(f"{cell_line} max_counts must be greater than min_counts")

        if not 0 <= values.max_mito_pct <= 100:
            raise ValueError(f"{cell_line} max_mito_pct must be between 0 and 100")


def add_qc_flags(
    dataset: ad.AnnData,
    thresholds: dict[str, QCThresholds] | None = None,
    copy: bool = True,
) -> ad.AnnData:
    thresholds = thresholds or DEFAULT_THRESHOLDS

    required_metrics = {
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mito",
    }

    if not required_metrics.issubset(dataset.obs.columns):
        qc_data = calculate_qc_metrics(dataset, copy=copy)
    else:
        qc_data = dataset.copy() if copy else dataset

    observed_cell_lines = set(qc_data.obs["cell_line"].astype(str))

    validate_thresholds(thresholds, observed_cell_lines)

    qc_data.obs["fails_min_genes"] = False
    qc_data.obs["fails_min_counts"] = False
    qc_data.obs["fails_max_counts"] = False
    qc_data.obs["fails_mito"] = False

    for cell_line in sorted(observed_cell_lines):
        cell_line_mask = qc_data.obs["cell_line"].astype(str) == cell_line
        values = thresholds[cell_line]

        qc_data.obs.loc[
            cell_line_mask,
            "fails_min_genes",
        ] = (
            qc_data.obs.loc[
                cell_line_mask,
                "n_genes_by_counts",
            ]
            < values.min_genes
        )

        qc_data.obs.loc[
            cell_line_mask,
            "fails_min_counts",
        ] = (
            qc_data.obs.loc[
                cell_line_mask,
                "total_counts",
            ]
            < values.min_counts
        )

        qc_data.obs.loc[
            cell_line_mask,
            "fails_max_counts",
        ] = (
            qc_data.obs.loc[
                cell_line_mask,
                "total_counts",
            ]
            > values.max_counts
        )

        qc_data.obs.loc[
            cell_line_mask,
            "fails_mito",
        ] = (
            qc_data.obs.loc[
                cell_line_mask,
                "pct_counts_mito",
            ]
            > values.max_mito_pct
        )

    failure_columns = [
        "fails_min_genes",
        "fails_min_counts",
        "fails_max_counts",
        "fails_mito",
    ]

    qc_data.obs["passes_qc"] = ~qc_data.obs[failure_columns].any(axis=1)

    qc_data.uns["qc_thresholds"] = {
        cell_line: asdict(values)
        for cell_line, values in thresholds.items()
        if cell_line in observed_cell_lines
    }

    return qc_data


def filter_cells(
    dataset: ad.AnnData,
    thresholds: dict[str, QCThresholds] | None = None,
) -> ad.AnnData:
    flagged_data = add_qc_flags(
        dataset,
        thresholds=thresholds,
        copy=True,
    )

    filtered_data = flagged_data[flagged_data.obs["passes_qc"]].copy()

    filtered_data.uns["cells_before_qc"] = flagged_data.n_obs
    filtered_data.uns["cells_after_qc"] = filtered_data.n_obs
    filtered_data.uns["cells_removed_qc"] = flagged_data.n_obs - filtered_data.n_obs

    return filtered_data


def create_qc_summary(dataset: ad.AnnData) -> pd.DataFrame:
    required_columns = {
        "cell_line",
        "condition",
        "sample_id",
        "passes_qc",
        "fails_min_genes",
        "fails_min_counts",
        "fails_max_counts",
        "fails_mito",
    }

    missing_columns = required_columns.difference(dataset.obs.columns)

    if missing_columns:
        raise ValueError(
            "QC flags must be calculated before creating a summary. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    summary_rows: list[dict[str, object]] = []

    grouped = dataset.obs.groupby(
        ["cell_line", "condition", "sample_id"],
        observed=True,
    )

    for (
        cell_line,
        condition,
        sample_id,
    ), group in grouped:
        cells_before = len(group)
        cells_after = int(group["passes_qc"].sum())

        summary_rows.append(
            {
                "cell_line": str(cell_line),
                "condition": str(condition),
                "sample_id": str(sample_id),
                "cells_before": cells_before,
                "cells_after": cells_after,
                "cells_removed": cells_before - cells_after,
                "retention_pct": (
                    cells_after / cells_before * 100 if cells_before else 0.0
                ),
                "failed_min_genes": int(group["fails_min_genes"].sum()),
                "failed_min_counts": int(group["fails_min_counts"].sum()),
                "failed_max_counts": int(group["fails_max_counts"].sum()),
                "failed_mito": int(group["fails_mito"].sum()),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .sort_values(["cell_line", "condition", "sample_id"])
        .reset_index(drop=True)
    )


def save_qc_outputs(
    filtered_data: ad.AnnData,
    summary: pd.DataFrame,
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_data.write_h5ad(output_path)
    summary.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform quality control on GSE280381."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_raw.h5ad"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/combined_qc.h5ad"),
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/qc_summary.csv"),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    dataset = ad.read_h5ad(args.input)

    flagged_data = add_qc_flags(dataset)
    summary = create_qc_summary(flagged_data)

    filtered_data = flagged_data[flagged_data.obs["passes_qc"]].copy()

    filtered_data.uns["cells_before_qc"] = flagged_data.n_obs
    filtered_data.uns["cells_after_qc"] = filtered_data.n_obs
    filtered_data.uns["cells_removed_qc"] = flagged_data.n_obs - filtered_data.n_obs

    save_qc_outputs(
        filtered_data,
        summary,
        args.output,
        args.summary,
    )

    print(f"Cells before QC: {flagged_data.n_obs:,}")
    print(f"Cells after QC: {filtered_data.n_obs:,}")
    print(f"Cells removed: {flagged_data.n_obs - filtered_data.n_obs:,}")
    print(f"Saved filtered dataset to {args.output}")
    print(f"Saved QC summary to {args.summary}")


if __name__ == "__main__":
    main()
