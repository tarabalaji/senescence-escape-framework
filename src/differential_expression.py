from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

REQUIRED_OBS_COLUMNS = {
    "cell_line",
    "condition",
    "sample_id",
}

DEFAULT_COMPARISONS = (
    ("TIS", "CTR"),
    ("REPOP", "TIS"),
    ("REPOP", "CTR"),
)


def validate_dataset(dataset: ad.AnnData) -> None:
    if dataset.n_obs == 0:
        raise ValueError("Dataset contains no cells")

    if dataset.n_vars == 0:
        raise ValueError("Dataset contains no genes")

    missing_columns = REQUIRED_OBS_COLUMNS.difference(dataset.obs.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required metadata columns: {sorted(missing_columns)}"
        )

    if dataset.raw is None:
        raise ValueError("Dataset does not contain normalized expression in .raw")


def validate_comparison(
    dataset: ad.AnnData,
    group: str,
    reference: str,
) -> None:
    observed_conditions = set(dataset.obs["condition"].astype(str))

    missing_conditions = {
        group,
        reference,
    }.difference(observed_conditions)

    if missing_conditions:
        raise ValueError(
            "Comparison contains conditions not found in dataset: "
            f"{sorted(missing_conditions)}"
        )

    if group == reference:
        raise ValueError("Comparison group and reference must be different")


def prepare_cell_line_dataset(
    dataset: ad.AnnData,
    cell_line: str,
) -> ad.AnnData:
    cell_line_mask = dataset.obs["cell_line"].astype(str) == cell_line

    subset = dataset[cell_line_mask].copy()

    if subset.n_obs == 0:
        raise ValueError(f"No cells were found for cell line: {cell_line}")

    return subset


def run_comparison(
    dataset: ad.AnnData,
    group: str,
    reference: str,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    validate_comparison(
        dataset,
        group=group,
        reference=reference,
    )

    comparison_mask = dataset.obs["condition"].astype(str).isin([group, reference])

    comparison_data = dataset[comparison_mask].copy()

    group_count = int((comparison_data.obs["condition"].astype(str) == group).sum())

    reference_count = int(
        (comparison_data.obs["condition"].astype(str) == reference).sum()
    )

    if group_count < 2:
        raise ValueError(f"Condition {group} has fewer than two cells")

    if reference_count < 2:
        raise ValueError(f"Condition {reference} has fewer than two cells")

    comparison_data.obs["condition"] = (
        comparison_data.obs["condition"].astype(str).astype("category")
    )

    sc.tl.rank_genes_groups(
        comparison_data,
        groupby="condition",
        groups=[group],
        reference=reference,
        method=method,
        use_raw=True,
        pts=True,
    )

    results = sc.get.rank_genes_groups_df(
        comparison_data,
        group=group,
    )

    results = results.rename(
        columns={
            "names": "gene",
            "scores": "score",
            "logfoldchanges": "log2_fold_change",
            "pvals": "p_value",
            "pvals_adj": "adjusted_p_value",
            "pct_nz_group": "fraction_expressed_group",
            "pct_nz_reference": "fraction_expressed_reference",
        }
    )

    expected_columns = [
        "gene",
        "score",
        "log2_fold_change",
        "p_value",
        "adjusted_p_value",
        "fraction_expressed_group",
        "fraction_expressed_reference",
    ]

    for column in expected_columns:
        if column not in results.columns:
            results[column] = np.nan

    results = results[expected_columns].copy()

    results["group"] = group
    results["reference"] = reference
    results["comparison"] = f"{group}_vs_{reference}"
    results["group_cell_count"] = group_count
    results["reference_cell_count"] = reference_count

    results = results.sort_values(
        [
            "adjusted_p_value",
            "log2_fold_change",
        ],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    return results


def run_cell_line_differential_expression(
    dataset: ad.AnnData,
    cell_line: str,
    comparisons: tuple[tuple[str, str], ...] = (DEFAULT_COMPARISONS),
    method: str = "wilcoxon",
) -> pd.DataFrame:
    cell_line_data = prepare_cell_line_dataset(
        dataset,
        cell_line,
    )

    comparison_results = []

    for group, reference in comparisons:
        result = run_comparison(
            cell_line_data,
            group=group,
            reference=reference,
            method=method,
        )

        result.insert(
            0,
            "cell_line",
            cell_line,
        )

        comparison_results.append(result)

    return pd.concat(
        comparison_results,
        ignore_index=True,
    )


def run_differential_expression(
    dataset: ad.AnnData,
    comparisons: tuple[tuple[str, str], ...] = (DEFAULT_COMPARISONS),
    method: str = "wilcoxon",
) -> pd.DataFrame:
    validate_dataset(dataset)

    cell_lines = sorted(dataset.obs["cell_line"].astype(str).unique().tolist())

    if not cell_lines:
        raise ValueError("No cell lines were found")

    all_results = []

    for cell_line in cell_lines:
        result = run_cell_line_differential_expression(
            dataset,
            cell_line=cell_line,
            comparisons=comparisons,
            method=method,
        )

        all_results.append(result)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def filter_significant_genes(
    results: pd.DataFrame,
    adjusted_p_value_threshold: float = 0.05,
    minimum_absolute_log2_fold_change: float = 0.25,
) -> pd.DataFrame:
    required_columns = {
        "adjusted_p_value",
        "log2_fold_change",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f"Results are missing required columns: {sorted(missing_columns)}"
        )

    if not 0 < adjusted_p_value_threshold <= 1:
        raise ValueError("adjusted_p_value_threshold must be between zero and one")

    if minimum_absolute_log2_fold_change < 0:
        raise ValueError("minimum_absolute_log2_fold_change cannot be negative")

    significant_mask = (results["adjusted_p_value"] <= adjusted_p_value_threshold) & (
        results["log2_fold_change"].abs() >= minimum_absolute_log2_fold_change
    )

    return results[significant_mask].copy().reset_index(drop=True)


def create_gene_summary(
    significant_results: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "cell_line",
        "comparison",
        "gene",
        "log2_fold_change",
        "adjusted_p_value",
    }

    missing_columns = required_columns.difference(significant_results.columns)

    if missing_columns:
        raise ValueError(
            f"Results are missing required columns: {sorted(missing_columns)}"
        )

    if significant_results.empty:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "comparison",
                "significant_genes",
                "upregulated_genes",
                "downregulated_genes",
            ]
        )

    summary_rows = []

    grouped = significant_results.groupby(
        ["cell_line", "comparison"],
        observed=True,
    )

    for (cell_line, comparison), group in grouped:
        summary_rows.append(
            {
                "cell_line": str(cell_line),
                "comparison": str(comparison),
                "significant_genes": len(group),
                "upregulated_genes": int((group["log2_fold_change"] > 0).sum()),
                "downregulated_genes": int((group["log2_fold_change"] < 0).sum()),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .sort_values(["cell_line", "comparison"])
        .reset_index(drop=True)
    )


def save_results(
    all_results: pd.DataFrame,
    significant_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results.to_csv(
        output_directory / "differential_expression_all.csv",
        index=False,
    )

    significant_results.to_csv(
        output_directory / "differential_expression_significant.csv",
        index=False,
    )

    summary.to_csv(
        output_directory / "differential_expression_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run differential expression analysis separately for MCF7 and T47D cells."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_reduced.h5ad"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/tables"),
    )

    parser.add_argument(
        "--method",
        choices=[
            "wilcoxon",
            "t-test",
            "t-test_overestim_var",
        ],
        default="wilcoxon",
    )

    parser.add_argument(
        "--adjusted-p-value",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--minimum-log2-fold-change",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    dataset = ad.read_h5ad(args.input)

    all_results = run_differential_expression(
        dataset,
        method=args.method,
    )

    significant_results = filter_significant_genes(
        all_results,
        adjusted_p_value_threshold=args.adjusted_p_value,
        minimum_absolute_log2_fold_change=(args.minimum_log2_fold_change),
    )

    summary = create_gene_summary(significant_results)

    save_results(
        all_results,
        significant_results,
        summary,
        args.output_directory,
    )

    print(f"Total tested gene results: {len(all_results):,}")
    print(f"Significant gene results: {len(significant_results):,}")

    for row in summary.itertuples(index=False):
        print(
            f"{row.cell_line} {row.comparison}: "
            f"{row.significant_genes:,} significant genes "
            f"({row.upregulated_genes:,} up, "
            f"{row.downregulated_genes:,} down)"
        )

    print(f"Saved differential-expression results to {args.output_directory}")


if __name__ == "__main__":
    main()
