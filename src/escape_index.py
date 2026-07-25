from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_ADJUSTED_P_VALUE = 0.05
DEFAULT_MINIMUM_LOG2_FOLD_CHANGE = 0.5
DEFAULT_TOP_GENES_PER_DIRECTION = 100
DEFAULT_SCORE_COLUMN = "repopulation_associated_potential"

REQUIRED_DE_COLUMNS = {
    "cell_line",
    "comparison",
    "gene",
    "log2_fold_change",
    "adjusted_p_value",
}

REQUIRED_OBS_COLUMNS = {
    "cell_line",
    "condition",
    "sample_id",
}


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


def validate_differential_expression(
    results: pd.DataFrame,
) -> None:
    missing_columns = REQUIRED_DE_COLUMNS.difference(results.columns)

    if missing_columns:
        raise ValueError(
            "Differential-expression results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError("Differential-expression results are empty")


def select_conserved_signature(
    results: pd.DataFrame,
    comparison: str = "REPOP_vs_TIS",
    adjusted_p_value_threshold: float = (DEFAULT_ADJUSTED_P_VALUE),
    minimum_absolute_log2_fold_change: float = (DEFAULT_MINIMUM_LOG2_FOLD_CHANGE),
    top_genes_per_direction: int = (DEFAULT_TOP_GENES_PER_DIRECTION),
    minimum_cell_lines: int | None = None,
) -> pd.DataFrame:
    validate_differential_expression(results)

    if not 0 < adjusted_p_value_threshold <= 1:
        raise ValueError("adjusted_p_value_threshold must be between zero and one")

    if minimum_absolute_log2_fold_change < 0:
        raise ValueError("minimum_absolute_log2_fold_change cannot be negative")

    if top_genes_per_direction <= 0:
        raise ValueError("top_genes_per_direction must be greater than zero")

    comparison_results = results.loc[
        results["comparison"].astype(str) == comparison
    ].copy()

    if comparison_results.empty:
        raise ValueError(f"No results were found for comparison: {comparison}")

    cell_lines = sorted(comparison_results["cell_line"].astype(str).unique().tolist())

    required_cell_lines = (
        len(cell_lines) if minimum_cell_lines is None else minimum_cell_lines
    )

    if not 1 <= required_cell_lines <= len(cell_lines):
        raise ValueError(
            "minimum_cell_lines must be between one and the "
            "number of observed cell lines"
        )

    significant = comparison_results.loc[
        (comparison_results["adjusted_p_value"] <= adjusted_p_value_threshold)
        & (
            comparison_results["log2_fold_change"].abs()
            >= minimum_absolute_log2_fold_change
        )
    ].copy()

    if significant.empty:
        raise ValueError("No genes passed the signature-selection thresholds")

    significant["direction"] = np.where(
        significant["log2_fold_change"] > 0,
        "up",
        "down",
    )

    gene_summary = (
        significant.groupby(
            ["gene", "direction"],
            observed=True,
        )
        .agg(
            cell_lines_supported=(
                "cell_line",
                "nunique",
            ),
            mean_log2_fold_change=(
                "log2_fold_change",
                "mean",
            ),
            minimum_adjusted_p_value=(
                "adjusted_p_value",
                "min",
            ),
        )
        .reset_index()
    )

    gene_summary = gene_summary.loc[
        gene_summary["cell_lines_supported"] >= required_cell_lines
    ].copy()

    if gene_summary.empty:
        raise ValueError(
            "No conserved genes were supported by the required number of cell lines"
        )

    selected_frames: list[pd.DataFrame] = []

    for direction in ["up", "down"]:
        direction_results = gene_summary.loc[
            gene_summary["direction"] == direction
        ].copy()

        direction_results["absolute_mean_log2_fold_change"] = direction_results[
            "mean_log2_fold_change"
        ].abs()

        direction_results = direction_results.sort_values(
            [
                "cell_lines_supported",
                "absolute_mean_log2_fold_change",
                "minimum_adjusted_p_value",
                "gene",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        ).head(top_genes_per_direction)

        selected_frames.append(direction_results)

    signature = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    if signature.empty:
        raise ValueError("The selected signature contains no genes")

    signature["comparison"] = comparison
    signature["required_cell_lines"] = required_cell_lines

    return signature.sort_values(
        [
            "direction",
            "absolute_mean_log2_fold_change",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def get_expression_matrix(
    dataset: ad.AnnData,
    genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    validate_dataset(dataset)

    raw_gene_names = dataset.raw.var_names.astype(str)

    available_genes = [gene for gene in genes if gene in raw_gene_names]

    if not available_genes:
        raise ValueError("None of the signature genes were found in the dataset")

    expression = dataset.raw[
        :,
        available_genes,
    ].X

    if sparse.issparse(expression):
        expression = expression.toarray()

    return (
        np.asarray(expression, dtype=np.float64),
        available_genes,
    )


def standardize_genes(
    expression: np.ndarray,
) -> np.ndarray:
    if expression.ndim != 2:
        raise ValueError("Expression matrix must be two-dimensional")

    means = expression.mean(axis=0)
    standard_deviations = expression.std(
        axis=0,
        ddof=0,
    )

    standard_deviations = np.where(
        standard_deviations == 0,
        1.0,
        standard_deviations,
    )

    return (expression - means) / standard_deviations


def calculate_signature_score(
    dataset: ad.AnnData,
    up_genes: list[str],
    down_genes: list[str],
) -> np.ndarray:
    if not up_genes:
        raise ValueError("At least one upregulated signature gene is required")

    all_genes = list(dict.fromkeys(up_genes + down_genes))

    expression, available_genes = get_expression_matrix(
        dataset,
        all_genes,
    )

    standardized = standardize_genes(expression)

    gene_positions = {gene: index for index, gene in enumerate(available_genes)}

    available_up_genes = [gene for gene in up_genes if gene in gene_positions]

    available_down_genes = [gene for gene in down_genes if gene in gene_positions]

    if not available_up_genes:
        raise ValueError("No upregulated signature genes were found in the dataset")

    up_indices = [gene_positions[gene] for gene in available_up_genes]

    up_score = standardized[
        :,
        up_indices,
    ].mean(axis=1)

    if available_down_genes:
        down_indices = [gene_positions[gene] for gene in available_down_genes]

        down_score = standardized[
            :,
            down_indices,
        ].mean(axis=1)
    else:
        down_score = np.zeros(
            dataset.n_obs,
            dtype=np.float64,
        )

    return up_score - down_score


def standardize_score_within_cell_line(
    dataset: ad.AnnData,
    scores: np.ndarray,
) -> np.ndarray:
    if len(scores) != dataset.n_obs:
        raise ValueError("Score count does not match the number of cells")

    standardized_scores = np.zeros(
        dataset.n_obs,
        dtype=np.float64,
    )

    cell_lines = dataset.obs["cell_line"].astype(str).to_numpy()

    for cell_line in np.unique(cell_lines):
        mask = cell_lines == cell_line
        group_scores = scores[mask]

        mean = group_scores.mean()
        standard_deviation = group_scores.std(ddof=0)

        if standard_deviation == 0:
            standardized_scores[mask] = 0.0
        else:
            standardized_scores[mask] = (group_scores - mean) / standard_deviation

    return standardized_scores


def add_escape_index(
    dataset: ad.AnnData,
    signature: pd.DataFrame,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> ad.AnnData:
    validate_dataset(dataset)

    required_signature_columns = {
        "gene",
        "direction",
    }

    missing_columns = required_signature_columns.difference(signature.columns)

    if missing_columns:
        raise ValueError(
            f"Signature is missing required columns: {sorted(missing_columns)}"
        )

    up_genes = (
        signature.loc[
            signature["direction"] == "up",
            "gene",
        ]
        .astype(str)
        .tolist()
    )

    down_genes = (
        signature.loc[
            signature["direction"] == "down",
            "gene",
        ]
        .astype(str)
        .tolist()
    )

    raw_scores = calculate_signature_score(
        dataset,
        up_genes=up_genes,
        down_genes=down_genes,
    )

    standardized_scores = standardize_score_within_cell_line(
        dataset,
        raw_scores,
    )

    result = dataset.copy()

    result.obs[f"{score_column}_raw"] = raw_scores
    result.obs[score_column] = standardized_scores

    result.uns["escape_index"] = {
        "name": "Repopulation-Associated Potential",
        "score_column": score_column,
        "up_genes": up_genes,
        "down_genes": down_genes,
        "up_gene_count": len(up_genes),
        "down_gene_count": len(down_genes),
        "standardization": "within_cell_line_z_score",
    }

    return result


def create_score_summary(
    dataset: ad.AnnData,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    if score_column not in dataset.obs.columns:
        raise ValueError(f"Dataset does not contain score column: {score_column}")

    grouping_columns = [
        "cell_line",
        "condition",
        "sample_id",
    ]

    summary = (
        dataset.obs.groupby(
            grouping_columns,
            observed=True,
        )[score_column]
        .agg(
            cell_count="count",
            mean_score="mean",
            median_score="median",
            standard_deviation="std",
            minimum_score="min",
            maximum_score="max",
        )
        .reset_index()
    )

    return summary.sort_values(grouping_columns).reset_index(drop=True)


def save_outputs(
    dataset: ad.AnnData,
    signature: pd.DataFrame,
    summary: pd.DataFrame,
    dataset_path: str | Path,
    signature_path: str | Path,
    summary_path: str | Path,
) -> None:
    dataset_path = Path(dataset_path)
    signature_path = Path(signature_path)
    summary_path = Path(summary_path)

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.write_h5ad(dataset_path)
    signature.to_csv(
        signature_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a conserved repopulation signature and "
            "calculate Repopulation-Associated Potential."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_reduced.h5ad"),
    )

    parser.add_argument(
        "--differential-expression",
        type=Path,
        default=Path("results/tables/differential_expression_all.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/combined_scored.h5ad"),
    )

    parser.add_argument(
        "--signature-output",
        type=Path,
        default=Path("results/tables/repopulation_signature.csv"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/tables/escape_index_summary.csv"),
    )

    parser.add_argument(
        "--adjusted-p-value",
        type=float,
        default=DEFAULT_ADJUSTED_P_VALUE,
    )

    parser.add_argument(
        "--minimum-log2-fold-change",
        type=float,
        default=DEFAULT_MINIMUM_LOG2_FOLD_CHANGE,
    )

    parser.add_argument(
        "--top-genes-per-direction",
        type=int,
        default=DEFAULT_TOP_GENES_PER_DIRECTION,
    )

    parser.add_argument(
        "--minimum-cell-lines",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    if not args.differential_expression.exists():
        raise FileNotFoundError(
            "Differential-expression file does not exist: "
            f"{args.differential_expression}"
        )

    dataset = ad.read_h5ad(args.input)

    differential_expression = pd.read_csv(args.differential_expression)

    signature = select_conserved_signature(
        differential_expression,
        adjusted_p_value_threshold=(args.adjusted_p_value),
        minimum_absolute_log2_fold_change=(args.minimum_log2_fold_change),
        top_genes_per_direction=(args.top_genes_per_direction),
        minimum_cell_lines=args.minimum_cell_lines,
    )

    scored_dataset = add_escape_index(
        dataset,
        signature,
    )

    summary = create_score_summary(scored_dataset)

    save_outputs(
        scored_dataset,
        signature,
        summary,
        args.output,
        args.signature_output,
        args.summary_output,
    )

    up_gene_count = int((signature["direction"] == "up").sum())

    down_gene_count = int((signature["direction"] == "down").sum())

    print(f"Conserved upregulated genes: {up_gene_count:,}")
    print(f"Conserved downregulated genes: {down_gene_count:,}")

    print("Mean RAP score by cell line and condition:")

    condition_summary = scored_dataset.obs.groupby(
        ["cell_line", "condition"],
        observed=True,
    )[DEFAULT_SCORE_COLUMN].mean()

    for (
        cell_line,
        condition,
    ), score in condition_summary.items():
        print(f"{cell_line} {condition}: {score:.3f}")

    print(f"Saved scored dataset to {args.output}")
    print(f"Saved signature to {args.signature_output}")
    print(f"Saved score summary to {args.summary_output}")


if __name__ == "__main__":
    main()
