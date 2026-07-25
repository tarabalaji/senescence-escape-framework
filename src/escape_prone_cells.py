from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

DEFAULT_CONDITION_COLUMN = "condition"
DEFAULT_CELL_LINE_COLUMN = "cell_line"
DEFAULT_TRANSITION_COLUMN = "transition_probability"
DEFAULT_PSEUDOTIME_COLUMN = "escape_pseudotime"
DEFAULT_RAP_COLUMN = "repopulation_associated_potential"

DEFAULT_ESCAPE_QUANTILE = 0.90
DEFAULT_MINIMUM_CELLS = 20
DEFAULT_MAXIMUM_MARKERS = 100
DEFAULT_LOG_FOLD_CHANGE_THRESHOLD = 0.25
DEFAULT_ADJUSTED_P_VALUE_THRESHOLD = 0.05


def validate_adata(
    adata: ad.AnnData,
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
) -> None:
    if adata.n_obs == 0:
        raise ValueError("AnnData contains no cells")

    if adata.n_vars == 0:
        raise ValueError("AnnData contains no genes")

    required_columns = {
        condition_column,
        cell_line_column,
        transition_column,
    }

    missing_columns = required_columns.difference(adata.obs.columns)

    if missing_columns:
        raise ValueError(
            f"AnnData observations are missing columns: {sorted(missing_columns)}"
        )

    conditions = adata.obs[condition_column].astype(str).str.upper()

    if "TIS" not in set(conditions):
        raise ValueError("AnnData does not contain TIS cells")

    transition_values = pd.to_numeric(
        adata.obs[transition_column],
        errors="coerce",
    )

    if transition_values.isna().any():
        raise ValueError(f"{transition_column} contains missing or nonnumeric values")

    if transition_values.lt(0).any() or transition_values.gt(1).any():
        raise ValueError(f"{transition_column} must contain values between 0 and 1")


def classify_escape_prone_cells(
    adata: ad.AnnData,
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
    escape_quantile: float = DEFAULT_ESCAPE_QUANTILE,
    label_column: str = "escape_prone_status",
) -> tuple[ad.AnnData, pd.DataFrame]:
    if not 0 < escape_quantile < 1:
        raise ValueError("escape_quantile must be between zero and one")

    validate_adata(
        adata,
        condition_column=condition_column,
        cell_line_column=cell_line_column,
        transition_column=transition_column,
    )

    scored = adata.copy()

    scored.obs[label_column] = "not_tis"
    scored.obs["escape_prone_threshold"] = np.nan

    threshold_rows = []

    for cell_line in sorted(scored.obs[cell_line_column].astype(str).unique()):
        tis_mask = scored.obs[cell_line_column].astype(str).eq(cell_line) & scored.obs[
            condition_column
        ].astype(str).str.upper().eq("TIS")

        tis_probabilities = pd.to_numeric(
            scored.obs.loc[tis_mask, transition_column],
            errors="coerce",
        )

        if tis_probabilities.empty:
            continue

        threshold = float(tis_probabilities.quantile(escape_quantile))

        escape_prone_mask = tis_mask & pd.to_numeric(
            scored.obs[transition_column],
            errors="coerce",
        ).ge(threshold)

        stable_tis_mask = tis_mask & ~escape_prone_mask

        scored.obs.loc[
            escape_prone_mask,
            label_column,
        ] = "escape_prone_tis"

        scored.obs.loc[
            stable_tis_mask,
            label_column,
        ] = "stable_tis"

        scored.obs.loc[
            tis_mask,
            "escape_prone_threshold",
        ] = threshold

        threshold_rows.append(
            {
                "cell_line": cell_line,
                "escape_quantile": escape_quantile,
                "threshold": threshold,
                "tis_cells": int(tis_mask.sum()),
                "escape_prone_cells": int(escape_prone_mask.sum()),
                "escape_prone_fraction": float(
                    escape_prone_mask.sum() / tis_mask.sum()
                ),
            }
        )

    threshold_summary = pd.DataFrame(threshold_rows)

    scored.obs[label_column] = pd.Categorical(
        scored.obs[label_column],
        categories=[
            "stable_tis",
            "escape_prone_tis",
            "not_tis",
        ],
    )

    return scored, threshold_summary


def summarize_escape_prone_cells(
    adata: ad.AnnData,
    label_column: str = "escape_prone_status",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
    pseudotime_column: str = DEFAULT_PSEUDOTIME_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
) -> pd.DataFrame:
    if label_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {label_column}")

    rows = []

    for cell_line in sorted(adata.obs[cell_line_column].astype(str).unique()):
        for status in [
            "stable_tis",
            "escape_prone_tis",
        ]:
            mask = adata.obs[cell_line_column].astype(str).eq(cell_line) & adata.obs[
                label_column
            ].astype(str).eq(status)

            if not mask.any():
                continue

            row = {
                "cell_line": cell_line,
                "status": status,
                "cells": int(mask.sum()),
                "mean_transition_probability": float(
                    pd.to_numeric(
                        adata.obs.loc[
                            mask,
                            transition_column,
                        ],
                        errors="coerce",
                    ).mean()
                ),
                "median_transition_probability": float(
                    pd.to_numeric(
                        adata.obs.loc[
                            mask,
                            transition_column,
                        ],
                        errors="coerce",
                    ).median()
                ),
            }

            if pseudotime_column in adata.obs.columns:
                row["mean_escape_pseudotime"] = float(
                    pd.to_numeric(
                        adata.obs.loc[
                            mask,
                            pseudotime_column,
                        ],
                        errors="coerce",
                    ).mean()
                )

            if rap_column in adata.obs.columns:
                row["mean_rap"] = float(
                    pd.to_numeric(
                        adata.obs.loc[
                            mask,
                            rap_column,
                        ],
                        errors="coerce",
                    ).mean()
                )

            rows.append(row)

    return (
        pd.DataFrame(rows).sort_values(["cell_line", "status"]).reset_index(drop=True)
    )


def extract_obs_features(
    adata: ad.AnnData,
    prefixes: tuple[str, ...] = (
        "pathway__",
        "regulator__",
    ),
    additional_columns: tuple[str, ...] = (
        DEFAULT_RAP_COLUMN,
        "rap_score",
    ),
) -> list[str]:
    features = []

    for column in adata.obs.columns:
        if any(str(column).startswith(prefix) for prefix in prefixes):
            features.append(str(column))

    for column in additional_columns:
        if column in adata.obs.columns and column not in features:
            features.append(column)

    return features


def compare_escape_features(
    adata: ad.AnnData,
    feature_columns: list[str],
    label_column: str = "escape_prone_status",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    minimum_cells: int = DEFAULT_MINIMUM_CELLS,
) -> pd.DataFrame:
    rows = []

    for cell_line in sorted(adata.obs[cell_line_column].astype(str).unique()):
        cell_line_mask = adata.obs[cell_line_column].astype(str).eq(cell_line)

        escape_mask = cell_line_mask & adata.obs[label_column].astype(str).eq(
            "escape_prone_tis"
        )

        stable_mask = cell_line_mask & adata.obs[label_column].astype(str).eq(
            "stable_tis"
        )

        if escape_mask.sum() < minimum_cells or stable_mask.sum() < minimum_cells:
            continue

        for feature in feature_columns:
            if feature not in adata.obs.columns:
                continue

            escape_values = pd.to_numeric(
                adata.obs.loc[escape_mask, feature],
                errors="coerce",
            ).dropna()

            stable_values = pd.to_numeric(
                adata.obs.loc[stable_mask, feature],
                errors="coerce",
            ).dropna()

            if len(escape_values) < minimum_cells or len(stable_values) < minimum_cells:
                continue

            if escape_values.nunique() < 2 and stable_values.nunique() < 2:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = mannwhitneyu(
                    escape_values,
                    stable_values,
                    alternative="two-sided",
                )

            mean_difference = float(escape_values.mean() - stable_values.mean())

            rows.append(
                {
                    "cell_line": cell_line,
                    "feature": feature,
                    "escape_prone_cells": len(escape_values),
                    "stable_tis_cells": len(stable_values),
                    "escape_prone_mean": float(escape_values.mean()),
                    "stable_tis_mean": float(stable_values.mean()),
                    "mean_difference": mean_difference,
                    "absolute_mean_difference": abs(mean_difference),
                    "mann_whitney_statistic": float(statistic),
                    "p_value": float(p_value),
                    "direction": (
                        "higher_in_escape_prone"
                        if mean_difference > 0
                        else (
                            "lower_in_escape_prone"
                            if mean_difference < 0
                            else "unchanged"
                        )
                    ),
                }
            )

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    results["adjusted_p_value"] = np.nan

    for cell_line, indices in results.groupby("cell_line").groups.items():
        results.loc[
            indices,
            "adjusted_p_value",
        ] = multipletests(
            results.loc[indices, "p_value"],
            method="fdr_bh",
        )[1]

    return results.sort_values(
        [
            "adjusted_p_value",
            "absolute_mean_difference",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)


def get_dense_expression(
    adata: ad.AnnData,
) -> np.ndarray:
    if sparse.issparse(adata.X):
        return adata.X.toarray()

    return np.asarray(adata.X, dtype=float)


def calculate_gene_markers(
    adata: ad.AnnData,
    label_column: str = "escape_prone_status",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    minimum_cells: int = DEFAULT_MINIMUM_CELLS,
    log_fold_change_threshold: float = (DEFAULT_LOG_FOLD_CHANGE_THRESHOLD),
    adjusted_p_value_threshold: float = (DEFAULT_ADJUSTED_P_VALUE_THRESHOLD),
    maximum_markers: int = DEFAULT_MAXIMUM_MARKERS,
) -> pd.DataFrame:
    expression = get_dense_expression(adata)

    rows = []

    for cell_line in sorted(adata.obs[cell_line_column].astype(str).unique()):
        escape_mask = (
            adata.obs[cell_line_column].astype(str).eq(cell_line)
            & adata.obs[label_column].astype(str).eq("escape_prone_tis")
        ).to_numpy()

        stable_mask = (
            adata.obs[cell_line_column].astype(str).eq(cell_line)
            & adata.obs[label_column].astype(str).eq("stable_tis")
        ).to_numpy()

        if escape_mask.sum() < minimum_cells or stable_mask.sum() < minimum_cells:
            continue

        escape_expression = expression[escape_mask]
        stable_expression = expression[stable_mask]

        escape_means = np.mean(
            escape_expression,
            axis=0,
        )

        stable_means = np.mean(
            stable_expression,
            axis=0,
        )

        log_fold_changes = escape_means - stable_means

        cell_line_rows = []

        for gene_index, gene in enumerate(adata.var_names):
            escape_values = escape_expression[:, gene_index]

            stable_values = stable_expression[:, gene_index]

            if np.unique(escape_values).size < 2 and np.unique(stable_values).size < 2:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = mannwhitneyu(
                    escape_values,
                    stable_values,
                    alternative="two-sided",
                )

            cell_line_rows.append(
                {
                    "cell_line": cell_line,
                    "gene": str(gene),
                    "escape_prone_mean": float(escape_means[gene_index]),
                    "stable_tis_mean": float(stable_means[gene_index]),
                    "log_fold_change": float(log_fold_changes[gene_index]),
                    "absolute_log_fold_change": float(
                        abs(log_fold_changes[gene_index])
                    ),
                    "mann_whitney_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )

        cell_line_results = pd.DataFrame(cell_line_rows)

        cell_line_results["adjusted_p_value"] = multipletests(
            cell_line_results["p_value"],
            method="fdr_bh",
        )[1]

        cell_line_results["direction"] = np.where(
            cell_line_results["log_fold_change"] > 0,
            "higher_in_escape_prone",
            "lower_in_escape_prone",
        )

        significant = cell_line_results.loc[
            (cell_line_results["adjusted_p_value"] <= adjusted_p_value_threshold)
            & (
                cell_line_results["absolute_log_fold_change"]
                >= log_fold_change_threshold
            )
        ].sort_values(
            [
                "adjusted_p_value",
                "absolute_log_fold_change",
            ],
            ascending=[True, False],
        )

        rows.append(significant.head(maximum_markers))

    if not rows:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "gene",
                "escape_prone_mean",
                "stable_tis_mean",
                "log_fold_change",
                "absolute_log_fold_change",
                "mann_whitney_statistic",
                "p_value",
                "adjusted_p_value",
                "direction",
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_conserved_marker_summary(
    marker_results: pd.DataFrame,
    minimum_cell_lines: int = 2,
) -> pd.DataFrame:
    if marker_results.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "cell_lines",
                "mean_log_fold_change",
                "minimum_absolute_log_fold_change",
                "maximum_adjusted_p_value",
                "direction",
            ]
        )

    rows = []

    for gene, group in marker_results.groupby("gene"):
        if group["cell_line"].nunique() < minimum_cell_lines:
            continue

        signs = np.sign(group["log_fold_change"])

        if not (np.all(signs > 0) or np.all(signs < 0)):
            continue

        mean_log_fold_change = float(group["log_fold_change"].mean())

        rows.append(
            {
                "gene": gene,
                "cell_lines": int(group["cell_line"].nunique()),
                "mean_log_fold_change": (mean_log_fold_change),
                "minimum_absolute_log_fold_change": float(
                    group["absolute_log_fold_change"].min()
                ),
                "maximum_adjusted_p_value": float(group["adjusted_p_value"].max()),
                "direction": (
                    "higher_in_escape_prone"
                    if mean_log_fold_change > 0
                    else "lower_in_escape_prone"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "gene",
                "cell_lines",
                "mean_log_fold_change",
                "minimum_absolute_log_fold_change",
                "maximum_adjusted_p_value",
                "direction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "minimum_absolute_log_fold_change",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def create_conserved_feature_summary(
    feature_results: pd.DataFrame,
    minimum_cell_lines: int = 2,
    adjusted_p_value_threshold: float = (DEFAULT_ADJUSTED_P_VALUE_THRESHOLD),
) -> pd.DataFrame:
    if feature_results.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "cell_lines",
                "mean_difference",
                "minimum_absolute_mean_difference",
                "maximum_adjusted_p_value",
                "direction",
            ]
        )

    significant = feature_results.loc[
        feature_results["adjusted_p_value"] <= adjusted_p_value_threshold
    ]

    rows = []

    for feature, group in significant.groupby("feature"):
        if group["cell_line"].nunique() < minimum_cell_lines:
            continue

        signs = np.sign(group["mean_difference"])

        if not (np.all(signs > 0) or np.all(signs < 0)):
            continue

        mean_difference = float(group["mean_difference"].mean())

        rows.append(
            {
                "feature": feature,
                "cell_lines": int(group["cell_line"].nunique()),
                "mean_difference": mean_difference,
                "minimum_absolute_mean_difference": float(
                    group["absolute_mean_difference"].min()
                ),
                "maximum_adjusted_p_value": float(group["adjusted_p_value"].max()),
                "direction": (
                    "higher_in_escape_prone"
                    if mean_difference > 0
                    else "lower_in_escape_prone"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "cell_lines",
                "mean_difference",
                "minimum_absolute_mean_difference",
                "maximum_adjusted_p_value",
                "direction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "minimum_absolute_mean_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def run_escape_prone_analysis(
    adata: ad.AnnData,
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
    escape_quantile: float = DEFAULT_ESCAPE_QUANTILE,
    minimum_cells: int = DEFAULT_MINIMUM_CELLS,
    maximum_markers: int = DEFAULT_MAXIMUM_MARKERS,
    log_fold_change_threshold: float = (DEFAULT_LOG_FOLD_CHANGE_THRESHOLD),
    adjusted_p_value_threshold: float = (DEFAULT_ADJUSTED_P_VALUE_THRESHOLD),
) -> tuple[
    ad.AnnData,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scored, threshold_summary = classify_escape_prone_cells(
        adata,
        condition_column=condition_column,
        cell_line_column=cell_line_column,
        transition_column=transition_column,
        escape_quantile=escape_quantile,
    )

    cell_summary = summarize_escape_prone_cells(
        scored,
        cell_line_column=cell_line_column,
        transition_column=transition_column,
    )

    feature_columns = extract_obs_features(scored)

    feature_results = compare_escape_features(
        scored,
        feature_columns=feature_columns,
        cell_line_column=cell_line_column,
        minimum_cells=minimum_cells,
    )

    conserved_features = create_conserved_feature_summary(
        feature_results,
        minimum_cell_lines=min(
            2,
            scored.obs[cell_line_column].nunique(),
        ),
        adjusted_p_value_threshold=(adjusted_p_value_threshold),
    )

    gene_markers = calculate_gene_markers(
        scored,
        cell_line_column=cell_line_column,
        minimum_cells=minimum_cells,
        maximum_markers=maximum_markers,
        log_fold_change_threshold=(log_fold_change_threshold),
        adjusted_p_value_threshold=(adjusted_p_value_threshold),
    )

    conserved_markers = create_conserved_marker_summary(
        gene_markers,
        minimum_cell_lines=min(
            2,
            scored.obs[cell_line_column].nunique(),
        ),
    )

    scored.uns["escape_prone_definition"] = (
        "TIS cells whose transition probability lies at or "
        f"above the {escape_quantile:.2f} within-cell-line quantile."
    )

    scored.uns["escape_prone_feature_columns"] = feature_columns

    return (
        scored,
        threshold_summary,
        cell_summary,
        feature_results,
        conserved_features,
        gene_markers,
        conserved_markers,
    )


def save_results(
    scored: ad.AnnData,
    threshold_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    feature_results: pd.DataFrame,
    conserved_features: pd.DataFrame,
    gene_markers: pd.DataFrame,
    conserved_markers: pd.DataFrame,
    output_adata: str | Path,
    output_directory: str | Path,
) -> None:
    output_adata = Path(output_adata)
    output_directory = Path(output_directory)

    output_adata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.write_h5ad(output_adata)

    threshold_summary.to_csv(
        output_directory / "escape_prone_thresholds.csv",
        index=False,
    )

    cell_summary.to_csv(
        output_directory / "escape_prone_summary.csv",
        index=False,
    )

    feature_results.to_csv(
        output_directory / "escape_prone_feature_comparisons.csv",
        index=False,
    )

    conserved_features.to_csv(
        output_directory / "escape_prone_conserved_features.csv",
        index=False,
    )

    gene_markers.to_csv(
        output_directory / "escape_prone_gene_markers.csv",
        index=False,
    )

    conserved_markers.to_csv(
        output_directory / "escape_prone_conserved_markers.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify and characterize TIS cells with "
            "elevated senescence-escape potential."
        )
    )

    parser.add_argument(
        "--adata",
        type=Path,
        default=Path("data/processed/trajectory_scored.h5ad"),
    )

    parser.add_argument(
        "--output-adata",
        type=Path,
        default=Path("data/processed/escape_prone_scored.h5ad"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/tables"),
    )

    parser.add_argument(
        "--condition-column",
        default=DEFAULT_CONDITION_COLUMN,
    )

    parser.add_argument(
        "--cell-line-column",
        default=DEFAULT_CELL_LINE_COLUMN,
    )

    parser.add_argument(
        "--transition-column",
        default=DEFAULT_TRANSITION_COLUMN,
    )

    parser.add_argument(
        "--escape-quantile",
        type=float,
        default=DEFAULT_ESCAPE_QUANTILE,
    )

    parser.add_argument(
        "--minimum-cells",
        type=int,
        default=DEFAULT_MINIMUM_CELLS,
    )

    parser.add_argument(
        "--maximum-markers",
        type=int,
        default=DEFAULT_MAXIMUM_MARKERS,
    )

    parser.add_argument(
        "--log-fold-change-threshold",
        type=float,
        default=DEFAULT_LOG_FOLD_CHANGE_THRESHOLD,
    )

    parser.add_argument(
        "--adjusted-p-value-threshold",
        type=float,
        default=DEFAULT_ADJUSTED_P_VALUE_THRESHOLD,
    )

    args = parser.parse_args()

    if not args.adata.exists():
        raise FileNotFoundError(f"Required AnnData file does not exist: {args.adata}")

    adata = ad.read_h5ad(args.adata)

    results = run_escape_prone_analysis(
        adata,
        condition_column=args.condition_column,
        cell_line_column=args.cell_line_column,
        transition_column=args.transition_column,
        escape_quantile=args.escape_quantile,
        minimum_cells=args.minimum_cells,
        maximum_markers=args.maximum_markers,
        log_fold_change_threshold=(args.log_fold_change_threshold),
        adjusted_p_value_threshold=(args.adjusted_p_value_threshold),
    )

    save_results(
        *results,
        output_adata=args.output_adata,
        output_directory=args.output_directory,
    )

    (
        scored,
        threshold_summary,
        _,
        _,
        conserved_features,
        gene_markers,
        conserved_markers,
    ) = results

    escape_prone_count = (
        scored.obs["escape_prone_status"].astype(str).eq("escape_prone_tis").sum()
    )

    print(f"Cells analyzed: {scored.n_obs:,}")
    print(f"Escape-prone TIS cells: {escape_prone_count:,}")
    print(f"Cell lines analyzed: {len(threshold_summary)}")
    print(f"Conserved escape-prone features: {len(conserved_features)}")
    print(f"Significant gene markers: {len(gene_markers)}")
    print(f"Conserved gene markers: {len(conserved_markers)}")

    if not conserved_features.empty:
        print("Top conserved features:")

        for row in conserved_features.head(10).itertuples(index=False):
            print(
                f"{row.feature}: "
                f"difference={row.mean_difference:.3f}, "
                f"direction={row.direction}"
            )

    if not conserved_markers.empty:
        print("Top conserved gene markers:")

        for row in conserved_markers.head(10).itertuples(index=False):
            print(
                f"{row.gene}: "
                f"logFC={row.mean_log_fold_change:.3f}, "
                f"direction={row.direction}"
            )

    print(f"Saved escape-prone cell results to {args.output_directory}")


if __name__ == "__main__":
    main()
