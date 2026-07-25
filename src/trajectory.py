from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

DEFAULT_CONDITION_COLUMN = "condition"
DEFAULT_CELL_LINE_COLUMN = "cell_line"
DEFAULT_TRANSITION_COLUMN = "transition_probability"
DEFAULT_RAP_COLUMN = "repopulation_associated_potential"

DEFAULT_NUMBER_OF_BINS = 10
DEFAULT_MINIMUM_CELLS_PER_BIN = 20
DEFAULT_MAXIMUM_GENES = 50
DEFAULT_MINIMUM_ABSOLUTE_CORRELATION = 0.20
DEFAULT_RANDOM_STATE = 42


def validate_trajectory_adata(
    adata: ad.AnnData,
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
) -> None:
    if adata.n_obs == 0:
        raise ValueError("AnnData object contains no cells")

    if adata.n_vars == 0:
        raise ValueError("AnnData object contains no genes")

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

    transition_values = pd.to_numeric(
        adata.obs[transition_column],
        errors="coerce",
    )

    if transition_values.isna().any():
        raise ValueError(f"{transition_column} contains missing or nonnumeric values")

    if transition_values.lt(0).any() or transition_values.gt(1).any():
        raise ValueError(f"{transition_column} must contain values between 0 and 1")

    conditions = set(adata.obs[condition_column].astype(str).str.upper())

    required_conditions = {
        "TIS",
        "REPOP",
    }

    missing_conditions = required_conditions.difference(conditions)

    if missing_conditions:
        raise ValueError(
            "AnnData object is missing required conditions: "
            f"{sorted(missing_conditions)}"
        )

    cell_lines = adata.obs[cell_line_column].astype(str).dropna().unique()

    if len(cell_lines) < 1:
        raise ValueError("At least one cell line is required")


def normalize_gene_names(
    genes: Iterable[str],
) -> list[str]:
    normalized = []

    for gene in genes:
        gene_name = str(gene).strip().upper()

        if gene_name and gene_name not in normalized:
            normalized.append(gene_name)

    return normalized


def sanitize_feature_name(
    feature_name: str,
) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            str(feature_name),
        )
        .strip("_")
        .lower()
    )


def get_expression_matrix(
    adata: ad.AnnData,
    genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    gene_lookup = {str(gene).upper(): str(gene) for gene in adata.var_names}

    available_genes = [
        gene_lookup[gene] for gene in normalize_gene_names(genes) if gene in gene_lookup
    ]

    if not available_genes:
        return (
            np.empty((adata.n_obs, 0)),
            [],
        )

    matrix = adata[:, available_genes].X

    if sparse.issparse(matrix):
        matrix = matrix.toarray()

    return (
        np.asarray(
            matrix,
            dtype=float,
        ),
        available_genes,
    )


def create_escape_pseudotime(
    adata: ad.AnnData,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
    pseudotime_column: str = "escape_pseudotime",
) -> ad.AnnData:
    if transition_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {transition_column}")

    scored = adata.copy()

    transition_values = pd.to_numeric(
        scored.obs[transition_column],
        errors="coerce",
    )

    if transition_values.isna().any():
        raise ValueError(f"{transition_column} contains invalid values")

    minimum_value = float(transition_values.min())

    maximum_value = float(transition_values.max())

    if np.isclose(
        minimum_value,
        maximum_value,
    ):
        raise ValueError("Transition probabilities contain no variation")

    scored.obs[pseudotime_column] = (transition_values - minimum_value) / (
        maximum_value - minimum_value
    )

    return scored


def assign_trajectory_bins(
    adata: ad.AnnData,
    pseudotime_column: str = "escape_pseudotime",
    number_of_bins: int = DEFAULT_NUMBER_OF_BINS,
    bin_column: str = "trajectory_bin",
) -> ad.AnnData:
    if number_of_bins < 3:
        raise ValueError("number_of_bins must be at least three")

    if pseudotime_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {pseudotime_column}")

    binned = adata.copy()

    pseudotime = pd.to_numeric(
        binned.obs[pseudotime_column],
        errors="coerce",
    )

    if pseudotime.isna().any():
        raise ValueError(f"{pseudotime_column} contains invalid values")

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_labels = np.arange(
        1,
        number_of_bins + 1,
    )

    binned.obs[bin_column] = pd.cut(
        pseudotime,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        ordered=True,
    )

    if binned.obs[bin_column].isna().any():
        raise ValueError("Some cells could not be assigned to trajectory bins")

    binned.obs["trajectory_stage"] = (
        binned.obs[bin_column]
        .astype(int)
        .map(
            lambda bin_number: classify_trajectory_stage(
                bin_number,
                number_of_bins,
            )
        )
    )

    return binned


def classify_trajectory_stage(
    bin_number: int,
    number_of_bins: int,
) -> str:
    relative_position = (bin_number - 0.5) / number_of_bins

    if relative_position < 1 / 3:
        return "early"

    if relative_position < 2 / 3:
        return "intermediate"

    return "late"


def extract_transition_features(
    adata: ad.AnnData,
    feature_prefixes: tuple[str, ...] = (
        "pathway__",
        "regulator__",
    ),
    additional_columns: tuple[str, ...] = (DEFAULT_RAP_COLUMN,),
) -> pd.DataFrame:
    features = pd.DataFrame(index=adata.obs_names)

    for column in adata.obs.columns:
        if any(str(column).startswith(prefix) for prefix in feature_prefixes):
            features[str(column)] = pd.to_numeric(
                adata.obs[column],
                errors="coerce",
            )

    for column in additional_columns:
        if column in adata.obs.columns:
            features[column] = pd.to_numeric(
                adata.obs[column],
                errors="coerce",
            )

    return features


def load_transition_feature_table(
    feature_table_path: str | Path,
    adata: ad.AnnData,
) -> pd.DataFrame:
    feature_table_path = Path(feature_table_path)

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Transition feature table does not exist: {feature_table_path}"
        )

    table = pd.read_csv(feature_table_path)

    if "cell_id" not in table.columns:
        raise ValueError("Transition feature table must contain cell_id")

    table["cell_id"] = table["cell_id"].astype(str)

    table = table.set_index("cell_id")

    duplicate_cells = table.index.duplicated(keep=False)

    if duplicate_cells.any():
        raise ValueError("Transition feature table contains duplicate cell IDs")

    feature_columns = [
        column
        for column in table.columns
        if (
            column.startswith("pathway__")
            or column.startswith("regulator__")
            or column == "rap_score"
        )
    ]

    features = table[feature_columns].copy()

    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.reindex(adata.obs_names.astype(str))

    return features


def add_features_to_adata(
    adata: ad.AnnData,
    features: pd.DataFrame,
) -> ad.AnnData:
    enriched = adata.copy()

    aligned_features = features.reindex(enriched.obs_names.astype(str))

    for column in aligned_features.columns:
        enriched.obs[column] = aligned_features[column].to_numpy()

    return enriched


def summarize_trajectory_bins(
    adata: ad.AnnData,
    feature_columns: list[str],
    pseudotime_column: str = "escape_pseudotime",
    bin_column: str = "trajectory_bin",
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    minimum_cells_per_bin: int = DEFAULT_MINIMUM_CELLS_PER_BIN,
) -> pd.DataFrame:
    if minimum_cells_per_bin <= 0:
        raise ValueError("minimum_cells_per_bin must be greater than zero")

    required_columns = {
        pseudotime_column,
        bin_column,
        condition_column,
        cell_line_column,
    }

    missing_columns = required_columns.difference(adata.obs.columns)

    if missing_columns:
        raise ValueError(
            "AnnData observations are missing trajectory columns: "
            f"{sorted(missing_columns)}"
        )

    valid_features = [
        feature for feature in feature_columns if feature in adata.obs.columns
    ]

    rows = []

    grouping_columns = [
        cell_line_column,
        bin_column,
    ]

    for group_values, group in adata.obs.groupby(
        grouping_columns,
        observed=True,
    ):
        cell_line, trajectory_bin = group_values

        cell_count = len(group)

        if cell_count < minimum_cells_per_bin:
            continue

        base_row = {
            "cell_line": str(cell_line),
            "trajectory_bin": int(trajectory_bin),
            "cells": int(cell_count),
            "mean_pseudotime": float(
                pd.to_numeric(
                    group[pseudotime_column],
                    errors="coerce",
                ).mean()
            ),
            "tis_fraction": float(
                group[condition_column].astype(str).str.upper().eq("TIS").mean()
            ),
            "repop_fraction": float(
                group[condition_column].astype(str).str.upper().eq("REPOP").mean()
            ),
            "ctr_fraction": float(
                group[condition_column].astype(str).str.upper().eq("CTR").mean()
            ),
        }

        for feature in valid_features:
            values = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            base_row[f"mean__{feature}"] = float(values.mean())

            base_row[f"median__{feature}"] = float(values.median())

        rows.append(base_row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "cell_line",
                "trajectory_bin",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_feature_trajectory_correlations(
    adata: ad.AnnData,
    feature_columns: list[str],
    pseudotime_column: str = "escape_pseudotime",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    minimum_cells: int = 20,
) -> pd.DataFrame:
    if minimum_cells < 3:
        raise ValueError("minimum_cells must be at least three")

    rows = []

    cell_lines = sorted(adata.obs[cell_line_column].astype(str).unique())

    for cell_line in cell_lines:
        mask = adata.obs[cell_line_column].astype(str) == cell_line

        pseudotime = pd.to_numeric(
            adata.obs.loc[
                mask,
                pseudotime_column,
            ],
            errors="coerce",
        )

        for feature in feature_columns:
            if feature not in adata.obs.columns:
                continue

            values = pd.to_numeric(
                adata.obs.loc[
                    mask,
                    feature,
                ],
                errors="coerce",
            )

            valid_mask = pseudotime.notna() & values.notna()

            valid_pseudotime = pseudotime.loc[valid_mask]

            valid_values = values.loc[valid_mask]

            if len(valid_values) < minimum_cells:
                continue

            if valid_values.nunique() < 2 or valid_pseudotime.nunique() < 2:
                correlation = 0.0
                p_value = 1.0
            else:
                correlation, p_value = spearmanr(
                    valid_pseudotime,
                    valid_values,
                )

            rows.append(
                {
                    "cell_line": cell_line,
                    "feature": feature,
                    "cells": len(valid_values),
                    "spearman_correlation": float(correlation),
                    "absolute_correlation": float(abs(correlation)),
                    "p_value": float(p_value),
                    "direction": (
                        "increasing"
                        if correlation > 0
                        else ("decreasing" if correlation < 0 else "stable")
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "absolute_correlation",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def create_conserved_feature_summary(
    correlation_results: pd.DataFrame,
    minimum_absolute_correlation: float = (DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
    minimum_cell_lines: int = 2,
) -> pd.DataFrame:
    required_columns = {
        "cell_line",
        "feature",
        "spearman_correlation",
        "absolute_correlation",
        "p_value",
    }

    missing_columns = required_columns.difference(correlation_results.columns)

    if missing_columns:
        raise ValueError(
            f"Correlation results are missing columns: {sorted(missing_columns)}"
        )

    if not 0 <= minimum_absolute_correlation <= 1:
        raise ValueError("minimum_absolute_correlation must be between zero and one")

    if minimum_cell_lines <= 0:
        raise ValueError("minimum_cell_lines must be greater than zero")

    rows = []

    for feature, group in correlation_results.groupby("feature"):
        qualifying = group.loc[
            group["absolute_correlation"] >= minimum_absolute_correlation
        ].copy()

        if len(qualifying) < minimum_cell_lines:
            continue

        signs = np.sign(qualifying["spearman_correlation"])

        if not (np.all(signs > 0) or np.all(signs < 0)):
            continue

        mean_correlation = float(qualifying["spearman_correlation"].mean())

        rows.append(
            {
                "feature": feature,
                "cell_lines": int(qualifying["cell_line"].nunique()),
                "mean_spearman_correlation": (mean_correlation),
                "minimum_absolute_correlation": float(
                    qualifying["absolute_correlation"].min()
                ),
                "maximum_p_value": float(qualifying["p_value"].max()),
                "direction": ("increasing" if mean_correlation > 0 else "decreasing"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "cell_lines",
                "mean_spearman_correlation",
                "minimum_absolute_correlation",
                "maximum_p_value",
                "direction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "minimum_absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_variable_genes(
    adata: ad.AnnData,
    maximum_genes: int = DEFAULT_MAXIMUM_GENES,
    excluded_genes: Iterable[str] = (),
) -> list[str]:
    if maximum_genes <= 0:
        raise ValueError("maximum_genes must be greater than zero")

    matrix = adata.X

    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()

        squared_means = np.asarray(matrix.power(2).mean(axis=0)).ravel()

        variances = squared_means - means**2
    else:
        matrix_array = np.asarray(
            matrix,
            dtype=float,
        )

        variances = np.nanvar(
            matrix_array,
            axis=0,
        )

    excluded = set(normalize_gene_names(excluded_genes))

    gene_variance = pd.DataFrame(
        {
            "gene": [str(gene) for gene in adata.var_names],
            "variance": variances,
        }
    )

    gene_variance["gene_upper"] = gene_variance["gene"].str.upper()

    gene_variance = gene_variance.loc[~gene_variance["gene_upper"].isin(excluded)]

    gene_variance = gene_variance.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=["variance"])

    return (
        gene_variance.sort_values(
            "variance",
            ascending=False,
        )
        .head(maximum_genes)["gene"]
        .tolist()
    )


def calculate_gene_trajectory_correlations(
    adata: ad.AnnData,
    genes: list[str],
    pseudotime_column: str = "escape_pseudotime",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    minimum_cells: int = 20,
) -> pd.DataFrame:
    matrix, available_genes = get_expression_matrix(
        adata,
        genes,
    )

    if len(available_genes) == 0:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "gene",
                "cells",
                "spearman_correlation",
                "absolute_correlation",
                "p_value",
                "direction",
            ]
        )

    expression = pd.DataFrame(
        matrix,
        index=adata.obs_names,
        columns=available_genes,
    )

    rows = []

    for cell_line in sorted(adata.obs[cell_line_column].astype(str).unique()):
        mask = adata.obs[cell_line_column].astype(str) == cell_line

        pseudotime = pd.to_numeric(
            adata.obs.loc[
                mask,
                pseudotime_column,
            ],
            errors="coerce",
        )

        for gene in available_genes:
            values = pd.to_numeric(
                expression.loc[
                    mask,
                    gene,
                ],
                errors="coerce",
            )

            valid_mask = pseudotime.notna() & values.notna()

            valid_pseudotime = pseudotime.loc[valid_mask]

            valid_values = values.loc[valid_mask]

            if len(valid_values) < minimum_cells:
                continue

            if valid_values.nunique() < 2 or valid_pseudotime.nunique() < 2:
                correlation = 0.0
                p_value = 1.0
            else:
                correlation, p_value = spearmanr(
                    valid_pseudotime,
                    valid_values,
                )

            rows.append(
                {
                    "cell_line": cell_line,
                    "gene": str(gene),
                    "cells": len(valid_values),
                    "spearman_correlation": float(correlation),
                    "absolute_correlation": float(abs(correlation)),
                    "p_value": float(p_value),
                    "direction": (
                        "increasing"
                        if correlation > 0
                        else ("decreasing" if correlation < 0 else "stable")
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "absolute_correlation",
                "gene",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def create_conserved_gene_summary(
    gene_correlations: pd.DataFrame,
    minimum_absolute_correlation: float = (DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
    minimum_cell_lines: int = 2,
) -> pd.DataFrame:
    if gene_correlations.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "cell_lines",
                "mean_spearman_correlation",
                "minimum_absolute_correlation",
                "maximum_p_value",
                "direction",
            ]
        )

    renamed = gene_correlations.rename(
        columns={
            "gene": "feature",
        }
    )

    conserved = create_conserved_feature_summary(
        renamed,
        minimum_absolute_correlation=(minimum_absolute_correlation),
        minimum_cell_lines=minimum_cell_lines,
    )

    return conserved.rename(
        columns={
            "feature": "gene",
        }
    )


def calculate_stage_markers(
    adata: ad.AnnData,
    feature_columns: list[str],
    stage_column: str = "trajectory_stage",
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
) -> pd.DataFrame:
    required_stages = [
        "early",
        "intermediate",
        "late",
    ]

    rows = []

    for cell_line in sorted(adata.obs[cell_line_column].astype(str).unique()):
        cell_line_data = adata.obs.loc[
            adata.obs[cell_line_column].astype(str) == cell_line
        ]

        for feature in feature_columns:
            if feature not in cell_line_data.columns:
                continue

            stage_means = {}

            for stage in required_stages:
                values = pd.to_numeric(
                    cell_line_data.loc[
                        cell_line_data[stage_column] == stage,
                        feature,
                    ],
                    errors="coerce",
                )

                stage_means[stage] = float(values.mean())

            if any(np.isnan(value) for value in stage_means.values()):
                continue

            largest_stage = max(
                stage_means,
                key=stage_means.get,
            )

            smallest_stage = min(
                stage_means,
                key=stage_means.get,
            )

            rows.append(
                {
                    "cell_line": cell_line,
                    "feature": feature,
                    "early_mean": stage_means["early"],
                    "intermediate_mean": stage_means["intermediate"],
                    "late_mean": stage_means["late"],
                    "peak_stage": largest_stage,
                    "minimum_stage": smallest_stage,
                    "dynamic_range": float(
                        max(stage_means.values()) - min(stage_means.values())
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "dynamic_range",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def create_condition_distribution_summary(
    adata: ad.AnnData,
    pseudotime_column: str = "escape_pseudotime",
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
) -> pd.DataFrame:
    summary_data = adata.obs[
        [
            cell_line_column,
            condition_column,
            pseudotime_column,
        ]
    ].copy()

    summary_data[pseudotime_column] = pd.to_numeric(
        summary_data[pseudotime_column],
        errors="coerce",
    )

    return (
        summary_data.groupby(
            [
                cell_line_column,
                condition_column,
            ],
            observed=True,
        )
        .agg(
            cells=(
                pseudotime_column,
                "size",
            ),
            mean_pseudotime=(
                pseudotime_column,
                "mean",
            ),
            median_pseudotime=(
                pseudotime_column,
                "median",
            ),
            standard_deviation=(
                pseudotime_column,
                "std",
            ),
            minimum_pseudotime=(
                pseudotime_column,
                "min",
            ),
            maximum_pseudotime=(
                pseudotime_column,
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                cell_line_column,
                condition_column,
            ]
        )
        .reset_index(drop=True)
    )


def run_trajectory_analysis(
    adata: ad.AnnData,
    transition_features: pd.DataFrame | None = None,
    condition_column: str = DEFAULT_CONDITION_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    transition_column: str = DEFAULT_TRANSITION_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
    number_of_bins: int = DEFAULT_NUMBER_OF_BINS,
    minimum_cells_per_bin: int = (DEFAULT_MINIMUM_CELLS_PER_BIN),
    maximum_genes: int = DEFAULT_MAXIMUM_GENES,
    minimum_absolute_correlation: float = (DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
) -> tuple[
    ad.AnnData,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_trajectory_adata(
        adata,
        condition_column=condition_column,
        cell_line_column=cell_line_column,
        transition_column=transition_column,
    )

    scored = create_escape_pseudotime(
        adata,
        transition_column=transition_column,
    )

    scored = assign_trajectory_bins(
        scored,
        number_of_bins=number_of_bins,
    )

    if transition_features is not None:
        scored = add_features_to_adata(
            scored,
            transition_features,
        )

    feature_table = extract_transition_features(
        scored,
        additional_columns=(
            rap_column,
            "rap_score",
        ),
    )

    feature_columns = list(feature_table.columns)

    trajectory_bins = summarize_trajectory_bins(
        scored,
        feature_columns=feature_columns,
        condition_column=condition_column,
        cell_line_column=cell_line_column,
        minimum_cells_per_bin=minimum_cells_per_bin,
    )

    feature_correlations = calculate_feature_trajectory_correlations(
        scored,
        feature_columns=feature_columns,
        cell_line_column=cell_line_column,
    )

    conserved_features = create_conserved_feature_summary(
        feature_correlations,
        minimum_absolute_correlation=(minimum_absolute_correlation),
        minimum_cell_lines=min(
            2,
            scored.obs[cell_line_column].nunique(),
        ),
    )

    variable_genes = select_variable_genes(
        scored,
        maximum_genes=maximum_genes,
    )

    gene_correlations = calculate_gene_trajectory_correlations(
        scored,
        genes=variable_genes,
        cell_line_column=cell_line_column,
    )

    conserved_genes = create_conserved_gene_summary(
        gene_correlations,
        minimum_absolute_correlation=(minimum_absolute_correlation),
        minimum_cell_lines=min(
            2,
            scored.obs[cell_line_column].nunique(),
        ),
    )

    stage_markers = calculate_stage_markers(
        scored,
        feature_columns=feature_columns,
        cell_line_column=cell_line_column,
    )

    condition_summary = create_condition_distribution_summary(
        scored,
        condition_column=condition_column,
        cell_line_column=cell_line_column,
    )

    scored.uns["trajectory_description"] = (
        "Inferred TIS-to-REPOP escape pseudotime derived "
        "from transition-model probability. This represents "
        "a computational continuum rather than direct lineage tracing."
    )

    scored.uns["trajectory_number_of_bins"] = number_of_bins

    scored.uns["trajectory_feature_columns"] = feature_columns

    scored.uns["trajectory_variable_genes"] = variable_genes

    return (
        scored,
        trajectory_bins,
        feature_correlations,
        conserved_features,
        gene_correlations,
        conserved_genes,
        stage_markers,
        condition_summary,
    )


def save_results(
    scored_adata: ad.AnnData,
    trajectory_bins: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    conserved_features: pd.DataFrame,
    gene_correlations: pd.DataFrame,
    conserved_genes: pd.DataFrame,
    stage_markers: pd.DataFrame,
    condition_summary: pd.DataFrame,
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

    scored_adata.write_h5ad(output_adata)

    trajectory_bins.to_csv(
        output_directory / "trajectory_bin_summary.csv",
        index=False,
    )

    feature_correlations.to_csv(
        output_directory / "trajectory_feature_correlations.csv",
        index=False,
    )

    conserved_features.to_csv(
        output_directory / "trajectory_conserved_features.csv",
        index=False,
    )

    gene_correlations.to_csv(
        output_directory / "trajectory_gene_correlations.csv",
        index=False,
    )

    conserved_genes.to_csv(
        output_directory / "trajectory_conserved_genes.csv",
        index=False,
    )

    stage_markers.to_csv(
        output_directory / "trajectory_stage_markers.csv",
        index=False,
    )

    condition_summary.to_csv(
        output_directory / "trajectory_condition_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer and characterize the TIS-to-REPOP senescence-escape continuum."
        )
    )

    parser.add_argument(
        "--adata",
        type=Path,
        default=Path("data/processed/transition_scored.h5ad"),
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path("results/tables/transition_model_features.csv"),
    )

    parser.add_argument(
        "--output-adata",
        type=Path,
        default=Path("data/processed/trajectory_scored.h5ad"),
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
        "--rap-column",
        default=DEFAULT_RAP_COLUMN,
    )

    parser.add_argument(
        "--number-of-bins",
        type=int,
        default=DEFAULT_NUMBER_OF_BINS,
    )

    parser.add_argument(
        "--minimum-cells-per-bin",
        type=int,
        default=DEFAULT_MINIMUM_CELLS_PER_BIN,
    )

    parser.add_argument(
        "--maximum-genes",
        type=int,
        default=DEFAULT_MAXIMUM_GENES,
    )

    parser.add_argument(
        "--minimum-absolute-correlation",
        type=float,
        default=(DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
    )

    args = parser.parse_args()

    if not args.adata.exists():
        raise FileNotFoundError(f"Required AnnData file does not exist: {args.adata}")

    adata = ad.read_h5ad(args.adata)

    transition_features = None

    if args.features.exists():
        transition_features = load_transition_feature_table(
            args.features,
            adata,
        )

    (
        scored,
        trajectory_bins,
        feature_correlations,
        conserved_features,
        gene_correlations,
        conserved_genes,
        stage_markers,
        condition_summary,
    ) = run_trajectory_analysis(
        adata,
        transition_features=transition_features,
        condition_column=args.condition_column,
        cell_line_column=args.cell_line_column,
        transition_column=args.transition_column,
        rap_column=args.rap_column,
        number_of_bins=args.number_of_bins,
        minimum_cells_per_bin=(args.minimum_cells_per_bin),
        maximum_genes=args.maximum_genes,
        minimum_absolute_correlation=(args.minimum_absolute_correlation),
    )

    save_results(
        scored,
        trajectory_bins,
        feature_correlations,
        conserved_features,
        gene_correlations,
        conserved_genes,
        stage_markers,
        condition_summary,
        args.output_adata,
        args.output_directory,
    )

    print(f"Cells analyzed: {scored.n_obs:,}")

    print(f"Trajectory bins: {scored.uns['trajectory_number_of_bins']}")

    print(
        f"Transition features analyzed: {len(scored.uns['trajectory_feature_columns'])}"
    )

    print(f"Variable genes analyzed: {len(scored.uns['trajectory_variable_genes'])}")

    print(f"Conserved trajectory features: {len(conserved_features)}")

    print(f"Conserved trajectory genes: {len(conserved_genes)}")

    if not conserved_features.empty:
        print("Top conserved trajectory features:")

        for row in conserved_features.head(10).itertuples(index=False):
            print(
                f"{row.feature}: "
                f"rho={row.mean_spearman_correlation:.3f}, "
                f"direction={row.direction}"
            )

    if not conserved_genes.empty:
        print("Top conserved trajectory genes:")

        for row in conserved_genes.head(10).itertuples(index=False):
            print(
                f"{row.gene}: "
                f"rho={row.mean_spearman_correlation:.3f}, "
                f"direction={row.direction}"
            )

    print(f"Saved trajectory results to {args.output_directory}")


if __name__ == "__main__":
    main()
