from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

DEFAULT_GENE_SETS = (
    "MSigDB_Hallmark_2020",
    "Reactome_2022",
)

DEFAULT_MIN_GENE_SET_SIZE = 10
DEFAULT_MAX_GENE_SET_SIZE = 500
DEFAULT_PERMUTATIONS = 1_000
DEFAULT_THREADS = 4
DEFAULT_SEED = 42

REQUIRED_COLUMNS = {
    "cell_line",
    "comparison",
    "gene",
    "score",
    "log2_fold_change",
    "adjusted_p_value",
}


def validate_differential_expression(
    results: pd.DataFrame,
) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(results.columns)

    if missing_columns:
        raise ValueError(
            "Differential-expression results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError("Differential-expression results are empty")


def validate_parameters(
    minimum_gene_set_size: int,
    maximum_gene_set_size: int,
    permutations: int,
    threads: int,
) -> None:
    if minimum_gene_set_size <= 0:
        raise ValueError("minimum_gene_set_size must be greater than zero")

    if maximum_gene_set_size < minimum_gene_set_size:
        raise ValueError(
            "maximum_gene_set_size must be greater than or "
            "equal to minimum_gene_set_size"
        )

    if permutations <= 0:
        raise ValueError("permutations must be greater than zero")

    if threads <= 0:
        raise ValueError("threads must be greater than zero")


def create_ranking_metric(
    results: pd.DataFrame,
) -> pd.DataFrame:
    validate_differential_expression(results)

    ranking = results[
        [
            "gene",
            "score",
            "log2_fold_change",
            "adjusted_p_value",
        ]
    ].copy()

    ranking["gene"] = ranking["gene"].astype(str).str.upper().str.strip()

    ranking = ranking.loc[ranking["gene"].ne("") & ranking["gene"].ne("NAN")].copy()

    ranking["score"] = pd.to_numeric(
        ranking["score"],
        errors="coerce",
    )

    ranking["log2_fold_change"] = pd.to_numeric(
        ranking["log2_fold_change"],
        errors="coerce",
    )

    ranking["adjusted_p_value"] = pd.to_numeric(
        ranking["adjusted_p_value"],
        errors="coerce",
    )

    ranking = ranking.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    ranking = ranking.dropna(
        subset=[
            "gene",
            "score",
            "log2_fold_change",
        ]
    )

    if ranking.empty:
        raise ValueError("No valid genes remained for pathway ranking")

    ranking["ranking_metric"] = ranking["score"]

    zero_score_mask = ranking["ranking_metric"].abs() < 1e-12

    ranking.loc[
        zero_score_mask,
        "ranking_metric",
    ] = ranking.loc[
        zero_score_mask,
        "log2_fold_change",
    ]

    ranking = ranking.sort_values(
        [
            "gene",
            "ranking_metric",
            "log2_fold_change",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    ranking = ranking.loc[
        ranking.groupby("gene")["ranking_metric"].apply(
            lambda values: values.abs().idxmax()
        )
    ].copy()

    ranking = ranking.reset_index(drop=True)

    if ranking["ranking_metric"].nunique() < 2:
        raise ValueError("Ranking metric does not contain enough variation")

    unique_scores = np.sort(ranking["ranking_metric"].unique())

    score_differences = np.diff(unique_scores)
    positive_differences = score_differences[score_differences > 0]

    if len(positive_differences) > 0:
        smallest_difference = positive_differences.min()

        epsilon = smallest_difference / (2 * (len(ranking) + 1))
    else:
        epsilon = 1e-12

    tie_order = (
        ranking.sort_values(
            [
                "ranking_metric",
                "log2_fold_change",
                "gene",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .groupby(
            "ranking_metric",
            observed=True,
        )
        .cumcount()
    )

    ranking["ranking_metric"] = ranking["ranking_metric"] + epsilon * tie_order

    ranking = (
        ranking[
            [
                "gene",
                "ranking_metric",
            ]
        ]
        .sort_values(
            "ranking_metric",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return ranking


def normalize_gsea_results(
    results: pd.DataFrame,
    cell_line: str,
    comparison: str,
    gene_set_library: str,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "comparison",
                "gene_set_library",
                "pathway",
                "enrichment_score",
                "normalized_enrichment_score",
                "nominal_p_value",
                "false_discovery_rate",
                "family_wise_error_rate",
                "gene_set_size",
                "matched_genes",
                "leading_edge_genes",
                "direction",
            ]
        )

    column_aliases = {
        "Term": "pathway",
        "ES": "enrichment_score",
        "NES": "normalized_enrichment_score",
        "NOM p-val": "nominal_p_value",
        "FDR q-val": "false_discovery_rate",
        "FWER p-val": "family_wise_error_rate",
        "Tag %": "gene_set_size",
        "Gene %": "matched_genes",
        "Lead_genes": "leading_edge_genes",
    }

    normalized = results.rename(columns=column_aliases).copy()

    required_output_columns = [
        "pathway",
        "enrichment_score",
        "normalized_enrichment_score",
        "nominal_p_value",
        "false_discovery_rate",
        "family_wise_error_rate",
        "gene_set_size",
        "matched_genes",
        "leading_edge_genes",
    ]

    for column in required_output_columns:
        if column not in normalized.columns:
            normalized[column] = np.nan

    normalized = normalized[required_output_columns].copy()

    numeric_columns = [
        "enrichment_score",
        "normalized_enrichment_score",
        "nominal_p_value",
        "false_discovery_rate",
        "family_wise_error_rate",
    ]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized.insert(
        0,
        "gene_set_library",
        gene_set_library,
    )

    normalized.insert(
        0,
        "comparison",
        comparison,
    )

    normalized.insert(
        0,
        "cell_line",
        cell_line,
    )

    normalized["direction"] = np.where(
        normalized["normalized_enrichment_score"] >= 0,
        "positive",
        "negative",
    )

    return normalized.sort_values(
        [
            "false_discovery_rate",
            "normalized_enrichment_score",
        ],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def run_preranked_gsea(
    ranking: pd.DataFrame,
    gene_set_library: str,
    cell_line: str,
    comparison: str,
    minimum_gene_set_size: int = (DEFAULT_MIN_GENE_SET_SIZE),
    maximum_gene_set_size: int = (DEFAULT_MAX_GENE_SET_SIZE),
    permutations: int = DEFAULT_PERMUTATIONS,
    threads: int = DEFAULT_THREADS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    validate_parameters(
        minimum_gene_set_size,
        maximum_gene_set_size,
        permutations,
        threads,
    )

    required_ranking_columns = {
        "gene",
        "ranking_metric",
    }

    missing_columns = required_ranking_columns.difference(ranking.columns)

    if missing_columns:
        raise ValueError(f"Ranking table is missing columns: {sorted(missing_columns)}")

    gsea_input = ranking[["gene", "ranking_metric"]].copy()

    result = gp.prerank(
        rnk=gsea_input,
        gene_sets=gene_set_library,
        min_size=minimum_gene_set_size,
        max_size=maximum_gene_set_size,
        permutation_num=permutations,
        threads=threads,
        seed=seed,
        outdir=None,
        verbose=False,
    )

    result_table = getattr(
        result,
        "res2d",
        None,
    )

    if result_table is None:
        raise RuntimeError("GSEApy did not return a pathway result table")

    return normalize_gsea_results(
        result_table,
        cell_line=cell_line,
        comparison=comparison,
        gene_set_library=gene_set_library,
    )


def run_pathway_analysis(
    differential_expression: pd.DataFrame,
    gene_set_libraries: tuple[str, ...] = (DEFAULT_GENE_SETS),
    minimum_gene_set_size: int = (DEFAULT_MIN_GENE_SET_SIZE),
    maximum_gene_set_size: int = (DEFAULT_MAX_GENE_SET_SIZE),
    permutations: int = DEFAULT_PERMUTATIONS,
    threads: int = DEFAULT_THREADS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    validate_differential_expression(differential_expression)

    if not gene_set_libraries:
        raise ValueError("At least one gene-set library is required")

    all_results: list[pd.DataFrame] = []

    grouped = differential_expression.groupby(
        [
            "cell_line",
            "comparison",
        ],
        observed=True,
    )

    for (
        cell_line,
        comparison,
    ), comparison_results in grouped:
        ranking = create_ranking_metric(comparison_results)

        for gene_set_library in gene_set_libraries:
            print(f"Running {cell_line} {comparison} with {gene_set_library}...")

            pathway_results = run_preranked_gsea(
                ranking,
                gene_set_library=gene_set_library,
                cell_line=str(cell_line),
                comparison=str(comparison),
                minimum_gene_set_size=(minimum_gene_set_size),
                maximum_gene_set_size=(maximum_gene_set_size),
                permutations=permutations,
                threads=threads,
                seed=seed,
            )

            all_results.append(pathway_results)

    if not all_results:
        raise ValueError("No pathway analyses were completed")

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def filter_significant_pathways(
    results: pd.DataFrame,
    maximum_false_discovery_rate: float = 0.05,
    minimum_absolute_nes: float = 1.0,
) -> pd.DataFrame:
    required_columns = {
        "false_discovery_rate",
        "normalized_enrichment_score",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f"Pathway results are missing columns: {sorted(missing_columns)}"
        )

    if not 0 < maximum_false_discovery_rate <= 1:
        raise ValueError("maximum_false_discovery_rate must be between zero and one")

    if minimum_absolute_nes < 0:
        raise ValueError("minimum_absolute_nes cannot be negative")

    significant = results.loc[
        (results["false_discovery_rate"] <= maximum_false_discovery_rate)
        & (results["normalized_enrichment_score"].abs() >= minimum_absolute_nes)
    ].copy()

    return significant.sort_values(
        [
            "cell_line",
            "comparison",
            "gene_set_library",
            "false_discovery_rate",
        ]
    ).reset_index(drop=True)


def create_conserved_pathway_summary(
    significant_results: pd.DataFrame,
    required_cell_lines: int = 2,
) -> pd.DataFrame:
    if required_cell_lines <= 0:
        raise ValueError("required_cell_lines must be greater than zero")

    required_columns = {
        "cell_line",
        "comparison",
        "gene_set_library",
        "pathway",
        "normalized_enrichment_score",
        "false_discovery_rate",
        "direction",
    }

    missing_columns = required_columns.difference(significant_results.columns)

    if missing_columns:
        raise ValueError(
            "Significant pathway results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if significant_results.empty:
        return pd.DataFrame(
            columns=[
                "comparison",
                "gene_set_library",
                "pathway",
                "direction",
                "cell_lines_supported",
                "mean_normalized_enrichment_score",
                "minimum_absolute_nes",
                "maximum_false_discovery_rate",
            ]
        )

    grouped = (
        significant_results.groupby(
            [
                "comparison",
                "gene_set_library",
                "pathway",
                "direction",
            ],
            observed=True,
        )
        .agg(
            cell_lines_supported=(
                "cell_line",
                "nunique",
            ),
            mean_normalized_enrichment_score=(
                "normalized_enrichment_score",
                "mean",
            ),
            minimum_absolute_nes=(
                "normalized_enrichment_score",
                lambda values: values.abs().min(),
            ),
            maximum_false_discovery_rate=(
                "false_discovery_rate",
                "max",
            ),
        )
        .reset_index()
    )

    conserved = grouped.loc[
        grouped["cell_lines_supported"] >= required_cell_lines
    ].copy()

    return conserved.sort_values(
        [
            "comparison",
            "cell_lines_supported",
            "minimum_absolute_nes",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def create_pathway_summary(
    significant_results: pd.DataFrame,
) -> pd.DataFrame:
    if significant_results.empty:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "comparison",
                "gene_set_library",
                "significant_pathways",
                "positively_enriched",
                "negatively_enriched",
            ]
        )

    summary = (
        significant_results.groupby(
            [
                "cell_line",
                "comparison",
                "gene_set_library",
            ],
            observed=True,
        )
        .agg(
            significant_pathways=(
                "pathway",
                "nunique",
            ),
            positively_enriched=(
                "direction",
                lambda values: (values == "positive").sum(),
            ),
            negatively_enriched=(
                "direction",
                lambda values: (values == "negative").sum(),
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "cell_line",
            "comparison",
            "gene_set_library",
        ]
    ).reset_index(drop=True)


def save_results(
    all_results: pd.DataFrame,
    significant_results: pd.DataFrame,
    conserved_pathways: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results.to_csv(
        output_directory / "pathway_results_all.csv",
        index=False,
    )

    significant_results.to_csv(
        output_directory / "pathway_results_significant.csv",
        index=False,
    )

    conserved_pathways.to_csv(
        output_directory / "conserved_pathways.csv",
        index=False,
    )

    summary.to_csv(
        output_directory / "pathway_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run preranked pathway enrichment for each "
            "cell-line and condition comparison."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tables/differential_expression_all.csv"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/pathways"),
    )

    parser.add_argument(
        "--gene-sets",
        nargs="+",
        default=list(DEFAULT_GENE_SETS),
    )

    parser.add_argument(
        "--minimum-size",
        type=int,
        default=DEFAULT_MIN_GENE_SET_SIZE,
    )

    parser.add_argument(
        "--maximum-size",
        type=int,
        default=DEFAULT_MAX_GENE_SET_SIZE,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--maximum-fdr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--minimum-absolute-nes",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Differential-expression file does not exist: {args.input}"
        )

    differential_expression = pd.read_csv(args.input)

    all_results = run_pathway_analysis(
        differential_expression,
        gene_set_libraries=tuple(args.gene_sets),
        minimum_gene_set_size=args.minimum_size,
        maximum_gene_set_size=args.maximum_size,
        permutations=args.permutations,
        threads=args.threads,
        seed=args.seed,
    )

    significant_results = filter_significant_pathways(
        all_results,
        maximum_false_discovery_rate=(args.maximum_fdr),
        minimum_absolute_nes=(args.minimum_absolute_nes),
    )

    conserved_pathways = create_conserved_pathway_summary(
        significant_results,
        required_cell_lines=2,
    )

    summary = create_pathway_summary(significant_results)

    save_results(
        all_results,
        significant_results,
        conserved_pathways,
        summary,
        args.output_directory,
    )

    print(f"Total pathway results: {len(all_results):,}")

    print(f"Significant pathway results: {len(significant_results):,}")

    print(f"Conserved pathway results: {len(conserved_pathways):,}")

    print("Top conserved REPOP vs TIS pathways:")

    top_escape_pathways = conserved_pathways.loc[
        conserved_pathways["comparison"] == "REPOP_vs_TIS"
    ].head(15)

    for row in top_escape_pathways.itertuples(index=False):
        print(
            f"{row.pathway}: "
            f"NES="
            f"{row.mean_normalized_enrichment_score:.3f}, "
            f"direction={row.direction}, "
            f"FDR≤"
            f"{row.maximum_false_discovery_rate:.4f}"
        )

    print(f"Saved pathway results to {args.output_directory}")


if __name__ == "__main__":
    main()
