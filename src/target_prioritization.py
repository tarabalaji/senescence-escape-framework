from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.regulatory_propagation import (
    min_max_scale,
    normalize_gene,
    prepare_network_regulatory_evidence,
    prepare_network_transition_evidence,
    require_columns,
)

DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")
DEFAULT_MARKERS_PATH = Path("results/tables/escape_prone_conserved_markers.csv")
DEFAULT_TRAJECTORY_PATH = Path("results/tables/trajectory_conserved_genes.csv")
DEFAULT_DE_PATH = Path("results/tables/differential_expression_all.csv")
DEFAULT_REGULATORS_PATH = Path("results/tables/conserved_regulators.csv")
DEFAULT_EDGES_PATH = Path("results/tables/regulatory_network_edges.csv")
DEFAULT_TRANSITION_PATH = Path("results/tables/transition_model_coefficients.csv")
DEFAULT_SIGNATURE_PATH = Path("results/tables/repopulation_signature.csv")

DEFAULT_COMPARISON = "REPOP_vs_TIS"
DEFAULT_MAXIMUM_TARGETS = 100
DEFAULT_MINIMUM_EVIDENCE_SOURCES = 2

DEFAULT_WEIGHTS = {
    "escape_marker_score": 0.20,
    "trajectory_score": 0.20,
    "differential_expression_score": 0.20,
    "regulatory_score": 0.20,
    "transition_model_score": 0.10,
    "signature_score": 0.10,
}


def validate_weights(weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError("At least one evidence weight is required")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Evidence weights cannot be negative")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one evidence weight must be positive")


def _first_existing_column(
    table: pd.DataFrame,
    candidates: list[str],
    table_name: str,
) -> str:
    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    raise ValueError(f"{table_name} must contain one of: {candidates}")


def prepare_escape_marker_evidence(
    markers: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        markers,
        {"gene"},
        "Escape-marker table",
    )

    effect_column = _first_existing_column(
        markers,
        [
            "mean_log_fold_change",
            "mean_log2_fold_change",
            "log2_fold_change",
            "score",
        ],
        "Escape-marker table",
    )

    prepared = markers[["gene", effect_column]].copy()
    prepared["gene"] = prepared["gene"].map(normalize_gene)
    prepared["escape_marker_effect"] = pd.to_numeric(
        prepared[effect_column],
        errors="coerce",
    )

    prepared = prepared.dropna(subset=["gene", "escape_marker_effect"])
    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby("gene", as_index=False).agg(
        escape_marker_effect=(
            "escape_marker_effect",
            "mean",
        )
    )

    prepared["escape_marker_score"] = min_max_scale(
        prepared["escape_marker_effect"].abs()
    )
    prepared["escape_marker_direction"] = np.where(
        prepared["escape_marker_effect"] >= 0,
        "higher_in_escape_prone",
        "lower_in_escape_prone",
    )

    return prepared


def prepare_trajectory_evidence(
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        trajectory,
        {"gene"},
        "Trajectory table",
    )

    effect_column = _first_existing_column(
        trajectory,
        [
            "mean_spearman_correlation",
            "spearman_correlation",
            "correlation",
            "trajectory_score",
        ],
        "Trajectory table",
    )

    prepared = trajectory[["gene", effect_column]].copy()
    prepared["gene"] = prepared["gene"].map(normalize_gene)
    prepared["trajectory_correlation"] = pd.to_numeric(
        prepared[effect_column],
        errors="coerce",
    )

    prepared = prepared.dropna(subset=["gene", "trajectory_correlation"])
    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby("gene", as_index=False).agg(
        trajectory_correlation=(
            "trajectory_correlation",
            "mean",
        )
    )

    prepared["trajectory_score"] = min_max_scale(
        prepared["trajectory_correlation"].abs()
    )
    prepared["trajectory_direction"] = np.where(
        prepared["trajectory_correlation"] >= 0,
        "increasing",
        "decreasing",
    )

    return prepared


def prepare_differential_expression_evidence(
    differential_expression: pd.DataFrame,
    comparison: str = DEFAULT_COMPARISON,
    maximum_adjusted_p_value: float = 0.05,
) -> pd.DataFrame:
    require_columns(
        differential_expression,
        {
            "gene",
            "comparison",
            "log2_fold_change",
            "adjusted_p_value",
        },
        "Differential-expression table",
    )

    selected = differential_expression.loc[
        differential_expression["comparison"].astype(str).eq(comparison)
    ].copy()

    if selected.empty:
        raise ValueError(f"No differential-expression rows found for {comparison}")

    selected["gene"] = selected["gene"].map(normalize_gene)
    selected["log2_fold_change"] = pd.to_numeric(
        selected["log2_fold_change"],
        errors="coerce",
    )
    selected["adjusted_p_value"] = pd.to_numeric(
        selected["adjusted_p_value"],
        errors="coerce",
    )

    selected = selected.dropna(
        subset=[
            "gene",
            "log2_fold_change",
            "adjusted_p_value",
        ]
    )
    selected = selected.loc[selected["adjusted_p_value"] <= maximum_adjusted_p_value]

    rows: list[dict[str, object]] = []
    has_cell_line = "cell_line" in selected.columns

    for gene, group in selected.groupby(
        "gene",
        observed=True,
    ):
        effects = group["log2_fold_change"]

        if not ((effects > 0).all() or (effects < 0).all()):
            continue

        mean_effect = float(effects.mean())
        worst_p_value = max(
            float(group["adjusted_p_value"].max()),
            1e-300,
        )

        rows.append(
            {
                "gene": gene,
                "differential_expression_effect": mean_effect,
                "de_direction": (
                    "higher_in_repop" if mean_effect > 0 else "lower_in_repop"
                ),
                "de_cell_lines_supported": (
                    int(group["cell_line"].nunique()) if has_cell_line else 1
                ),
                "de_significance": -np.log10(worst_p_value),
            }
        )

    prepared = pd.DataFrame(rows)

    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "differential_expression_effect",
                "de_direction",
                "de_cell_lines_supported",
                "differential_expression_score",
            ]
        )

    effect_score = min_max_scale(
        prepared["differential_expression_effect"].abs()
    ).fillna(0.0)
    significance_score = min_max_scale(prepared["de_significance"]).fillna(0.0)
    support_score = min_max_scale(prepared["de_cell_lines_supported"]).fillna(0.0)

    prepared["differential_expression_score"] = (
        0.50 * effect_score + 0.30 * significance_score + 0.20 * support_score
    )

    return prepared.drop(columns=["de_significance"])


def prepare_signature_evidence(
    signature: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        signature,
        {"gene"},
        "Signature table",
    )

    direction_column = _first_existing_column(
        signature,
        ["direction", "signature_direction"],
        "Signature table",
    )

    prepared = signature[["gene", direction_column]].copy()
    prepared["gene"] = prepared["gene"].map(normalize_gene)
    prepared = prepared.loc[prepared["gene"].ne("")]
    prepared = prepared.drop_duplicates(subset=["gene"])
    prepared = prepared.rename(
        columns={
            direction_column: "signature_direction",
        }
    )
    prepared["signature_score"] = 1.0

    return prepared


def merge_evidence_tables(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    usable = [table for table in tables if not table.empty]

    if not usable:
        raise ValueError("No evidence tables contained data")

    return reduce(
        lambda left, right: left.merge(
            right,
            on="gene",
            how="outer",
        ),
        usable,
    )


def calculate_priority_scores(
    evidence: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    minimum_evidence_sources: int = (DEFAULT_MINIMUM_EVIDENCE_SOURCES),
) -> pd.DataFrame:
    validate_weights(weights)

    if minimum_evidence_sources < 1:
        raise ValueError("minimum_evidence_sources must be at least 1")

    scored = evidence.copy()

    for score_column in weights:
        if score_column not in scored.columns:
            scored[score_column] = np.nan

    weighted_sum = pd.Series(
        0.0,
        index=scored.index,
    )
    available_weight = pd.Series(
        0.0,
        index=scored.index,
    )
    source_count = pd.Series(
        0,
        index=scored.index,
        dtype=int,
    )

    for score_column, weight in weights.items():
        values = pd.to_numeric(
            scored[score_column],
            errors="coerce",
        )
        available = values.notna()

        weighted_sum.loc[available] += values.loc[available] * weight
        available_weight.loc[available] += weight
        source_count.loc[available] += 1

    scored["evidence_source_count"] = source_count
    scored["available_evidence_weight"] = available_weight
    scored["priority_score"] = np.where(
        available_weight > 0,
        weighted_sum / available_weight,
        0.0,
    )

    scored = scored.loc[
        scored["evidence_source_count"] >= minimum_evidence_sources
    ].copy()

    scored["evidence_sources"] = scored.apply(
        lambda row: ";".join(
            column.removesuffix("_score") for column in weights if pd.notna(row[column])
        ),
        axis=1,
    )

    scored = scored.sort_values(
        [
            "priority_score",
            "evidence_source_count",
            "gene",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    scored.insert(
        0,
        "rank",
        np.arange(1, len(scored) + 1),
    )

    return scored


def run_ablation_analysis(
    evidence: pd.DataFrame,
    weights: dict[str, float],
    minimum_evidence_sources: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    full = calculate_priority_scores(
        evidence,
        weights,
        minimum_evidence_sources,
    )

    ablated: dict[str, pd.DataFrame] = {}

    for removed_source in weights:
        reduced_weights = {
            source: weight
            for source, weight in weights.items()
            if source != removed_source
        }

        ablated[removed_source] = calculate_priority_scores(
            evidence,
            reduced_weights,
            max(1, minimum_evidence_sources - 1),
        )

    stability = full[["gene", "rank", "priority_score"]].rename(
        columns={
            "rank": "full_rank",
            "priority_score": "full_priority_score",
        }
    )

    missing_rank = len(full) + 1

    for source, rankings in ablated.items():
        source_name = source.removesuffix("_score")
        comparison = rankings[["gene", "rank", "priority_score"]].rename(
            columns={
                "rank": f"rank_without_{source_name}",
                "priority_score": f"score_without_{source_name}",
            }
        )

        stability = stability.merge(
            comparison,
            on="gene",
            how="left",
        )

        rank_column = f"rank_without_{source_name}"
        stability[rank_column] = stability[rank_column].fillna(missing_rank)

        stability[f"rank_shift_without_{source_name}"] = (
            stability[rank_column] - stability["full_rank"]
        )

    shift_columns = [
        column
        for column in stability.columns
        if column.startswith("rank_shift_without_")
    ]

    stability["mean_absolute_rank_shift"] = stability[shift_columns].abs().mean(axis=1)
    stability["maximum_absolute_rank_shift"] = (
        stability[shift_columns].abs().max(axis=1)
    )
    stability["rank_stability_score"] = 1.0 / (
        1.0 + stability["mean_absolute_rank_shift"]
    )

    return stability, ablated


def create_ablation_summary(
    full_rankings: pd.DataFrame,
    ablated_rankings: dict[str, pd.DataFrame],
    top_k: int = 20,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    full_top = set(full_rankings.head(top_k)["gene"])
    rows: list[dict[str, object]] = []

    for source, rankings in ablated_rankings.items():
        ablated_top = set(rankings.head(top_k)["gene"])
        intersection = full_top & ablated_top
        union = full_top | ablated_top

        rows.append(
            {
                "removed_evidence_source": (source.removesuffix("_score")),
                "top_k": top_k,
                "top_k_overlap": len(intersection),
                "top_k_overlap_fraction": (len(intersection) / max(len(full_top), 1)),
                "jaccard_similarity": (len(intersection) / max(len(union), 1)),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("top_k_overlap_fraction").reset_index(drop=True)
    )


def create_evidence_matrix(
    rankings: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    columns = [
        "rank",
        "gene",
        "priority_score",
        "evidence_source_count",
    ] + [column for column in score_columns if column in rankings.columns]

    matrix = rankings[columns].copy()

    for column in score_columns:
        if column in matrix.columns:
            matrix[f"has_{column.removesuffix('_score')}"] = matrix[column].notna()

    return matrix


def create_candidate_summary(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    preferred = [
        "rank",
        "gene",
        "priority_score",
        "evidence_source_count",
        "evidence_sources",
        "escape_marker_effect",
        "escape_marker_direction",
        "trajectory_correlation",
        "trajectory_direction",
        "differential_expression_effect",
        "de_direction",
        "de_cell_lines_supported",
        "regulatory_effect",
        "regulatory_score",
        "regulatory_evidence_type",
        "upstream_regulators",
        "transition_model_effect",
        "transition_model_score",
        "transition_model_direction",
        "transition_evidence_type",
        "transition_upstream_regulators",
        "signature_direction",
    ]

    return rankings[
        [column for column in preferred if column in rankings.columns]
    ].copy()


def run_target_prioritization(
    markers: pd.DataFrame,
    trajectory_genes: pd.DataFrame,
    differential_expression: pd.DataFrame,
    regulators: pd.DataFrame,
    regulatory_edges: pd.DataFrame,
    transition_coefficients: pd.DataFrame,
    signature: pd.DataFrame,
    comparison: str = DEFAULT_COMPARISON,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    minimum_evidence_sources: int = (DEFAULT_MINIMUM_EVIDENCE_SOURCES),
    maximum_targets: int = DEFAULT_MAXIMUM_TARGETS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if maximum_targets < 1:
        raise ValueError("maximum_targets must be at least 1")

    evidence = merge_evidence_tables(
        [
            prepare_escape_marker_evidence(markers),
            prepare_trajectory_evidence(trajectory_genes),
            prepare_differential_expression_evidence(
                differential_expression,
                comparison=comparison,
            ),
            prepare_network_regulatory_evidence(
                regulators,
                regulatory_edges,
            ),
            prepare_network_transition_evidence(
                transition_coefficients,
                regulatory_edges,
            ),
            prepare_signature_evidence(signature),
        ]
    )

    full_rankings = calculate_priority_scores(
        evidence,
        weights,
        minimum_evidence_sources,
    )
    rankings = full_rankings.head(maximum_targets).copy()

    evidence_matrix = create_evidence_matrix(
        rankings,
        list(weights),
    )

    stability, ablated = run_ablation_analysis(
        evidence,
        weights,
        minimum_evidence_sources,
    )
    stability = stability.loc[stability["gene"].isin(rankings["gene"])].copy()

    ablation_summary = create_ablation_summary(
        full_rankings,
        ablated,
        top_k=min(20, len(full_rankings)),
    )

    candidate_summary = create_candidate_summary(rankings)

    return (
        rankings,
        evidence_matrix,
        stability,
        ablation_summary,
        candidate_summary,
    )


def save_results(
    rankings: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    stability: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "target_priority_rankings.csv": rankings,
        "target_evidence_matrix.csv": evidence_matrix,
        "target_rank_stability.csv": stability,
        "target_ablation_summary.csv": ablation_summary,
        "target_candidate_summary.csv": candidate_summary,
    }

    for filename, table in outputs.items():
        table.to_csv(
            output_directory / filename,
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prioritize senescence-escape targets using "
            "multi-source and propagated regulatory evidence."
        )
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_MARKERS_PATH,
    )
    parser.add_argument(
        "--trajectory-genes",
        type=Path,
        default=DEFAULT_TRAJECTORY_PATH,
    )
    parser.add_argument(
        "--differential-expression",
        type=Path,
        default=DEFAULT_DE_PATH,
    )
    parser.add_argument(
        "--regulators",
        type=Path,
        default=DEFAULT_REGULATORS_PATH,
    )
    parser.add_argument(
        "--regulatory-edges",
        type=Path,
        default=DEFAULT_EDGES_PATH,
    )
    parser.add_argument(
        "--transition-coefficients",
        type=Path,
        default=DEFAULT_TRANSITION_PATH,
    )
    parser.add_argument(
        "--signature",
        type=Path,
        default=DEFAULT_SIGNATURE_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--comparison",
        default=DEFAULT_COMPARISON,
    )
    parser.add_argument(
        "--minimum-evidence-sources",
        type=int,
        default=DEFAULT_MINIMUM_EVIDENCE_SOURCES,
    )
    parser.add_argument(
        "--maximum-targets",
        type=int,
        default=DEFAULT_MAXIMUM_TARGETS,
    )

    args = parser.parse_args()

    input_paths = [
        args.markers,
        args.trajectory_genes,
        args.differential_expression,
        args.regulators,
        args.regulatory_edges,
        args.transition_coefficients,
        args.signature,
    ]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    results = run_target_prioritization(
        markers=pd.read_csv(args.markers),
        trajectory_genes=pd.read_csv(args.trajectory_genes),
        differential_expression=pd.read_csv(args.differential_expression),
        regulators=pd.read_csv(args.regulators),
        regulatory_edges=pd.read_csv(args.regulatory_edges),
        transition_coefficients=pd.read_csv(args.transition_coefficients),
        signature=pd.read_csv(args.signature),
        comparison=args.comparison,
        minimum_evidence_sources=(args.minimum_evidence_sources),
        maximum_targets=args.maximum_targets,
    )

    save_results(
        *results,
        output_directory=args.output_directory,
    )

    rankings = results[0]
    propagated_count = int(
        rankings["regulatory_evidence_type"]
        .isin(
            [
                "propagated",
                "direct_and_propagated",
            ]
        )
        .sum()
    )

    print(f"Candidate targets ranked: {len(rankings):,}")
    print(f"Targets with propagated regulatory evidence: {propagated_count:,}")
    print("Top candidate targets:")

    for row in rankings.head(15).itertuples(index=False):
        upstream = getattr(
            row,
            "upstream_regulators",
            "",
        )

        if pd.isna(upstream) or not upstream:
            upstream = "none"

        print(
            f"{row.rank}. {row.gene}: "
            f"score={row.priority_score:.3f}, "
            f"sources={row.evidence_source_count}, "
            f"upstream={upstream}"
        )

    print(f"Saved target-prioritization results to {args.output_directory}")


if __name__ == "__main__":
    main()
