from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")

DEFAULT_MARKER_PATH = Path("results/tables/escape_prone_conserved_markers.csv")

DEFAULT_TRAJECTORY_PATH = Path("results/tables/trajectory_conserved_genes.csv")

DEFAULT_DIFFERENTIAL_EXPRESSION_PATH = Path(
    "results/tables/differential_expression_all.csv"
)

DEFAULT_REGULATOR_PATH = Path("results/tables/conserved_regulators.csv")

DEFAULT_TRANSITION_COEFFICIENT_PATH = Path(
    "results/tables/transition_model_coefficients.csv"
)

DEFAULT_SIGNATURE_PATH = Path("results/tables/repopulation_signature.csv")

DEFAULT_COMPARISON = "REPOP_vs_TIS"
DEFAULT_MAXIMUM_TARGETS = 100
DEFAULT_MINIMUM_EVIDENCE_SOURCES = 2

DEFAULT_WEIGHTS = {
    "escape_marker_score": 0.25,
    "trajectory_score": 0.20,
    "differential_expression_score": 0.20,
    "regulatory_score": 0.15,
    "transition_model_score": 0.10,
    "signature_score": 0.10,
}


def validate_weights(
    weights: dict[str, float],
) -> None:
    if not weights:
        raise ValueError("At least one evidence weight is required")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Evidence weights cannot be negative")

    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("At least one evidence weight must be positive")


def normalize_gene_name(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_zero_to_one(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = pd.Series(
        0.0,
        index=values.index,
        dtype=float,
    )

    valid = numeric.notna()

    if not valid.any():
        return result

    valid_values = numeric.loc[valid]

    minimum = float(valid_values.min())
    maximum = float(valid_values.max())

    if np.isclose(minimum, maximum):
        result.loc[valid] = 1.0 if maximum > 0 else 0.0

        return result

    result.loc[valid] = (valid_values - minimum) / (maximum - minimum)

    return result


def normalize_absolute_effect(
    values: pd.Series,
) -> pd.Series:
    absolute_values = pd.to_numeric(
        values,
        errors="coerce",
    ).abs()

    return normalize_zero_to_one(absolute_values)


def find_first_column(
    table: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    return None


def require_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    missing_columns = required_columns.difference(table.columns)

    if missing_columns:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing_columns)}")


def prepare_escape_marker_evidence(
    markers: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        markers,
        {
            "gene",
            "mean_log_fold_change",
        },
        "Escape-marker table",
    )

    prepared = markers[
        [
            "gene",
            "mean_log_fold_change",
        ]
    ].copy()

    prepared["gene"] = prepared["gene"].map(normalize_gene_name)

    prepared["escape_marker_effect"] = pd.to_numeric(
        prepared["mean_log_fold_change"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            "gene",
            "escape_marker_effect",
        ]
    )

    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby(
        "gene",
        as_index=False,
    ).agg(
        escape_marker_effect=(
            "escape_marker_effect",
            "mean",
        )
    )

    prepared["escape_marker_score"] = normalize_absolute_effect(
        prepared["escape_marker_effect"]
    )

    prepared["escape_marker_direction"] = np.where(
        prepared["escape_marker_effect"] > 0,
        "higher_in_escape_prone",
        "lower_in_escape_prone",
    )

    return prepared


def prepare_trajectory_evidence(
    trajectory_genes: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        trajectory_genes,
        {
            "gene",
            "mean_spearman_correlation",
        },
        "Trajectory-gene table",
    )

    prepared = trajectory_genes[
        [
            "gene",
            "mean_spearman_correlation",
        ]
    ].copy()

    prepared["gene"] = prepared["gene"].map(normalize_gene_name)

    prepared["trajectory_correlation"] = pd.to_numeric(
        prepared["mean_spearman_correlation"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            "gene",
            "trajectory_correlation",
        ]
    )

    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby(
        "gene",
        as_index=False,
    ).agg(
        trajectory_correlation=(
            "trajectory_correlation",
            "mean",
        )
    )

    prepared["trajectory_score"] = normalize_absolute_effect(
        prepared["trajectory_correlation"]
    )

    prepared["trajectory_direction"] = np.where(
        prepared["trajectory_correlation"] > 0,
        "increasing",
        "decreasing",
    )

    return prepared


def prepare_differential_expression_evidence(
    results: pd.DataFrame,
    comparison: str = DEFAULT_COMPARISON,
    maximum_adjusted_p_value: float = 0.05,
) -> pd.DataFrame:
    require_columns(
        results,
        {
            "cell_line",
            "comparison",
            "gene",
            "log2_fold_change",
            "adjusted_p_value",
        },
        "Differential-expression table",
    )

    selected = results.loc[results["comparison"].astype(str).eq(comparison)].copy()

    if selected.empty:
        raise ValueError(f"No differential-expression rows found for {comparison}")

    selected["gene"] = selected["gene"].map(normalize_gene_name)

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

    selected = selected.loc[
        selected["adjusted_p_value"] <= maximum_adjusted_p_value
    ].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "differential_expression_effect",
                "differential_expression_score",
                "de_cell_lines_supported",
                "de_direction",
            ]
        )

    rows = []

    for gene, group in selected.groupby(
        "gene",
        observed=True,
    ):
        effects = group["log2_fold_change"]

        positive_count = int((effects > 0).sum())

        negative_count = int((effects < 0).sum())

        direction_consistent = positive_count == len(effects) or negative_count == len(
            effects
        )

        if not direction_consistent:
            continue

        mean_effect = float(effects.mean())

        minimum_significance = float(
            -np.log10(
                max(
                    group["adjusted_p_value"].max(),
                    1e-300,
                )
            )
        )

        rows.append(
            {
                "gene": gene,
                "differential_expression_effect": (mean_effect),
                "de_cell_lines_supported": int(group["cell_line"].nunique()),
                "de_significance": minimum_significance,
                "de_direction": (
                    "higher_in_repop" if mean_effect > 0 else "lower_in_repop"
                ),
            }
        )

    prepared = pd.DataFrame(rows)

    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "differential_expression_effect",
                "differential_expression_score",
                "de_cell_lines_supported",
                "de_direction",
            ]
        )

    effect_score = normalize_absolute_effect(prepared["differential_expression_effect"])

    significance_score = normalize_zero_to_one(prepared["de_significance"])

    support_score = normalize_zero_to_one(prepared["de_cell_lines_supported"])

    prepared["differential_expression_score"] = (
        0.50 * effect_score + 0.30 * significance_score + 0.20 * support_score
    )

    return prepared.drop(columns=["de_significance"])


def prepare_regulatory_evidence(
    regulators: pd.DataFrame,
) -> pd.DataFrame:
    regulator_column = find_first_column(
        regulators,
        (
            "transcription_factor",
            "regulator",
            "gene",
        ),
    )

    if regulator_column is None:
        raise ValueError("Regulator table does not contain a regulator column")

    score_column = find_first_column(
        regulators,
        (
            "mean_regulatory_score",
            "regulatory_score",
            "minimum_regulatory_score",
            "absolute_rap_correlation",
        ),
    )

    if score_column is None:
        raise ValueError("Regulator table does not contain a regulatory score column")

    prepared = regulators[
        [
            regulator_column,
            score_column,
        ]
    ].copy()

    prepared.columns = [
        "gene",
        "regulatory_effect",
    ]

    prepared["gene"] = prepared["gene"].map(normalize_gene_name)

    prepared["regulatory_effect"] = pd.to_numeric(
        prepared["regulatory_effect"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            "gene",
            "regulatory_effect",
        ]
    )

    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby(
        "gene",
        as_index=False,
    ).agg(
        regulatory_effect=(
            "regulatory_effect",
            "max",
        )
    )

    prepared["regulatory_score"] = normalize_absolute_effect(
        prepared["regulatory_effect"]
    )

    return prepared


def prepare_transition_model_evidence(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        coefficients,
        {
            "feature",
            "coefficient",
        },
        "Transition-coefficient table",
    )

    prepared = coefficients.copy()

    prepared = prepared.loc[
        prepared["feature"].astype(str).str.startswith("regulator__")
    ].copy()

    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "transition_model_effect",
                "transition_model_score",
                "transition_model_direction",
            ]
        )

    prepared["gene"] = (
        prepared["feature"]
        .astype(str)
        .str.replace(
            "regulator__",
            "",
            regex=False,
        )
        .map(normalize_gene_name)
    )

    prepared["transition_model_effect"] = pd.to_numeric(
        prepared["coefficient"],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            "gene",
            "transition_model_effect",
        ]
    )

    prepared = prepared.groupby(
        "gene",
        as_index=False,
    ).agg(
        transition_model_effect=(
            "transition_model_effect",
            "mean",
        )
    )

    prepared["transition_model_score"] = normalize_absolute_effect(
        prepared["transition_model_effect"]
    )

    prepared["transition_model_direction"] = np.where(
        prepared["transition_model_effect"] > 0,
        "supports_repop_prediction",
        "opposes_repop_prediction",
    )

    return prepared


def prepare_signature_evidence(
    signature: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        signature,
        {
            "gene",
            "direction",
        },
        "Repopulation-signature table",
    )

    prepared = signature[
        [
            "gene",
            "direction",
        ]
    ].copy()

    prepared["gene"] = prepared["gene"].map(normalize_gene_name)

    prepared = prepared.loc[prepared["gene"].ne("")].drop_duplicates(subset=["gene"])

    prepared["signature_score"] = 1.0

    prepared["signature_direction"] = prepared["direction"].astype(str)

    return prepared[
        [
            "gene",
            "signature_score",
            "signature_direction",
        ]
    ]


def merge_evidence_tables(
    evidence_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    nonempty_tables = [table for table in evidence_tables if not table.empty]

    if not nonempty_tables:
        raise ValueError("No target-prioritization evidence was available")

    merged = nonempty_tables[0].copy()

    for table in nonempty_tables[1:]:
        merged = merged.merge(
            table,
            on="gene",
            how="outer",
        )

    return merged


def calculate_priority_scores(
    evidence: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    minimum_evidence_sources: int = (DEFAULT_MINIMUM_EVIDENCE_SOURCES),
) -> pd.DataFrame:
    validate_weights(weights)

    if minimum_evidence_sources <= 0:
        raise ValueError("minimum_evidence_sources must be positive")

    scored = evidence.copy()

    for score_column in weights:
        if score_column not in scored.columns:
            scored[score_column] = np.nan

    weighted_score = pd.Series(
        0.0,
        index=scored.index,
        dtype=float,
    )

    available_weight = pd.Series(
        0.0,
        index=scored.index,
        dtype=float,
    )

    evidence_source_count = pd.Series(
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

        weighted_score.loc[available] += values.loc[available] * weight

        available_weight.loc[available] += weight

        evidence_source_count.loc[available] += 1

    scored["evidence_source_count"] = evidence_source_count

    scored["available_evidence_weight"] = available_weight

    scored["priority_score"] = np.where(
        available_weight > 0,
        weighted_score / available_weight,
        0.0,
    )

    scored = scored.loc[
        scored["evidence_source_count"] >= minimum_evidence_sources
    ].copy()

    scored["evidence_sources"] = scored.apply(
        lambda row: ";".join(
            score_column.replace(
                "_score",
                "",
            )
            for score_column in weights
            if pd.notna(row[score_column])
        ),
        axis=1,
    )

    scored = scored.sort_values(
        [
            "priority_score",
            "evidence_source_count",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    scored.insert(
        0,
        "rank",
        np.arange(
            1,
            len(scored) + 1,
        ),
    )

    return scored


def create_evidence_matrix(
    rankings: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    required_columns = {
        "rank",
        "gene",
        "priority_score",
        "evidence_source_count",
    }

    missing_columns = required_columns.difference(rankings.columns)

    if missing_columns:
        raise ValueError(f"Rankings are missing columns: {sorted(missing_columns)}")

    retained_columns = [
        "rank",
        "gene",
        "priority_score",
        "evidence_source_count",
    ]

    retained_columns.extend(
        column for column in score_columns if column in rankings.columns
    )

    matrix = rankings[retained_columns].copy()

    for score_column in score_columns:
        if score_column in matrix.columns:
            matrix[f"has_{score_column.replace('_score', '')}"] = matrix[
                score_column
            ].notna()

    return matrix


def calculate_rank_stability(
    full_rankings: pd.DataFrame,
    ablated_rankings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if full_rankings.empty:
        return pd.DataFrame()

    full = full_rankings[
        [
            "gene",
            "rank",
            "priority_score",
        ]
    ].rename(
        columns={
            "rank": "full_rank",
            "priority_score": "full_priority_score",
        }
    )

    stability = full.copy()

    maximum_rank = len(full_rankings) + 1

    for removed_source, rankings in ablated_rankings.items():
        source_name = removed_source.replace(
            "_score",
            "",
        )

        ablated = rankings[
            [
                "gene",
                "rank",
                "priority_score",
            ]
        ].rename(
            columns={
                "rank": (f"rank_without_{source_name}"),
                "priority_score": (f"score_without_{source_name}"),
            }
        )

        stability = stability.merge(
            ablated,
            on="gene",
            how="left",
        )

        rank_column = f"rank_without_{source_name}"

        stability[rank_column] = stability[rank_column].fillna(maximum_rank)

        stability[f"rank_shift_without_{source_name}"] = (
            stability[rank_column] - stability["full_rank"]
        )

    rank_shift_columns = [
        column
        for column in stability.columns
        if column.startswith("rank_shift_without_")
    ]

    if rank_shift_columns:
        stability["mean_absolute_rank_shift"] = (
            stability[rank_shift_columns].abs().mean(axis=1)
        )

        stability["maximum_absolute_rank_shift"] = (
            stability[rank_shift_columns].abs().max(axis=1)
        )

        stability["rank_stability_score"] = 1.0 / (
            1.0 + stability["mean_absolute_rank_shift"]
        )
    else:
        stability["mean_absolute_rank_shift"] = 0.0

        stability["maximum_absolute_rank_shift"] = 0.0

        stability["rank_stability_score"] = 1.0

    return stability.sort_values("full_rank").reset_index(drop=True)


def run_ablation_analysis(
    evidence: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    minimum_evidence_sources: int = (DEFAULT_MINIMUM_EVIDENCE_SOURCES),
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    full_rankings = calculate_priority_scores(
        evidence,
        weights=weights,
        minimum_evidence_sources=(minimum_evidence_sources),
    )

    ablated_rankings = {}

    for removed_source in weights:
        ablated_weights = {
            source: weight
            for source, weight in weights.items()
            if source != removed_source
        }

        ablated_rankings[removed_source] = calculate_priority_scores(
            evidence,
            weights=ablated_weights,
            minimum_evidence_sources=max(
                1,
                minimum_evidence_sources - 1,
            ),
        )

    stability = calculate_rank_stability(
        full_rankings,
        ablated_rankings,
    )

    return stability, ablated_rankings


def create_ablation_summary(
    full_rankings: pd.DataFrame,
    ablated_rankings: dict[str, pd.DataFrame],
    top_k: int = 20,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    full_top_genes = set(full_rankings.head(top_k)["gene"])

    rows = []

    for removed_source, rankings in ablated_rankings.items():
        ablated_top_genes = set(rankings.head(top_k)["gene"])

        intersection = full_top_genes & ablated_top_genes

        union = full_top_genes | ablated_top_genes

        rows.append(
            {
                "removed_evidence_source": (
                    removed_source.replace(
                        "_score",
                        "",
                    )
                ),
                "top_k": top_k,
                "top_k_overlap": len(intersection),
                "top_k_overlap_fraction": (
                    len(intersection)
                    / max(
                        len(full_top_genes),
                        1,
                    )
                ),
                "jaccard_similarity": (
                    len(intersection)
                    / max(
                        len(union),
                        1,
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("top_k_overlap_fraction").reset_index(drop=True)
    )


def create_candidate_summary(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    summary_columns = [
        "rank",
        "gene",
        "priority_score",
        "evidence_source_count",
        "evidence_sources",
    ]

    optional_columns = [
        "escape_marker_effect",
        "escape_marker_direction",
        "trajectory_correlation",
        "trajectory_direction",
        "differential_expression_effect",
        "de_direction",
        "de_cell_lines_supported",
        "regulatory_effect",
        "transition_model_effect",
        "transition_model_direction",
        "signature_direction",
    ]

    summary_columns.extend(
        column for column in optional_columns if column in rankings.columns
    )

    return rankings[summary_columns].copy()


def run_target_prioritization(
    markers: pd.DataFrame,
    trajectory_genes: pd.DataFrame,
    differential_expression: pd.DataFrame,
    regulators: pd.DataFrame,
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
    if maximum_targets <= 0:
        raise ValueError("maximum_targets must be greater than zero")

    marker_evidence = prepare_escape_marker_evidence(markers)

    trajectory_evidence = prepare_trajectory_evidence(trajectory_genes)

    differential_evidence = prepare_differential_expression_evidence(
        differential_expression,
        comparison=comparison,
    )

    regulatory_evidence = prepare_regulatory_evidence(regulators)

    transition_evidence = prepare_transition_model_evidence(transition_coefficients)

    signature_evidence = prepare_signature_evidence(signature)

    evidence = merge_evidence_tables(
        [
            marker_evidence,
            trajectory_evidence,
            differential_evidence,
            regulatory_evidence,
            transition_evidence,
            signature_evidence,
        ]
    )

    rankings = (
        calculate_priority_scores(
            evidence,
            weights=weights,
            minimum_evidence_sources=(minimum_evidence_sources),
        )
        .head(maximum_targets)
        .copy()
    )

    score_columns = list(weights)

    evidence_matrix = create_evidence_matrix(
        rankings,
        score_columns=score_columns,
    )

    stability, ablated_rankings = run_ablation_analysis(
        evidence,
        weights=weights,
        minimum_evidence_sources=(minimum_evidence_sources),
    )

    stability = stability.loc[stability["gene"].isin(rankings["gene"])].copy()

    ablation_summary = create_ablation_summary(
        calculate_priority_scores(
            evidence,
            weights=weights,
            minimum_evidence_sources=(minimum_evidence_sources),
        ),
        ablated_rankings,
        top_k=min(
            20,
            len(rankings),
        ),
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

    rankings.to_csv(
        output_directory / "target_priority_rankings.csv",
        index=False,
    )

    evidence_matrix.to_csv(
        output_directory / "target_evidence_matrix.csv",
        index=False,
    )

    stability.to_csv(
        output_directory / "target_rank_stability.csv",
        index=False,
    )

    ablation_summary.to_csv(
        output_directory / "target_ablation_summary.csv",
        index=False,
    )

    candidate_summary.to_csv(
        output_directory / "target_candidate_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate conserved transcriptomic, trajectory, "
            "regulatory, and predictive evidence to prioritize "
            "candidate senescence-escape intervention targets."
        )
    )

    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_MARKER_PATH,
    )

    parser.add_argument(
        "--trajectory-genes",
        type=Path,
        default=DEFAULT_TRAJECTORY_PATH,
    )

    parser.add_argument(
        "--differential-expression",
        type=Path,
        default=(DEFAULT_DIFFERENTIAL_EXPRESSION_PATH),
    )

    parser.add_argument(
        "--regulators",
        type=Path,
        default=DEFAULT_REGULATOR_PATH,
    )

    parser.add_argument(
        "--transition-coefficients",
        type=Path,
        default=(DEFAULT_TRANSITION_COEFFICIENT_PATH),
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
        default=(DEFAULT_MINIMUM_EVIDENCE_SOURCES),
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
        args.transition_coefficients,
        args.signature,
    ]

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Required input does not exist: {input_path}")

    markers = pd.read_csv(args.markers)

    trajectory_genes = pd.read_csv(args.trajectory_genes)

    differential_expression = pd.read_csv(args.differential_expression)

    regulators = pd.read_csv(args.regulators)

    transition_coefficients = pd.read_csv(args.transition_coefficients)

    signature = pd.read_csv(args.signature)

    results = run_target_prioritization(
        markers,
        trajectory_genes,
        differential_expression,
        regulators,
        transition_coefficients,
        signature,
        comparison=args.comparison,
        minimum_evidence_sources=(args.minimum_evidence_sources),
        maximum_targets=args.maximum_targets,
    )

    save_results(
        *results,
        output_directory=(args.output_directory),
    )

    (
        rankings,
        _,
        stability,
        ablation_summary,
        _,
    ) = results

    print(f"Candidate targets ranked: {len(rankings):,}")

    print("Top candidate targets:")

    for row in rankings.head(15).itertuples(index=False):
        print(
            f"{row.rank}. {row.gene}: "
            f"score={row.priority_score:.3f}, "
            f"evidence sources="
            f"{row.evidence_source_count}"
        )

    if not stability.empty:
        stable_targets = stability.sort_values(
            [
                "rank_stability_score",
                "full_rank",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(10)

        print("Most stable prioritized targets:")

        for row in stable_targets.itertuples(index=False):
            print(
                f"{row.gene}: "
                f"full rank={int(row.full_rank)}, "
                f"stability="
                f"{row.rank_stability_score:.3f}"
            )

    print("Ablation top-rank overlap:")

    for row in ablation_summary.itertuples(index=False):
        print(
            f"Without {row.removed_evidence_source}: {row.top_k_overlap_fraction:.2%}"
        )

    print(f"Saved target-prioritization results to {args.output_directory}")


if __name__ == "__main__":
    main()
