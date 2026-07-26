from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def min_max_scale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna()

    if not valid.any():
        return result

    minimum = float(numeric.loc[valid].min())
    maximum = float(numeric.loc[valid].max())

    if np.isclose(minimum, maximum):
        result.loc[valid] = 1.0 if maximum > 0 else 0.0
        return result

    result.loc[valid] = (numeric.loc[valid] - minimum) / (maximum - minimum)

    return result


def require_columns(
    table: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = required.difference(table.columns)

    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")


def parse_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
                "yes": 1.0,
                "no": 0.0,
            }
        )
    )


def prepare_direct_regulator_evidence(
    regulators: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        regulators,
        {"transcription_factor", "mean_regulatory_score"},
        "Conserved-regulator table",
    )

    prepared = regulators[["transcription_factor", "mean_regulatory_score"]].copy()

    prepared["gene"] = prepared["transcription_factor"].map(normalize_gene)
    prepared["regulatory_effect"] = pd.to_numeric(
        prepared["mean_regulatory_score"],
        errors="coerce",
    )

    prepared = prepared.dropna(subset=["gene", "regulatory_effect"])
    prepared = prepared.loc[prepared["gene"].ne("")]

    prepared = prepared.groupby("gene", as_index=False).agg(
        regulatory_effect=(
            "regulatory_effect",
            "max",
        )
    )

    prepared["regulatory_score"] = min_max_scale(prepared["regulatory_effect"].abs())
    prepared["regulatory_evidence_type"] = "direct"
    prepared["upstream_regulators"] = prepared["gene"]

    return prepared


def prepare_direct_transition_evidence(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        coefficients,
        {"feature", "coefficient"},
        "Transition-model coefficient table",
    )

    selected = coefficients.loc[
        coefficients["feature"].astype(str).str.startswith("regulator__", na=False)
    ].copy()

    selected["gene"] = (
        selected["feature"]
        .astype(str)
        .str.replace("regulator__", "", regex=False)
        .map(normalize_gene)
    )
    selected["transition_model_effect"] = pd.to_numeric(
        selected["coefficient"],
        errors="coerce",
    )

    selected = selected.dropna(subset=["gene", "transition_model_effect"])
    selected = selected.loc[selected["gene"].ne("")]

    prepared = selected.groupby("gene", as_index=False).agg(
        transition_model_effect=(
            "transition_model_effect",
            "mean",
        )
    )

    prepared["transition_model_score"] = min_max_scale(
        prepared["transition_model_effect"].abs()
    )
    prepared["transition_evidence_type"] = "direct"
    prepared["transition_upstream_regulators"] = prepared["gene"]
    prepared["transition_model_direction"] = np.where(
        prepared["transition_model_effect"] >= 0,
        "supports_repopulation",
        "opposes_repopulation",
    )

    return prepared


def summarize_regulatory_edges(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        edges,
        {
            "cell_line",
            "transcription_factor",
            "target_gene",
            "absolute_tf_target_correlation",
            "rap_consistent",
        },
        "Regulatory-network edge table",
    )

    prepared = edges.copy()
    prepared["transcription_factor"] = prepared["transcription_factor"].map(
        normalize_gene
    )
    prepared["target_gene"] = prepared["target_gene"].map(normalize_gene)
    prepared["absolute_tf_target_correlation"] = pd.to_numeric(
        prepared["absolute_tf_target_correlation"],
        errors="coerce",
    )
    prepared["rap_consistent"] = parse_boolean(prepared["rap_consistent"])

    prepared = prepared.dropna(
        subset=[
            "transcription_factor",
            "target_gene",
            "absolute_tf_target_correlation",
            "rap_consistent",
        ]
    )
    prepared = prepared.loc[
        prepared["transcription_factor"].ne("") & prepared["target_gene"].ne("")
    ]

    summarized = prepared.groupby(
        ["transcription_factor", "target_gene"],
        as_index=False,
    ).agg(
        mean_absolute_correlation=(
            "absolute_tf_target_correlation",
            "mean",
        ),
        cell_lines_supported=(
            "cell_line",
            "nunique",
        ),
        rap_consistency_rate=(
            "rap_consistent",
            "mean",
        ),
    )

    correlation_score = min_max_scale(summarized["mean_absolute_correlation"]).fillna(
        0.0
    )

    total_cell_lines = max(
        int(prepared["cell_line"].nunique()),
        1,
    )
    cell_line_score = (summarized["cell_lines_supported"] / total_cell_lines).clip(
        0.0, 1.0
    )

    summarized["edge_support_score"] = (
        0.50 * correlation_score
        + 0.30 * cell_line_score
        + 0.20 * summarized["rap_consistency_rate"]
    ).clip(0.0, 1.0)

    return summarized


def _propagate_to_targets(
    summarized_edges: pd.DataFrame,
    direct_evidence: pd.DataFrame,
    direct_score_column: str,
    direct_effect_column: str,
    output_score_column: str,
    output_effect_column: str,
    output_regulator_column: str,
) -> pd.DataFrame:
    direct = direct_evidence.rename(
        columns={
            "gene": "transcription_factor",
            direct_score_column: "direct_score",
            direct_effect_column: "direct_effect",
        }
    )[["transcription_factor", "direct_score", "direct_effect"]]

    merged = summarized_edges.merge(
        direct,
        on="transcription_factor",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                output_score_column,
                output_effect_column,
                output_regulator_column,
            ]
        )

    merged["propagated_score"] = merged["edge_support_score"] * merged["direct_score"]
    merged["propagated_effect"] = merged["edge_support_score"] * merged["direct_effect"]

    rows: list[dict[str, object]] = []

    for target_gene, group in merged.groupby(
        "target_gene",
        observed=True,
    ):
        ordered = group.sort_values(
            "propagated_score",
            ascending=False,
        )
        strongest = ordered.iloc[0]
        regulator_count = int(ordered["transcription_factor"].nunique())
        regulator_bonus = min(regulator_count / 3.0, 1.0)

        combined_score = (
            0.65 * float(ordered["propagated_score"].max())
            + 0.25 * float(ordered["propagated_score"].mean())
            + 0.10 * regulator_bonus
        )

        regulators = ";".join(
            ordered["transcription_factor"].drop_duplicates().astype(str)
        )

        rows.append(
            {
                "gene": target_gene,
                output_score_column: float(np.clip(combined_score, 0.0, 1.0)),
                output_effect_column: float(strongest["propagated_effect"]),
                output_regulator_column: regulators,
            }
        )

    return pd.DataFrame(rows)


def _combine_direct_and_propagated(
    direct: pd.DataFrame,
    propagated: pd.DataFrame,
    score_column: str,
    effect_column: str,
    regulator_column: str,
    evidence_type_column: str,
) -> pd.DataFrame:
    direct_columns = direct[
        ["gene", score_column, effect_column, regulator_column]
    ].rename(
        columns={
            score_column: f"direct_{score_column}",
            effect_column: f"direct_{effect_column}",
            regulator_column: f"direct_{regulator_column}",
        }
    )

    propagated_columns = propagated.rename(
        columns={
            score_column: f"propagated_{score_column}",
            effect_column: f"propagated_{effect_column}",
            regulator_column: f"propagated_{regulator_column}",
        }
    )

    combined = direct_columns.merge(
        propagated_columns,
        on="gene",
        how="outer",
    )

    direct_score = combined[f"direct_{score_column}"]
    propagated_score = combined[f"propagated_{score_column}"]

    combined[score_column] = pd.concat(
        [direct_score, propagated_score],
        axis=1,
    ).max(axis=1, skipna=True)

    combined[effect_column] = combined[f"propagated_{effect_column}"].combine_first(
        combined[f"direct_{effect_column}"]
    )

    combined[regulator_column] = combined[
        f"propagated_{regulator_column}"
    ].combine_first(combined[f"direct_{regulator_column}"])

    combined[evidence_type_column] = np.select(
        [
            direct_score.notna() & propagated_score.notna(),
            propagated_score.notna(),
            direct_score.notna(),
        ],
        [
            "direct_and_propagated",
            "propagated",
            "direct",
        ],
        default="none",
    )

    return combined[
        [
            "gene",
            effect_column,
            score_column,
            evidence_type_column,
            regulator_column,
        ]
    ]


def prepare_network_regulatory_evidence(
    regulators: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    direct = prepare_direct_regulator_evidence(regulators)
    summarized_edges = summarize_regulatory_edges(edges)

    propagated = _propagate_to_targets(
        summarized_edges=summarized_edges,
        direct_evidence=direct,
        direct_score_column="regulatory_score",
        direct_effect_column="regulatory_effect",
        output_score_column="regulatory_score",
        output_effect_column="regulatory_effect",
        output_regulator_column="upstream_regulators",
    )

    return _combine_direct_and_propagated(
        direct=direct,
        propagated=propagated,
        score_column="regulatory_score",
        effect_column="regulatory_effect",
        regulator_column="upstream_regulators",
        evidence_type_column="regulatory_evidence_type",
    )


def prepare_network_transition_evidence(
    coefficients: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    direct = prepare_direct_transition_evidence(coefficients)
    summarized_edges = summarize_regulatory_edges(edges)

    propagated = _propagate_to_targets(
        summarized_edges=summarized_edges,
        direct_evidence=direct,
        direct_score_column="transition_model_score",
        direct_effect_column="transition_model_effect",
        output_score_column="transition_model_score",
        output_effect_column="transition_model_effect",
        output_regulator_column="transition_upstream_regulators",
    )

    combined = _combine_direct_and_propagated(
        direct=direct,
        propagated=propagated,
        score_column="transition_model_score",
        effect_column="transition_model_effect",
        regulator_column="transition_upstream_regulators",
        evidence_type_column="transition_evidence_type",
    )

    combined["transition_model_direction"] = np.where(
        combined["transition_model_effect"] >= 0,
        "supports_repopulation",
        "opposes_repopulation",
    )

    return combined
