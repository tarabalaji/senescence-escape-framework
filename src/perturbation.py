from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_TABLE_PATH = Path("results/tables/transition_feature_table.csv")
DEFAULT_TARGET_RANKINGS_PATH = Path("results/tables/target_priority_rankings.csv")
DEFAULT_PATHWAY_RESULTS_PATH = Path("results/pathways/pathway_results_all.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")
DEFAULT_MODEL_ARTIFACT_PATH = Path("results/tables/rap_plus_pathways_model.joblib")

DEFAULT_LABEL_COLUMN = "transition_label"
DEFAULT_CELL_ID_COLUMN = "cell_id"
DEFAULT_RAP_COLUMN = "rap_score"
DEFAULT_COMPARISON = "REPOP_vs_TIS"
DEFAULT_MAXIMUM_TARGETS = 50
DEFAULT_PERTURBATION_STRENGTH = 1.0
DEFAULT_RAP_EFFECT_FRACTION = 0.25
DEFAULT_RANDOM_STATE = 42


def parse_gene_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    values = re.split(r"[;,|]", str(value))
    genes: list[str] = []
    seen: set[str] = set()

    for value_item in values:
        gene = str(value_item).strip().upper()

        if gene and gene not in seen:
            genes.append(gene)
            seen.add(gene)

    return genes


def sanitize_pathway_name(value: str) -> str:
    sanitized = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")

    return sanitized or "unnamed"


def validate_inputs(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
    cell_id_column: str = DEFAULT_CELL_ID_COLUMN,
) -> None:
    required_feature_columns = {
        cell_id_column,
        label_column,
    }
    missing_features = required_feature_columns.difference(feature_table.columns)

    if missing_features:
        raise ValueError(
            f"Feature table is missing columns: {sorted(missing_features)}"
        )

    if feature_table[cell_id_column].duplicated().any():
        raise ValueError("Feature table contains duplicate cell IDs")

    pathway_columns = [
        column
        for column in feature_table.columns
        if str(column).startswith("pathway__")
    ]

    if not pathway_columns:
        raise ValueError("Feature table contains no pathway features")

    required_target_columns = {
        "gene",
        "priority_score",
    }
    missing_targets = required_target_columns.difference(target_rankings.columns)

    if missing_targets:
        raise ValueError(
            f"Target rankings are missing columns: {sorted(missing_targets)}"
        )

    required_pathway_columns = {
        "comparison",
        "pathway",
        "leading_edge_genes",
    }
    missing_pathways = required_pathway_columns.difference(pathway_results.columns)

    if missing_pathways:
        raise ValueError(
            f"Pathway results are missing columns: {sorted(missing_pathways)}"
        )

    labels = pd.to_numeric(
        feature_table[label_column],
        errors="coerce",
    )

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{label_column} must contain only 0 and 1")

    if labels.nunique() < 2:
        raise ValueError("Transition labels must contain both classes")


def load_model_artifact(
    artifact_path: str | Path,
) -> tuple[Pipeline, list[str], str]:
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            "Saved baseline model does not exist: "
            f"{artifact_path}. Run python -m src.baseline_model first."
        )

    artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise ValueError("Saved model artifact must be a dictionary")

    required_keys = {"pipeline", "feature_columns", "model_name"}
    missing = required_keys.difference(artifact)

    if missing:
        raise ValueError(f"Saved model artifact is missing keys: {sorted(missing)}")

    model = artifact["pipeline"]
    feature_columns = [str(x) for x in artifact["feature_columns"]]
    model_name = str(artifact["model_name"])

    if not hasattr(model, "predict_proba"):
        raise ValueError("Saved pipeline does not support predict_proba")

    if model_name != "rap_plus_pathways":
        raise ValueError(
            "Perturbation requires the rap_plus_pathways model, "
            f"but artifact contains {model_name}"
        )

    return model, feature_columns, model_name


def create_model(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def prepare_model_data(
    feature_table: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
    cell_id_column: str = DEFAULT_CELL_ID_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    pathway_columns = [
        column
        for column in feature_table.columns
        if str(column).startswith("pathway__")
    ]

    selected_columns = []

    if rap_column in feature_table.columns:
        selected_columns.append(rap_column)

    selected_columns.extend(pathway_columns)

    if not selected_columns:
        raise ValueError("No RAP or pathway features were found")

    indexed = feature_table.set_index(cell_id_column).copy()
    indexed.index = indexed.index.astype(str)

    features = indexed[selected_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    labels = pd.to_numeric(
        indexed[label_column],
        errors="coerce",
    ).astype(int)

    if features.isna().any().any():
        invalid = features.columns[features.isna().any()].tolist()

        raise ValueError(f"Model features must be numeric and nonmissing: {invalid}")

    return features.astype(float), labels


def infer_escape_association(
    target_row: pd.Series,
) -> tuple[float, str]:
    candidate_columns = [
        "transition_model_effect",
        "differential_expression_effect",
        "escape_marker_effect",
        "trajectory_correlation",
        "regulatory_effect",
    ]

    for column in candidate_columns:
        if column not in target_row.index:
            continue

        value = pd.to_numeric(
            pd.Series([target_row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value) and abs(float(value)) > 1e-12:
            association = float(np.sign(value))

            return association, column

    direction_columns = [
        (
            "transition_model_direction",
            {
                "increases_transition": 1.0,
                "decreases_transition": -1.0,
                "positive": 1.0,
                "negative": -1.0,
            },
        ),
        (
            "de_direction",
            {
                "higher_in_repop": 1.0,
                "lower_in_repop": -1.0,
                "up": 1.0,
                "down": -1.0,
            },
        ),
        (
            "escape_marker_direction",
            {
                "higher_in_escape_prone": 1.0,
                "lower_in_escape_prone": -1.0,
            },
        ),
        (
            "trajectory_direction",
            {
                "increasing": 1.0,
                "decreasing": -1.0,
            },
        ),
    ]

    for column, mapping in direction_columns:
        if column not in target_row.index:
            continue

        normalized = str(target_row[column]).strip().lower()

        if normalized in mapping:
            return mapping[normalized], column

    return 1.0, "default_positive_association"


def determine_intervention(
    target_row: pd.Series,
) -> tuple[str, float, float, str]:
    association_sign, evidence_column = infer_escape_association(target_row)

    if association_sign >= 0:
        intervention = "inhibit"
        perturbation_sign = -1.0
    else:
        intervention = "activate"
        perturbation_sign = 1.0

    return (
        intervention,
        perturbation_sign,
        association_sign,
        evidence_column,
    )


def create_gene_pathway_map(
    pathway_results: pd.DataFrame,
    available_feature_columns: list[str],
    comparison: str = DEFAULT_COMPARISON,
    maximum_false_discovery_rate: float = 0.25,
) -> dict[str, list[dict[str, object]]]:
    selected = pathway_results.loc[
        pathway_results["comparison"].astype(str).eq(comparison)
    ].copy()

    if "false_discovery_rate" in selected.columns:
        selected["false_discovery_rate"] = pd.to_numeric(
            selected["false_discovery_rate"],
            errors="coerce",
        )

        selected = selected.loc[
            selected["false_discovery_rate"].isna()
            | (selected["false_discovery_rate"] <= maximum_false_discovery_rate)
        ]

    score_column = None

    for candidate in [
        "normalized_enrichment_score",
        "enrichment_score",
        "score",
    ]:
        if candidate in selected.columns:
            score_column = candidate
            break

    gene_map: dict[str, list[dict[str, object]]] = {}

    for row in selected.itertuples(index=False):
        pathway = str(row.pathway).strip()
        feature = f"pathway__{sanitize_pathway_name(pathway)}"

        if feature not in available_feature_columns:
            continue

        score = 1.0

        if score_column is not None:
            raw_score = pd.to_numeric(
                pd.Series([getattr(row, score_column)]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(raw_score) and abs(float(raw_score)) > 1e-12:
                score = float(raw_score)

        genes = parse_gene_list(row.leading_edge_genes)

        for gene in genes:
            gene_map.setdefault(gene, []).append(
                {
                    "pathway": pathway,
                    "feature": feature,
                    "enrichment_score": score,
                    "pathway_sign": float(np.sign(score)),
                }
            )

    deduplicated: dict[str, list[dict[str, object]]] = {}

    for gene, pathway_entries in gene_map.items():
        by_feature: dict[str, dict[str, object]] = {}

        for entry in pathway_entries:
            feature = str(entry["feature"])
            current = by_feature.get(feature)

            if current is None or abs(float(entry["enrichment_score"])) > abs(
                float(current["enrichment_score"])
            ):
                by_feature[feature] = entry

        deduplicated[gene] = list(by_feature.values())

    return deduplicated


def calculate_confidence_score(
    priority_score: float,
    pathway_count: int,
    maximum_pathway_count: int,
    evidence_source_count: float | None = None,
) -> float:
    bounded_priority = float(np.clip(priority_score, 0.0, 1.0))

    if maximum_pathway_count > 0:
        coverage = pathway_count / maximum_pathway_count
    else:
        coverage = 0.0

    coverage = float(np.clip(coverage, 0.0, 1.0))

    if evidence_source_count is None or np.isnan(evidence_source_count):
        evidence_component = bounded_priority
    else:
        evidence_component = float(np.clip(evidence_source_count / 6.0, 0.0, 1.0))

    confidence = 0.50 * bounded_priority + 0.30 * coverage + 0.20 * evidence_component

    return float(np.clip(confidence, 0.0, 1.0))


def simulate_target_perturbation(
    features: pd.DataFrame,
    model: Pipeline,
    target_row: pd.Series,
    affected_pathways: list[dict[str, object]],
    perturbation_strength: float = (DEFAULT_PERTURBATION_STRENGTH),
    rap_effect_fraction: float = (DEFAULT_RAP_EFFECT_FRACTION),
    rap_column: str = DEFAULT_RAP_COLUMN,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if perturbation_strength <= 0:
        raise ValueError("perturbation_strength must be greater than zero")

    if not 0 <= rap_effect_fraction <= 1:
        raise ValueError("rap_effect_fraction must be between zero and one")

    (
        intervention,
        perturbation_sign,
        association_sign,
        direction_evidence,
    ) = determine_intervention(target_row)

    baseline_probabilities = model.predict_proba(features)[:, 1]

    perturbed_features = features.copy()
    affected_feature_names = []

    for pathway in affected_pathways:
        feature = str(pathway["feature"])

        if feature not in perturbed_features.columns:
            continue

        pathway_sign = float(pathway["pathway_sign"])

        perturbed_features[feature] = (
            perturbed_features[feature]
            + perturbation_sign * pathway_sign * perturbation_strength
        )

        affected_feature_names.append(feature)

    if rap_column in perturbed_features.columns:
        perturbed_features[rap_column] = (
            perturbed_features[rap_column]
            + perturbation_sign
            * association_sign
            * perturbation_strength
            * rap_effect_fraction
        )

    perturbed_probabilities = model.predict_proba(perturbed_features)[:, 1]

    probability_change = perturbed_probabilities - baseline_probabilities

    target_gene = str(target_row.get("gene", "")).strip().upper()

    cell_results = pd.DataFrame(
        {
            "cell_id": features.index.astype(str),
            "target": target_gene,
            "intervention": intervention,
            "baseline_escape_probability": (baseline_probabilities),
            "perturbed_escape_probability": (perturbed_probabilities),
            "delta_escape_probability": probability_change,
            "predicted_escape_reduction": (-probability_change),
        }
    )

    priority_score = float(
        pd.to_numeric(
            pd.Series([target_row.get("priority_score", 0.0)]),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    evidence_source_count_raw = pd.to_numeric(
        pd.Series(
            [
                target_row.get(
                    "evidence_source_count",
                    np.nan,
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    summary = {
        "target": target_gene,
        "intervention": intervention,
        "association_sign": association_sign,
        "direction_evidence": direction_evidence,
        "priority_score": priority_score,
        "evidence_source_count": (
            float(evidence_source_count_raw)
            if pd.notna(evidence_source_count_raw)
            else np.nan
        ),
        "affected_pathway_count": len(set(affected_feature_names)),
        "affected_pathways": ";".join(
            sorted(
                {
                    str(pathway["pathway"])
                    for pathway in affected_pathways
                    if str(pathway["feature"]) in affected_feature_names
                }
            )
        ),
        "mean_baseline_escape_probability": float(np.mean(baseline_probabilities)),
        "mean_perturbed_escape_probability": float(np.mean(perturbed_probabilities)),
        "mean_delta_escape_probability": float(np.mean(probability_change)),
        "mean_predicted_escape_reduction": float(-np.mean(probability_change)),
        "median_predicted_escape_reduction": float(-np.median(probability_change)),
        "maximum_predicted_escape_reduction": float(np.max(-probability_change)),
        "cells_with_reduced_escape_probability": int(np.sum(probability_change < 0)),
        "fraction_cells_with_reduced_escape_probability": float(
            np.mean(probability_change < 0)
        ),
        "perturbation_strength": perturbation_strength,
    }

    return cell_results, summary


def rank_interventions(
    perturbation_summary: pd.DataFrame,
) -> pd.DataFrame:
    if perturbation_summary.empty:
        return perturbation_summary.copy()

    ranked = perturbation_summary.copy()

    maximum_pathway_count = int(ranked["affected_pathway_count"].max())

    ranked["confidence_score"] = ranked.apply(
        lambda row: calculate_confidence_score(
            priority_score=float(row["priority_score"]),
            pathway_count=int(row["affected_pathway_count"]),
            maximum_pathway_count=maximum_pathway_count,
            evidence_source_count=(
                float(row["evidence_source_count"])
                if pd.notna(row["evidence_source_count"])
                else None
            ),
        ),
        axis=1,
    )

    ranked["intervention_score"] = (
        ranked["mean_predicted_escape_reduction"] * ranked["confidence_score"]
    )

    ranked = ranked.sort_values(
        [
            "intervention_score",
            "mean_predicted_escape_reduction",
            "priority_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "perturbation_rank",
        np.arange(1, len(ranked) + 1),
    )

    ranked["predicted_effect"] = np.select(
        [
            ranked["mean_predicted_escape_reduction"] > 0,
            ranked["mean_predicted_escape_reduction"] < 0,
        ],
        [
            "predicted_suppression",
            "predicted_worsening",
        ],
        default="minimal_change",
    )

    return ranked


def run_perturbation_analysis(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
    comparison: str = DEFAULT_COMPARISON,
    maximum_targets: int = DEFAULT_MAXIMUM_TARGETS,
    perturbation_strength: float = (DEFAULT_PERTURBATION_STRENGTH),
    rap_effect_fraction: float = (DEFAULT_RAP_EFFECT_FRACTION),
    random_state: int = DEFAULT_RANDOM_STATE,
    fitted_model: Pipeline | None = None,
    model_feature_columns: list[str] | None = None,
    model_name: str = "rap_plus_pathways_refit",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if maximum_targets < 1:
        raise ValueError("maximum_targets must be at least 1")

    validate_inputs(
        feature_table,
        target_rankings,
        pathway_results,
    )

    features, labels = prepare_model_data(feature_table)

    if fitted_model is None:
        model = create_model(random_state=random_state)
        model.fit(features, labels)
    else:
        model = fitted_model

        if model_feature_columns is None:
            raise ValueError("model_feature_columns are required with fitted_model")

        missing_model_features = [
            column for column in model_feature_columns if column not in features.columns
        ]

        if missing_model_features:
            raise ValueError(
                "Feature table is missing saved-model features: "
                f"{missing_model_features}"
            )

        extra_features = [
            column for column in features.columns if column not in model_feature_columns
        ]

        if extra_features:
            features = features.drop(columns=extra_features)

        features = features.loc[:, model_feature_columns]

    gene_pathway_map = create_gene_pathway_map(
        pathway_results,
        available_feature_columns=features.columns.tolist(),
        comparison=comparison,
    )

    selected_targets = target_rankings.copy()
    selected_targets["gene"] = (
        selected_targets["gene"].astype(str).str.strip().str.upper()
    )
    selected_targets["priority_score"] = pd.to_numeric(
        selected_targets["priority_score"],
        errors="coerce",
    )

    selected_targets = (
        selected_targets.dropna(subset=["gene", "priority_score"])
        .loc[selected_targets["gene"].ne("") & selected_targets["gene"].ne("NAN")]
        .sort_values(
            "priority_score",
            ascending=False,
        )
        .drop_duplicates("gene")
        .head(maximum_targets)
    )

    if selected_targets.empty:
        raise ValueError("No valid targets remained for perturbation")

    cell_tables = []
    summary_rows = []

    for _, target_row in selected_targets.iterrows():
        gene = str(target_row["gene"])
        affected_pathways = gene_pathway_map.get(
            gene,
            [],
        )

        cell_results, summary = simulate_target_perturbation(
            features,
            model,
            target_row,
            affected_pathways,
            perturbation_strength=(perturbation_strength),
            rap_effect_fraction=(rap_effect_fraction),
        )

        cell_tables.append(cell_results)
        summary_rows.append(summary)

    perturbation_predictions = pd.concat(
        cell_tables,
        ignore_index=True,
    )
    perturbation_predictions.insert(0, "model_name", model_name)

    perturbation_summary = pd.DataFrame(summary_rows)
    perturbation_summary.insert(0, "model_name", model_name)

    ranked_interventions = rank_interventions(perturbation_summary)

    return (
        perturbation_predictions,
        perturbation_summary,
        ranked_interventions,
    )


def save_results(
    perturbation_predictions: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    ranked_interventions: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    perturbation_predictions.to_csv(
        output_directory / "perturbation_predictions.csv",
        index=False,
    )

    perturbation_summary.to_csv(
        output_directory / "perturbation_summary.csv",
        index=False,
    )

    ranked_interventions.to_csv(
        output_directory / "top_predicted_interventions.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate target-directed changes in RAP and "
            "pathway activity and rank predicted suppressors "
            "of senescence escape."
        )
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_TABLE_PATH,
    )

    parser.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGET_RANKINGS_PATH,
    )

    parser.add_argument(
        "--pathways",
        type=Path,
        default=DEFAULT_PATHWAY_RESULTS_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT_PATH,
    )

    parser.add_argument(
        "--comparison",
        default=DEFAULT_COMPARISON,
    )

    parser.add_argument(
        "--maximum-targets",
        type=int,
        default=DEFAULT_MAXIMUM_TARGETS,
    )

    parser.add_argument(
        "--perturbation-strength",
        type=float,
        default=DEFAULT_PERTURBATION_STRENGTH,
    )

    parser.add_argument(
        "--rap-effect-fraction",
        type=float,
        default=DEFAULT_RAP_EFFECT_FRACTION,
    )

    args = parser.parse_args()

    for path in [
        args.features,
        args.targets,
        args.pathways,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    fitted_model, model_feature_columns, model_name = load_model_artifact(
        args.model_artifact
    )

    print(f"Using saved validated model: {model_name}")

    results = run_perturbation_analysis(
        pd.read_csv(args.features),
        pd.read_csv(args.targets),
        pd.read_csv(args.pathways),
        comparison=args.comparison,
        maximum_targets=args.maximum_targets,
        perturbation_strength=(args.perturbation_strength),
        rap_effect_fraction=(args.rap_effect_fraction),
        fitted_model=fitted_model,
        model_feature_columns=model_feature_columns,
        model_name=model_name,
    )

    save_results(
        *results,
        output_directory=args.output_directory,
    )

    rankings = results[2]

    print(f"Perturbation analysis completed for {len(rankings)} targets")
    print("Top predicted interventions:")

    for row in rankings.head(15).itertuples(index=False):
        print(
            f"{row.perturbation_rank}. "
            f"{row.target} ({row.intervention}): "
            f"escape reduction="
            f"{row.mean_predicted_escape_reduction:.4f}, "
            f"confidence={row.confidence_score:.3f}, "
            f"pathways={row.affected_pathway_count}"
        )

    print(f"Saved perturbation outputs to {args.output_directory}")


if __name__ == "__main__":
    main()
