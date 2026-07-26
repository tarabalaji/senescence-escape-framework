from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_INPUT_PATH = Path("data/processed/trajectory_scored.h5ad")
DEFAULT_PATHWAY_RESULTS_PATH = Path("results/pathways/pathway_results_all.csv")
DEFAULT_REGULATOR_RESULTS_PATH = Path("results/tables/conserved_regulators.csv")
DEFAULT_OUTPUT_DATASET_PATH = Path("data/processed/transition_model.h5ad")
DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")

DEFAULT_STATE_COLUMN = "condition"
DEFAULT_CELL_LINE_COLUMN = "cell_line"
DEFAULT_RAP_COLUMN = "rap_score"
DEFAULT_PROBABILITY_COLUMN = "transition_probability"
DEFAULT_PREDICTION_COLUMN = "transition_prediction"

DEFAULT_POSITIVE_STATE = "REPOP"
DEFAULT_COMPARISON = "REPOP_vs_TIS"
DEFAULT_MINIMUM_PATHWAY_GENES = 2
DEFAULT_MAXIMUM_PATHWAYS = 10
DEFAULT_MAXIMUM_REGULATORS = 10
DEFAULT_RANDOM_STATE = 42


def validate_adata(
    adata: ad.AnnData,
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
) -> str:
    if adata.n_obs == 0:
        raise ValueError("AnnData object contains no cells")
    if adata.n_vars == 0:
        raise ValueError("AnnData object contains no genes")

    if state_column not in adata.obs.columns:
        if "condition" in adata.obs.columns:
            state_column = "condition"
        elif "state" in adata.obs.columns:
            state_column = "state"
        else:
            raise ValueError(
                "AnnData observations must contain either 'condition' or 'state'"
            )

    required = {cell_line_column, rap_column}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AnnData observations are missing columns: {sorted(missing)}")

    return state_column


def parse_gene_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value
    else:
        values = re.split(r"[;,|]", str(value))

    genes: list[str] = []
    seen: set[str] = set()
    for gene in values:
        normalized = str(gene).strip().upper()
        if normalized and normalized not in seen:
            genes.append(normalized)
            seen.add(normalized)
    return genes


def _first_existing_column(
    table: pd.DataFrame,
    candidates: Iterable[str],
    table_name: str,
) -> str:
    for column in candidates:
        if column in table.columns:
            return column
    raise ValueError(f"{table_name} must contain one of: {list(candidates)}")


def select_pathway_gene_sets(
    pathway_results: pd.DataFrame,
    comparison: str = DEFAULT_COMPARISON,
    maximum_pathways: int = DEFAULT_MAXIMUM_PATHWAYS,
    maximum_fdr: float = 0.25,
) -> dict[str, list[str]]:
    if maximum_pathways < 1:
        raise ValueError("maximum_pathways must be at least 1")

    required = {"comparison", "pathway", "leading_edge_genes"}
    missing = required.difference(pathway_results.columns)
    if missing:
        raise ValueError(f"Pathway results are missing columns: {sorted(missing)}")

    selected = pathway_results.loc[
        pathway_results["comparison"].astype(str).eq(comparison)
    ].copy()

    if "false_discovery_rate" in selected.columns:
        selected["false_discovery_rate"] = pd.to_numeric(
            selected["false_discovery_rate"], errors="coerce"
        )
        selected = selected.loc[selected["false_discovery_rate"] <= maximum_fdr]

    score_column = _first_existing_column(
        selected,
        ["normalized_enrichment_score", "enrichment_score", "score"],
        "Pathway results",
    )
    selected[score_column] = pd.to_numeric(selected[score_column], errors="coerce")
    selected = selected.dropna(subset=[score_column])
    selected["absolute_score"] = selected[score_column].abs()
    selected = selected.sort_values("absolute_score", ascending=False).head(
        maximum_pathways
    )

    gene_sets: dict[str, list[str]] = {}
    for row in selected.itertuples(index=False):
        pathway = str(row.pathway).strip()
        genes = parse_gene_list(row.leading_edge_genes)
        if pathway and genes:
            gene_sets[pathway] = genes
    return gene_sets


def select_regulators(
    regulator_results: pd.DataFrame,
    maximum_regulators: int = DEFAULT_MAXIMUM_REGULATORS,
) -> list[str]:
    if maximum_regulators < 1:
        raise ValueError("maximum_regulators must be at least 1")

    regulator_column = _first_existing_column(
        regulator_results,
        ["regulator", "transcription_factor", "gene"],
        "Regulator results",
    )
    score_column = _first_existing_column(
        regulator_results,
        ["priority_score", "mean_regulatory_score", "regulatory_score", "score"],
        "Regulator results",
    )

    selected = regulator_results[[regulator_column, score_column]].copy()
    selected["regulator"] = (
        selected[regulator_column].astype(str).str.strip().str.upper()
    )
    selected["score"] = pd.to_numeric(selected[score_column], errors="coerce")
    selected = selected.dropna(subset=["regulator", "score"])
    selected = selected.loc[selected["regulator"].ne("")]
    selected["absolute_score"] = selected["score"].abs()

    return (
        selected.sort_values("absolute_score", ascending=False)["regulator"]
        .drop_duplicates()
        .head(maximum_regulators)
        .tolist()
    )


def standardize_gene_expression(
    adata: ad.AnnData,
    genes: Iterable[str],
) -> pd.DataFrame:
    requested = list(
        dict.fromkeys(str(gene).strip().upper() for gene in genes if str(gene).strip())
    )

    variable_names = pd.Index(adata.var_names.astype(str))

    uppercase_to_original: dict[str, str] = {}
    for variable_name in variable_names:
        uppercase_to_original.setdefault(variable_name.upper(), variable_name)

    available = [gene for gene in requested if gene in uppercase_to_original]

    if not available:
        return pd.DataFrame(index=adata.obs_names)

    original_names = [uppercase_to_original[gene] for gene in available]

    # Subset genes before loading or converting the expression matrix.
    matrix = adata[:, original_names].X

    # Backed AnnData sparse datasets may need conversion to an
    # in-memory sparse matrix before normal SciPy operations.
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()

    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)

    matrix = matrix.astype(np.float32, copy=False)

    selected = pd.DataFrame(
        matrix,
        index=adata.obs_names,
        columns=available,
    )

    means = selected.mean(axis=0)
    standard_deviations = selected.std(axis=0, ddof=0).replace(0, 1.0)

    return (selected - means) / standard_deviations


def _sanitize_feature_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return sanitized or "unnamed"


def calculate_pathway_activity(
    adata: ad.AnnData,
    pathway_gene_sets: dict[str, list[str]],
    minimum_pathway_genes: int = DEFAULT_MINIMUM_PATHWAY_GENES,
) -> pd.DataFrame:
    if minimum_pathway_genes < 1:
        raise ValueError("minimum_pathway_genes must be at least 1")

    features = pd.DataFrame(index=adata.obs_names)
    for pathway, genes in pathway_gene_sets.items():
        standardized = standardize_gene_expression(adata, genes)
        if standardized.shape[1] < minimum_pathway_genes:
            continue
        column = f"pathway__{_sanitize_feature_name(pathway)}"
        features[column] = standardized.mean(axis=1)
    return features


def calculate_regulator_features(
    adata: ad.AnnData,
    regulators: Iterable[str],
) -> pd.DataFrame:
    standardized = standardize_gene_expression(adata, regulators)
    features = pd.DataFrame(index=adata.obs_names)
    for regulator in standardized.columns:
        features[f"regulator__{regulator}"] = standardized[regulator]
    return features


def build_transition_features(
    adata: ad.AnnData,
    pathway_gene_sets: dict[str, list[str]],
    regulators: list[str],
    rap_column: str = DEFAULT_RAP_COLUMN,
    minimum_pathway_genes: int = DEFAULT_MINIMUM_PATHWAY_GENES,
) -> pd.DataFrame:
    if rap_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {rap_column}")

    rap = pd.to_numeric(adata.obs[rap_column], errors="coerce")
    if rap.isna().any():
        raise ValueError(f"{rap_column} must contain numeric values")

    features = pd.DataFrame(
        {"rap_score": rap.to_numpy(dtype=float)},
        index=adata.obs_names,
    )
    return pd.concat(
        [
            features,
            calculate_pathway_activity(
                adata,
                pathway_gene_sets,
                minimum_pathway_genes,
            ),
            calculate_regulator_features(adata, regulators),
        ],
        axis=1,
    )


def create_transition_labels(
    adata: ad.AnnData,
    state_column: str = DEFAULT_STATE_COLUMN,
    positive_state: str = DEFAULT_POSITIVE_STATE,
) -> pd.Series:
    if state_column not in adata.obs.columns:
        if "condition" in adata.obs.columns:
            state_column = "condition"
        elif "state" in adata.obs.columns:
            state_column = "state"
        else:
            raise ValueError(
                "AnnData observations must contain either 'condition' or 'state'"
            )

    states = adata.obs[state_column].astype(str).str.upper()
    labels = states.eq(str(positive_state).upper()).astype(int)
    labels.name = "transition_label"

    if labels.nunique() < 2:
        raise ValueError("Transition labels must contain both classes")

    return labels


def evaluate_predictions(
    labels: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    y_prediction = (
        (y_probability >= 0.5).astype(int)
        if predictions is None
        else np.asarray(predictions, dtype=int)
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_prediction)),
        "precision": float(precision_score(y_true, y_prediction, zero_division=0)),
        "recall": float(recall_score(y_true, y_prediction, zero_division=0)),
        "f1": float(f1_score(y_true, y_prediction, zero_division=0)),
    }

    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_probability)
        )
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def _create_model(random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
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


def run_leave_one_cell_line_out(
    features: pd.DataFrame,
    labels: pd.Series,
    cell_lines: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_labels = labels.reindex(features.index)
    aligned_cell_lines = cell_lines.reindex(features.index)

    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for held_out in sorted(aligned_cell_lines.astype(str).unique()):
        test_mask = aligned_cell_lines.astype(str) == held_out
        train_mask = ~test_mask
        train_labels = aligned_labels.loc[train_mask]
        test_labels = aligned_labels.loc[test_mask]

        if train_labels.nunique() < 2:
            raise ValueError("Training fold contains only one class")

        model = _create_model(random_state)
        model.fit(features.loc[train_mask], train_labels)
        probabilities = model.predict_proba(features.loc[test_mask])[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(int)

        fold_predictions = pd.DataFrame(
            {
                "cell_line": held_out,
                "transition_label": test_labels.to_numpy(dtype=int),
                "transition_probability": probabilities,
                "transition_prediction": predicted_labels,
            },
            index=features.loc[test_mask].index,
        )
        predictions.append(fold_predictions)

        metrics = evaluate_predictions(
            test_labels,
            probabilities,
            predicted_labels,
        )
        metrics["held_out_cell_line"] = held_out
        metrics["n_test_cells"] = int(test_mask.sum())
        metric_rows.append(metrics)

    prediction_table = pd.concat(predictions).loc[features.index]
    metrics_table = pd.DataFrame(metric_rows)
    return prediction_table, metrics_table


def fit_final_model(
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame]:
    aligned_labels = labels.reindex(features.index)
    if aligned_labels.nunique() < 2:
        raise ValueError("Transition labels contain only one class")

    model = _create_model(random_state)
    model.fit(features, aligned_labels)

    estimator = model.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "feature": features.columns,
            "coefficient": estimator.coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient", ascending=False
    ).reset_index(drop=True)
    return model, coefficients


def create_prediction_summary(
    adata: ad.AnnData,
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
) -> pd.DataFrame:
    if state_column not in adata.obs.columns:
        if "condition" in adata.obs.columns:
            state_column = "condition"
        elif "state" in adata.obs.columns:
            state_column = "state"
        else:
            raise ValueError(
                "AnnData observations must contain either 'condition' or 'state'"
            )

    required = {cell_line_column, probability_column}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AnnData observations are missing columns: {sorted(missing)}")

    return (
        adata.obs.groupby(
            [cell_line_column, state_column],
            observed=True,
        )
        .agg(
            cell_count=(probability_column, "size"),
            mean_transition_probability=(probability_column, "mean"),
            median_transition_probability=(probability_column, "median"),
            standard_deviation=(probability_column, "std"),
        )
        .reset_index()
    )


def run_transition_model(
    adata: ad.AnnData,
    pathway_results: pd.DataFrame,
    regulator_results: pd.DataFrame,
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
    comparison: str = DEFAULT_COMPARISON,
    maximum_pathways: int = DEFAULT_MAXIMUM_PATHWAYS,
    maximum_regulators: int = DEFAULT_MAXIMUM_REGULATORS,
    minimum_pathway_genes: int = DEFAULT_MINIMUM_PATHWAY_GENES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    ad.AnnData,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    state_column = validate_adata(
        adata,
        state_column=state_column,
        cell_line_column=cell_line_column,
        rap_column=rap_column,
    )

    pathway_gene_sets = select_pathway_gene_sets(
        pathway_results,
        comparison=comparison,
        maximum_pathways=maximum_pathways,
    )
    regulators = select_regulators(
        regulator_results,
        maximum_regulators=maximum_regulators,
    )
    features = build_transition_features(
        adata,
        pathway_gene_sets,
        regulators,
        rap_column=rap_column,
        minimum_pathway_genes=minimum_pathway_genes,
    )
    labels = create_transition_labels(adata, state_column=state_column)

    predictions, metrics = run_leave_one_cell_line_out(
        features,
        labels,
        adata.obs[cell_line_column],
        random_state=random_state,
    )
    _, coefficients = fit_final_model(
        features,
        labels,
        random_state=random_state,
    )

    scored = adata.copy()
    scored.obs[DEFAULT_PROBABILITY_COLUMN] = predictions[
        DEFAULT_PROBABILITY_COLUMN
    ].reindex(scored.obs_names)
    scored.obs[DEFAULT_PREDICTION_COLUMN] = predictions[
        DEFAULT_PREDICTION_COLUMN
    ].reindex(scored.obs_names)
    scored.obs["transition_label"] = labels.reindex(scored.obs_names)

    summary = create_prediction_summary(
        scored,
        state_column=state_column,
        cell_line_column=cell_line_column,
    )

    feature_table = features.copy()
    feature_table.insert(0, "cell_id", feature_table.index.astype(str))
    feature_table["transition_label"] = labels.to_numpy()

    return (
        scored,
        metrics,
        coefficients,
        summary,
        feature_table.reset_index(drop=True),
    )


def save_results(
    scored: ad.AnnData,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    summary: pd.DataFrame,
    feature_table: pd.DataFrame,
    output_dataset_path: str | Path,
    output_directory: str | Path,
) -> None:
    output_dataset_path = Path(output_dataset_path)
    output_directory = Path(output_directory)
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    scored.write_h5ad(output_dataset_path)
    metrics.to_csv(output_directory / "transition_model_metrics.csv", index=False)
    coefficients.to_csv(
        output_directory / "transition_model_coefficients.csv", index=False
    )
    summary.to_csv(output_directory / "transition_prediction_summary.csv", index=False)
    feature_table.to_csv(output_directory / "transition_feature_table.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a senescence-to-repopulation transition model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--pathways", type=Path, default=DEFAULT_PATHWAY_RESULTS_PATH)
    parser.add_argument(
        "--regulators", type=Path, default=DEFAULT_REGULATOR_RESULTS_PATH
    )
    parser.add_argument(
        "--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET_PATH
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--maximum-pathways", type=int, default=DEFAULT_MAXIMUM_PATHWAYS
    )
    parser.add_argument(
        "--maximum-regulators", type=int, default=DEFAULT_MAXIMUM_REGULATORS
    )
    args = parser.parse_args()

    for path in [args.input, args.pathways, args.regulators]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    results = run_transition_model(
        ad.read_h5ad(args.input),
        pd.read_csv(args.pathways),
        pd.read_csv(args.regulators),
        maximum_pathways=args.maximum_pathways,
        maximum_regulators=args.maximum_regulators,
    )
    save_results(
        *results,
        output_dataset_path=args.output_dataset,
        output_directory=args.output_directory,
    )

    print(f"Transition model completed with {len(results[1])} held-out cell-line folds")
    print(f"Saved transition-model outputs to {args.output_directory}")


if __name__ == "__main__":
    main()
