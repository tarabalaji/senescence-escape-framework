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
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITERATIONS = 2_000
DEFAULT_REGULARIZATION = 1.0
DEFAULT_MAX_REGULATORS = 10
DEFAULT_MAX_PATHWAYS = 10

DEFAULT_STATE_COLUMN = "condition"
DEFAULT_CELL_LINE_COLUMN = "cell_line"
DEFAULT_RAP_COLUMN = "repopulation_associated_potential"

TRAINING_STATES = {
    "TIS": 0,
    "REPOP": 1,
}


def validate_adata(
    adata: ad.AnnData,
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
) -> None:
    if adata.n_obs == 0:
        raise ValueError("AnnData object contains no cells")

    if adata.n_vars == 0:
        raise ValueError("AnnData object contains no genes")

    required_obs_columns = {
        state_column,
        cell_line_column,
        rap_column,
    }

    missing_columns = required_obs_columns.difference(adata.obs.columns)

    if missing_columns:
        raise ValueError(
            f"AnnData observations are missing columns: {sorted(missing_columns)}"
        )

    states = set(adata.obs[state_column].astype(str).str.upper())

    missing_states = set(TRAINING_STATES).difference(states)

    if missing_states:
        raise ValueError(
            f"AnnData object is missing training states: {sorted(missing_states)}"
        )

    cell_lines = adata.obs[cell_line_column].astype(str).dropna().unique()

    if len(cell_lines) < 2:
        raise ValueError(
            "At least two cell lines are required for "
            "leave-one-cell-line-out validation"
        )

    rap_values = pd.to_numeric(
        adata.obs[rap_column],
        errors="coerce",
    )

    if rap_values.isna().any():
        raise ValueError(f"{rap_column} contains missing or nonnumeric values")


def validate_pathway_results(
    pathway_results: pd.DataFrame,
) -> None:
    required_columns = {
        "comparison",
        "pathway",
        "normalized_enrichment_score",
        "false_discovery_rate",
        "leading_edge_genes",
    }

    missing_columns = required_columns.difference(pathway_results.columns)

    if missing_columns:
        raise ValueError(
            f"Pathway results are missing columns: {sorted(missing_columns)}"
        )


def validate_regulator_results(
    regulator_results: pd.DataFrame,
) -> None:
    regulator_column = identify_regulator_column(regulator_results)

    if regulator_column is None:
        raise ValueError(
            "Regulator results must contain one of these columns: "
            "regulator, transcription_factor, tf, or gene"
        )


def identify_regulator_column(
    regulator_results: pd.DataFrame,
) -> str | None:
    candidates = (
        "regulator",
        "transcription_factor",
        "tf",
        "gene",
    )

    for candidate in candidates:
        if candidate in regulator_results.columns:
            return candidate

    return None


def parse_gene_list(
    value: object,
) -> list[str]:
    if pd.isna(value):
        return []

    genes = re.split(
        r"[;,|/\s]+",
        str(value),
    )

    cleaned = []

    for gene in genes:
        gene = gene.strip().upper()

        if gene and gene not in cleaned:
            cleaned.append(gene)

    return cleaned


def normalize_gene_names(
    genes: Iterable[str],
) -> list[str]:
    normalized = []

    for gene in genes:
        gene_name = str(gene).strip().upper()

        if gene_name and gene_name not in normalized:
            normalized.append(gene_name)

    return normalized


def select_pathway_gene_sets(
    pathway_results: pd.DataFrame,
    comparison: str = "REPOP_vs_TIS",
    maximum_pathways: int = DEFAULT_MAX_PATHWAYS,
    maximum_false_discovery_rate: float = 0.05,
    require_positive_enrichment: bool = True,
) -> dict[str, list[str]]:
    validate_pathway_results(pathway_results)

    if maximum_pathways <= 0:
        raise ValueError("maximum_pathways must be greater than zero")

    if not 0 < maximum_false_discovery_rate <= 1:
        raise ValueError("maximum_false_discovery_rate must be between zero and one")

    selected = pathway_results.loc[
        pathway_results["comparison"].astype(str) == comparison
    ].copy()

    selected["normalized_enrichment_score"] = pd.to_numeric(
        selected["normalized_enrichment_score"],
        errors="coerce",
    )

    selected["false_discovery_rate"] = pd.to_numeric(
        selected["false_discovery_rate"],
        errors="coerce",
    )

    selected = selected.dropna(
        subset=[
            "normalized_enrichment_score",
            "false_discovery_rate",
        ]
    )

    selected = selected.loc[
        selected["false_discovery_rate"] <= maximum_false_discovery_rate
    ].copy()

    if require_positive_enrichment:
        selected = selected.loc[selected["normalized_enrichment_score"] > 0].copy()

    selected = selected.sort_values(
        [
            "false_discovery_rate",
            "normalized_enrichment_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    gene_sets: dict[str, list[str]] = {}

    for row in selected.itertuples(index=False):
        pathway_name = str(row.pathway)

        if pathway_name in gene_sets:
            continue

        genes = parse_gene_list(row.leading_edge_genes)

        if genes:
            gene_sets[pathway_name] = genes

        if len(gene_sets) >= maximum_pathways:
            break

    return gene_sets


def select_regulators(
    regulator_results: pd.DataFrame,
    maximum_regulators: int = DEFAULT_MAX_REGULATORS,
) -> list[str]:
    validate_regulator_results(regulator_results)

    if maximum_regulators <= 0:
        raise ValueError("maximum_regulators must be greater than zero")

    regulator_column = identify_regulator_column(regulator_results)

    ranked = regulator_results.copy()

    score_candidates = (
        "priority_score",
        "score",
        "conservation_score",
        "mean_absolute_correlation",
        "absolute_rap_correlation",
        "rap_correlation",
    )

    score_column = next(
        (column for column in score_candidates if column in ranked.columns),
        None,
    )

    if score_column is not None:
        ranked[score_column] = pd.to_numeric(
            ranked[score_column],
            errors="coerce",
        )

        if score_column == "rap_correlation":
            ranked["_ranking_score"] = ranked[score_column].abs()
        else:
            ranked["_ranking_score"] = ranked[score_column]

        ranked = ranked.sort_values(
            "_ranking_score",
            ascending=False,
            na_position="last",
        )

    regulators = normalize_gene_names(ranked[regulator_column].tolist())

    return regulators[:maximum_regulators]


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

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    return matrix, available_genes


def standardize_gene_expression(
    matrix: np.ndarray,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expression matrix must be two-dimensional")

    if matrix.shape[1] == 0:
        return matrix.copy()

    means = np.nanmean(
        matrix,
        axis=0,
    )

    standard_deviations = np.nanstd(
        matrix,
        axis=0,
    )

    standard_deviations[standard_deviations == 0] = 1.0

    standardized = (matrix - means) / standard_deviations

    return np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def calculate_pathway_activity(
    adata: ad.AnnData,
    pathway_gene_sets: dict[str, list[str]],
    minimum_genes: int = 2,
) -> pd.DataFrame:
    if minimum_genes <= 0:
        raise ValueError("minimum_genes must be greater than zero")

    activities = pd.DataFrame(index=adata.obs_names)

    for pathway_name, genes in pathway_gene_sets.items():
        matrix, available_genes = get_expression_matrix(
            adata,
            genes,
        )

        if len(available_genes) < minimum_genes:
            continue

        standardized = standardize_gene_expression(matrix)

        column_name = (
            "pathway__"
            + re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                pathway_name,
            )
            .strip("_")
            .lower()
        )

        activities[column_name] = standardized.mean(axis=1)

    return activities


def calculate_regulator_features(
    adata: ad.AnnData,
    regulators: list[str],
) -> pd.DataFrame:
    matrix, available_genes = get_expression_matrix(
        adata,
        regulators,
    )

    features = pd.DataFrame(index=adata.obs_names)

    if len(available_genes) == 0:
        return features

    standardized = standardize_gene_expression(matrix)

    for gene_index, gene in enumerate(available_genes):
        features[f"regulator__{gene.upper()}"] = standardized[:, gene_index]

    return features


def build_transition_features(
    adata: ad.AnnData,
    pathway_gene_sets: dict[str, list[str]],
    regulators: list[str],
    rap_column: str = DEFAULT_RAP_COLUMN,
    minimum_pathway_genes: int = 2,
) -> pd.DataFrame:
    if rap_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {rap_column}")

    features = pd.DataFrame(index=adata.obs_names)

    features["rap_score"] = pd.to_numeric(
        adata.obs[rap_column],
        errors="coerce",
    ).to_numpy()

    pathway_features = calculate_pathway_activity(
        adata,
        pathway_gene_sets,
        minimum_genes=minimum_pathway_genes,
    )

    regulator_features = calculate_regulator_features(
        adata,
        regulators,
    )

    features = features.join(
        pathway_features,
        how="left",
    )

    features = features.join(
        regulator_features,
        how="left",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(features.median(numeric_only=True))

    features = features.fillna(0.0)

    if features.shape[1] < 2:
        raise ValueError(
            "Transition model requires RAP plus at least "
            "one pathway or regulator feature"
        )

    return features


def create_transition_labels(
    adata: ad.AnnData,
    state_column: str = DEFAULT_STATE_COLUMN,
) -> pd.Series:
    states = adata.obs[state_column].astype(str).str.upper()

    labels = states.map(TRAINING_STATES)

    labels.name = "transition_label"

    return labels


def create_model(
    regularization: float = DEFAULT_REGULARIZATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Pipeline:
    if regularization <= 0:
        raise ValueError("regularization must be greater than zero")

    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be greater than zero")

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    max_iter=maximum_iterations,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate_predictions(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(np.unique(true_labels)) < 2:
        raise ValueError("Evaluation requires both TIS and REPOP cells")

    predicted_labels = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                true_labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                true_labels,
                probabilities,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                true_labels,
                probabilities,
            )
        ),
    }


def run_leave_one_cell_line_out(
    features: pd.DataFrame,
    labels: pd.Series,
    cell_lines: pd.Series,
    regularization: float = DEFAULT_REGULARIZATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[pd.Series, pd.DataFrame]:
    if not features.index.equals(labels.index):
        labels = labels.reindex(features.index)

    if not features.index.equals(cell_lines.index):
        cell_lines = cell_lines.reindex(features.index)

    training_mask = labels.notna()

    training_features = features.loc[training_mask]

    training_labels = labels.loc[training_mask].astype(int)

    training_cell_lines = cell_lines.loc[training_mask].astype(str)

    unique_cell_lines = sorted(training_cell_lines.unique())

    if len(unique_cell_lines) < 2:
        raise ValueError("At least two cell lines are required")

    out_of_fold_probabilities = pd.Series(
        np.nan,
        index=features.index,
        name="transition_probability_oof",
        dtype=float,
    )

    metric_rows = []

    for held_out_cell_line in unique_cell_lines:
        test_mask = training_cell_lines == held_out_cell_line

        train_mask = ~test_mask

        x_train = training_features.loc[train_mask]

        y_train = training_labels.loc[train_mask]

        x_test = training_features.loc[test_mask]

        y_test = training_labels.loc[test_mask]

        if y_train.nunique() < 2:
            raise ValueError("Training fold does not contain both classes")

        if y_test.nunique() < 2:
            raise ValueError(
                f"Held-out cell line {held_out_cell_line} does not contain both classes"
            )

        model = create_model(
            regularization=regularization,
            random_state=random_state,
            maximum_iterations=maximum_iterations,
        )

        model.fit(
            x_train,
            y_train,
        )

        probabilities = model.predict_proba(x_test)[:, 1]

        out_of_fold_probabilities.loc[x_test.index] = probabilities

        metrics = evaluate_predictions(
            y_test.to_numpy(),
            probabilities,
        )

        metric_rows.append(
            {
                "held_out_cell_line": (held_out_cell_line),
                "training_cells": len(x_train),
                "testing_cells": len(x_test),
                **metrics,
            }
        )

    metrics_table = pd.DataFrame(metric_rows)

    return (
        out_of_fold_probabilities,
        metrics_table,
    )


def fit_final_model(
    features: pd.DataFrame,
    labels: pd.Series,
    regularization: float = DEFAULT_REGULARIZATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[
    Pipeline,
    pd.Series,
    pd.DataFrame,
]:
    training_mask = labels.notna()

    x_train = features.loc[training_mask]

    y_train = labels.loc[training_mask].astype(int)

    model = create_model(
        regularization=regularization,
        random_state=random_state,
        maximum_iterations=maximum_iterations,
    )

    model.fit(
        x_train,
        y_train,
    )

    all_probabilities = pd.Series(
        model.predict_proba(features)[:, 1],
        index=features.index,
        name="transition_probability",
    )

    classifier = model.named_steps["classifier"]

    coefficient_table = pd.DataFrame(
        {
            "feature": features.columns,
            "coefficient": (classifier.coef_[0]),
        }
    )

    coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()

    coefficient_table = coefficient_table.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return (
        model,
        all_probabilities,
        coefficient_table,
    )


def create_prediction_summary(
    adata: ad.AnnData,
    probability_column: str = ("transition_probability"),
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
) -> pd.DataFrame:
    if probability_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {probability_column}")

    summary_data = adata.obs[
        [
            state_column,
            cell_line_column,
            probability_column,
        ]
    ].copy()

    summary_data[probability_column] = pd.to_numeric(
        summary_data[probability_column],
        errors="coerce",
    )

    summary = (
        summary_data.groupby(
            [
                cell_line_column,
                state_column,
            ],
            observed=True,
        )
        .agg(
            cells=(
                probability_column,
                "size",
            ),
            mean_transition_probability=(
                probability_column,
                "mean",
            ),
            median_transition_probability=(
                probability_column,
                "median",
            ),
            standard_deviation=(
                probability_column,
                "std",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            cell_line_column,
            state_column,
        ]
    ).reset_index(drop=True)


def run_transition_model(
    adata: ad.AnnData,
    pathway_results: pd.DataFrame,
    regulator_results: pd.DataFrame,
    state_column: str = DEFAULT_STATE_COLUMN,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    rap_column: str = DEFAULT_RAP_COLUMN,
    maximum_pathways: int = DEFAULT_MAX_PATHWAYS,
    maximum_regulators: int = DEFAULT_MAX_REGULATORS,
    minimum_pathway_genes: int = 2,
    regularization: float = DEFAULT_REGULARIZATION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    ad.AnnData,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_adata(
        adata,
        state_column=state_column,
        cell_line_column=cell_line_column,
        rap_column=rap_column,
    )

    pathway_gene_sets = select_pathway_gene_sets(
        pathway_results,
        maximum_pathways=maximum_pathways,
    )

    regulators = select_regulators(
        regulator_results,
        maximum_regulators=maximum_regulators,
    )

    features = build_transition_features(
        adata,
        pathway_gene_sets=pathway_gene_sets,
        regulators=regulators,
        rap_column=rap_column,
        minimum_pathway_genes=minimum_pathway_genes,
    )

    labels = create_transition_labels(
        adata,
        state_column=state_column,
    )

    cell_lines = adata.obs[cell_line_column].astype(str)

    out_of_fold_probabilities, metrics = run_leave_one_cell_line_out(
        features,
        labels,
        cell_lines,
        regularization=regularization,
        random_state=random_state,
    )

    (
        model,
        all_probabilities,
        coefficients,
    ) = fit_final_model(
        features,
        labels,
        regularization=regularization,
        random_state=random_state,
    )

    scored = adata.copy()

    scored.obs["transition_probability"] = all_probabilities.reindex(
        scored.obs_names
    ).to_numpy()

    scored.obs["transition_probability_oof"] = out_of_fold_probabilities.reindex(
        scored.obs_names
    ).to_numpy()

    scored.uns["transition_model_features"] = list(features.columns)

    scored.uns["transition_model_pathways"] = {
        pathway: genes for pathway, genes in pathway_gene_sets.items()
    }

    scored.uns["transition_model_regulators"] = regulators

    scored.uns["transition_model_description"] = (
        "Logistic regression estimating REPOP-like "
        "transition probability from RAP, pathway activity, "
        "and regulator expression."
    )

    feature_table = features.copy()

    feature_table.insert(
        0,
        "cell_id",
        feature_table.index,
    )

    feature_table["state"] = scored.obs[state_column].astype(str).to_numpy()

    feature_table["cell_line"] = scored.obs[cell_line_column].astype(str).to_numpy()

    feature_table["transition_probability"] = scored.obs[
        "transition_probability"
    ].to_numpy()

    feature_table["transition_probability_oof"] = scored.obs[
        "transition_probability_oof"
    ].to_numpy()

    summary = create_prediction_summary(
        scored,
        probability_column="transition_probability",
        state_column=state_column,
        cell_line_column=cell_line_column,
    )

    return (
        scored,
        metrics,
        coefficients,
        summary,
        feature_table,
    )


def save_results(
    scored_adata: ad.AnnData,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    summary: pd.DataFrame,
    feature_table: pd.DataFrame,
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

    metrics.to_csv(
        output_directory / "transition_model_metrics.csv",
        index=False,
    )

    coefficients.to_csv(
        output_directory / "transition_model_coefficients.csv",
        index=False,
    )

    summary.to_csv(
        output_directory / "transition_probability_summary.csv",
        index=False,
    )

    feature_table.to_csv(
        output_directory / "transition_model_features.csv",
        index=False,
    )


def find_default_regulator_file() -> Path:
    candidates = [
        Path("results/tables/conserved_regulators.csv"),
        Path("results/tables/prioritized_regulators.csv"),
        Path("results/tables/regulator_summary.csv"),
        Path("results/tables/regulatory_network_regulators.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def find_default_adata_file() -> Path:
    candidates = [
        Path("data/processed/combined_scored.h5ad"),
        Path("data/processed/rap_scored.h5ad"),
        Path("data/processed/escape_index.h5ad"),
        Path("data/processed/scored_data.h5ad"),
        Path("data/processed/processed.h5ad"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate TIS-to-REPOP transition probabilities "
            "using RAP, pathway, and regulator features."
        )
    )

    parser.add_argument(
        "--adata",
        type=Path,
        default=find_default_adata_file(),
    )

    parser.add_argument(
        "--pathways",
        type=Path,
        default=Path("results/pathways/pathway_results_significant.csv"),
    )

    parser.add_argument(
        "--regulators",
        type=Path,
        default=find_default_regulator_file(),
    )

    parser.add_argument(
        "--output-adata",
        type=Path,
        default=Path("data/processed/transition_scored.h5ad"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/tables"),
    )

    parser.add_argument(
        "--state-column",
        default=DEFAULT_STATE_COLUMN,
    )

    parser.add_argument(
        "--cell-line-column",
        default=DEFAULT_CELL_LINE_COLUMN,
    )

    parser.add_argument(
        "--rap-column",
        default=DEFAULT_RAP_COLUMN,
    )

    parser.add_argument(
        "--maximum-pathways",
        type=int,
        default=DEFAULT_MAX_PATHWAYS,
    )

    parser.add_argument(
        "--maximum-regulators",
        type=int,
        default=DEFAULT_MAX_REGULATORS,
    )

    parser.add_argument(
        "--minimum-pathway-genes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=DEFAULT_REGULARIZATION,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    args = parser.parse_args()

    for input_path in (
        args.adata,
        args.pathways,
        args.regulators,
    ):
        if not input_path.exists():
            raise FileNotFoundError(f"Required input does not exist: {input_path}")

    adata = ad.read_h5ad(args.adata)

    pathway_results = pd.read_csv(args.pathways)

    regulator_results = pd.read_csv(args.regulators)

    (
        scored,
        metrics,
        coefficients,
        summary,
        feature_table,
    ) = run_transition_model(
        adata,
        pathway_results,
        regulator_results,
        state_column=args.state_column,
        cell_line_column=args.cell_line_column,
        rap_column=args.rap_column,
        maximum_pathways=args.maximum_pathways,
        maximum_regulators=args.maximum_regulators,
        minimum_pathway_genes=(args.minimum_pathway_genes),
        regularization=args.regularization,
        random_state=args.random_state,
    )

    save_results(
        scored,
        metrics,
        coefficients,
        summary,
        feature_table,
        args.output_adata,
        args.output_directory,
    )

    print(f"Transition features: {len(scored.uns['transition_model_features'])}")

    print(f"Pathways included: {len(scored.uns['transition_model_pathways'])}")

    print(f"Regulators included: {len(scored.uns['transition_model_regulators'])}")

    print("Leave-one-cell-line-out performance:")

    for row in metrics.itertuples(index=False):
        print(
            f"{row.held_out_cell_line}: "
            f"ROC AUC={row.roc_auc:.3f}, "
            f"AP={row.average_precision:.3f}, "
            f"balanced accuracy="
            f"{row.balanced_accuracy:.3f}, "
            f"Brier={row.brier_score:.3f}"
        )

    print("Top transition features:")

    for row in coefficients.head(10).itertuples(index=False):
        print(f"{row.feature}: coefficient={row.coefficient:.4f}")

    print(f"Saved transition-model results to {args.output_directory}")


if __name__ == "__main__":
    main()
