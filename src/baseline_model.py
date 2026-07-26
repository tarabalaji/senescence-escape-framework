from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transition_model import evaluate_predictions

DEFAULT_INPUT_DATASET_PATH = Path("data/processed/transition_model.h5ad")
DEFAULT_FEATURE_TABLE_PATH = Path("results/tables/transition_feature_table.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("results/tables")

DEFAULT_CELL_LINE_COLUMN = "cell_line"
DEFAULT_LABEL_COLUMN = "transition_label"
DEFAULT_CELL_ID_COLUMN = "cell_id"
DEFAULT_RANDOM_STATE = 42


def validate_inputs(
    adata: ad.AnnData,
    feature_table: pd.DataFrame,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    cell_id_column: str = DEFAULT_CELL_ID_COLUMN,
) -> pd.DataFrame:
    if adata.n_obs == 0:
        raise ValueError("AnnData object contains no cells")

    if cell_line_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations do not contain {cell_line_column}")

    required_columns = {cell_id_column, label_column}
    missing_columns = required_columns.difference(feature_table.columns)

    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {sorted(missing_columns)}")

    if feature_table[cell_id_column].duplicated().any():
        raise ValueError("Feature table contains duplicate cell IDs")

    indexed = feature_table.set_index(cell_id_column).copy()
    indexed.index = indexed.index.astype(str)

    missing_cells = pd.Index(adata.obs_names.astype(str)).difference(indexed.index)

    if len(missing_cells) > 0:
        raise ValueError(
            "Feature table is missing cells present in AnnData: "
            f"{missing_cells[:5].tolist()}"
        )

    indexed = indexed.reindex(adata.obs_names.astype(str))

    labels = pd.to_numeric(indexed[label_column], errors="coerce")

    if labels.isna().any():
        raise ValueError(f"{label_column} must contain numeric values")

    if not labels.isin([0, 1]).all():
        raise ValueError(f"{label_column} must contain only 0 and 1")

    if labels.nunique() < 2:
        raise ValueError("Transition labels must contain both classes")

    feature_columns = [column for column in indexed.columns if column != label_column]

    if not feature_columns:
        raise ValueError("Feature table contains no model features")

    numeric_features = indexed[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_features.isna().any().any():
        invalid_columns = numeric_features.columns[
            numeric_features.isna().any()
        ].tolist()

        raise ValueError(
            f"Model features must be numeric and nonmissing: {invalid_columns}"
        )

    indexed.loc[:, feature_columns] = numeric_features

    return indexed


def identify_feature_groups(
    feature_table: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> dict[str, list[str]]:
    feature_columns = [
        column for column in feature_table.columns if column != label_column
    ]

    rap_columns = [
        column
        for column in feature_columns
        if column
        in {
            "rap_score",
            "repopulation_associated_potential",
        }
    ]

    pathway_columns = [
        column for column in feature_columns if str(column).startswith("pathway__")
    ]

    regulator_columns = [
        column for column in feature_columns if str(column).startswith("regulator__")
    ]

    groups: dict[str, list[str]] = {}

    if rap_columns:
        groups["rap_only"] = rap_columns

    if pathway_columns:
        groups["pathways_only"] = pathway_columns

    if regulator_columns:
        groups["regulators_only"] = regulator_columns

    if rap_columns and pathway_columns:
        groups["rap_plus_pathways"] = rap_columns + pathway_columns

    if rap_columns and regulator_columns:
        groups["rap_plus_regulators"] = rap_columns + regulator_columns

    biological_columns = pathway_columns + regulator_columns

    if biological_columns:
        groups["pathways_plus_regulators"] = biological_columns

    groups["full_model"] = feature_columns

    return groups


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


def evaluate_majority_baseline(
    labels: pd.Series,
    cell_lines: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []

    aligned_cell_lines = cell_lines.reindex(labels.index)

    for held_out in sorted(aligned_cell_lines.astype(str).unique()):
        test_mask = aligned_cell_lines.astype(str).eq(held_out)
        train_mask = ~test_mask

        train_labels = labels.loc[train_mask]
        test_labels = labels.loc[test_mask]

        positive_probability = float(train_labels.mean())
        majority_prediction = int(positive_probability >= 0.5)

        probabilities = np.full(
            test_mask.sum(),
            positive_probability,
            dtype=float,
        )

        predictions = np.full(
            test_mask.sum(),
            majority_prediction,
            dtype=int,
        )

        metrics = evaluate_predictions(
            test_labels,
            probabilities,
            predictions,
        )

        metrics.update(
            {
                "model": "majority_class",
                "held_out_cell_line": held_out,
                "n_features": 0,
                "n_test_cells": int(test_mask.sum()),
            }
        )

        metric_rows.append(metrics)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "cell_id": labels.index[test_mask].astype(str),
                    "model": "majority_class",
                    "held_out_cell_line": held_out,
                    "transition_label": test_labels.to_numpy(dtype=int),
                    "transition_probability": probabilities,
                    "transition_prediction": predictions,
                }
            )
        )

    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def evaluate_logistic_baseline(
    features: pd.DataFrame,
    labels: pd.Series,
    cell_lines: pd.Series,
    model_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.shape[1] == 0:
        raise ValueError(f"{model_name} contains no features")

    prediction_rows = []
    metric_rows = []

    aligned_labels = labels.reindex(features.index)
    aligned_cell_lines = cell_lines.reindex(features.index)

    for held_out in sorted(aligned_cell_lines.astype(str).unique()):
        test_mask = aligned_cell_lines.astype(str).eq(held_out)
        train_mask = ~test_mask

        train_labels = aligned_labels.loc[train_mask]
        test_labels = aligned_labels.loc[test_mask]

        if train_labels.nunique() < 2:
            raise ValueError(f"Training fold for {held_out} contains only one class")

        model = create_model(random_state=random_state)
        model.fit(features.loc[train_mask], train_labels)

        probabilities = model.predict_proba(features.loc[test_mask])[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        metrics = evaluate_predictions(
            test_labels,
            probabilities,
            predictions,
        )

        metrics.update(
            {
                "model": model_name,
                "held_out_cell_line": held_out,
                "n_features": int(features.shape[1]),
                "n_test_cells": int(test_mask.sum()),
            }
        )

        metric_rows.append(metrics)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "cell_id": features.index[test_mask].astype(str),
                    "model": model_name,
                    "held_out_cell_line": held_out,
                    "transition_label": test_labels.to_numpy(dtype=int),
                    "transition_probability": probabilities,
                    "transition_prediction": predictions,
                }
            )
        )

    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def fit_baseline_coefficients(
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    model = create_model(random_state=random_state)
    model.fit(features, labels.reindex(features.index))

    estimator = model.named_steps["model"]

    coefficients = pd.DataFrame(
        {
            "model": model_name,
            "feature": features.columns.astype(str),
            "coefficient": estimator.coef_[0],
        }
    )

    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()

    return coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def summarize_baseline_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    available_metrics = [
        column for column in metric_columns if column in metrics.columns
    ]

    aggregation = {
        metric: ["mean", "std", "min", "max"] for metric in available_metrics
    }

    summary = metrics.groupby("model").agg(aggregation)

    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]

    fold_counts = metrics.groupby("model").size().rename("folds")
    feature_counts = metrics.groupby("model")["n_features"].max()

    summary = summary.join(fold_counts).join(feature_counts.rename("n_features"))

    return (
        summary.reset_index()
        .sort_values(
            "roc_auc_mean",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def run_baseline_analysis(
    adata: ad.AnnData,
    feature_table: pd.DataFrame,
    cell_line_column: str = DEFAULT_CELL_LINE_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    cell_id_column: str = DEFAULT_CELL_ID_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    indexed = validate_inputs(
        adata,
        feature_table,
        cell_line_column=cell_line_column,
        label_column=label_column,
        cell_id_column=cell_id_column,
    )

    labels = indexed[label_column].astype(int)
    features = indexed.drop(columns=[label_column]).astype(float)

    cell_lines = pd.Series(
        adata.obs[cell_line_column].astype(str).to_numpy(),
        index=adata.obs_names.astype(str),
        name=cell_line_column,
    )

    feature_groups = identify_feature_groups(
        indexed,
        label_column=label_column,
    )

    prediction_tables = []
    metric_tables = []
    coefficient_tables = []

    majority_predictions, majority_metrics = evaluate_majority_baseline(
        labels,
        cell_lines,
    )

    prediction_tables.append(majority_predictions)
    metric_tables.append(majority_metrics)

    for model_name, selected_columns in feature_groups.items():
        model_features = features[selected_columns]

        predictions, metrics = evaluate_logistic_baseline(
            model_features,
            labels,
            cell_lines,
            model_name=model_name,
            random_state=random_state,
        )

        coefficients = fit_baseline_coefficients(
            model_features,
            labels,
            model_name=model_name,
            random_state=random_state,
        )

        prediction_tables.append(predictions)
        metric_tables.append(metrics)
        coefficient_tables.append(coefficients)

    all_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    all_metrics = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    all_coefficients = pd.concat(
        coefficient_tables,
        ignore_index=True,
    )

    summary = summarize_baseline_metrics(all_metrics)

    return (
        all_metrics,
        all_predictions,
        all_coefficients,
        summary,
    )


def save_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        output_directory / "baseline_model_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        output_directory / "baseline_model_predictions.csv",
        index=False,
    )

    coefficients.to_csv(
        output_directory / "baseline_model_coefficients.csv",
        index=False,
    )

    summary.to_csv(
        output_directory / "baseline_model_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the full senescence-transition model against "
            "simpler leave-one-cell-line-out baselines."
        )
    )

    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=DEFAULT_INPUT_DATASET_PATH,
    )

    parser.add_argument(
        "--feature-table",
        type=Path,
        default=DEFAULT_FEATURE_TABLE_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    args = parser.parse_args()

    for path in [
        args.input_dataset,
        args.feature_table,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    results = run_baseline_analysis(
        ad.read_h5ad(args.input_dataset),
        pd.read_csv(args.feature_table),
    )

    save_results(
        *results,
        output_directory=args.output_directory,
    )

    metrics, _, _, summary = results

    print(
        "Baseline analysis completed with "
        f"{metrics['model'].nunique()} models and "
        f"{metrics['held_out_cell_line'].nunique()} held-out folds"
    )

    if not summary.empty:
        print("Model ranking by mean ROC-AUC:")

        for row in summary.itertuples(index=False):
            print(
                f"{row.model}: "
                f"ROC-AUC={row.roc_auc_mean:.3f}, "
                f"balanced accuracy={row.balanced_accuracy_mean:.3f}"
            )

    print(f"Saved baseline-model outputs to {args.output_directory}")


if __name__ == "__main__":
    main()
