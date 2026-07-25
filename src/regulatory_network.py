from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

DEFAULT_SCORE_COLUMN = "repopulation_associated_potential"
DEFAULT_TOP_REGULATORS_PER_CELL_LINE = 50
DEFAULT_MINIMUM_ABSOLUTE_CORRELATION = 0.1
DEFAULT_MINIMUM_TARGETS = 3

REQUIRED_OBS_COLUMNS = {
    "cell_line",
    "condition",
    "sample_id",
}


def validate_dataset(
    dataset: ad.AnnData,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> None:
    if dataset.n_obs == 0:
        raise ValueError("Dataset contains no cells")

    if dataset.n_vars == 0:
        raise ValueError("Dataset contains no genes")

    missing_columns = REQUIRED_OBS_COLUMNS.difference(dataset.obs.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required metadata columns: {sorted(missing_columns)}"
        )

    if score_column not in dataset.obs.columns:
        raise ValueError(f"Dataset does not contain score column: {score_column}")

    if dataset.raw is None:
        raise ValueError("Dataset does not contain normalized expression in .raw")


def validate_signature(signature: pd.DataFrame) -> None:
    required_columns = {
        "gene",
        "direction",
    }

    missing_columns = required_columns.difference(signature.columns)

    if missing_columns:
        raise ValueError(
            f"Signature is missing required columns: {sorted(missing_columns)}"
        )

    if signature.empty:
        raise ValueError("Signature contains no genes")

    valid_directions = {"up", "down"}

    observed_directions = set(signature["direction"].astype(str))

    if not observed_directions.issubset(valid_directions):
        raise ValueError("Signature directions must be 'up' or 'down'")


def load_transcription_factors(
    path: str | Path,
) -> list[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transcription-factor list does not exist: {path}")

    transcription_factors = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            gene = line.strip()

            if gene and not gene.startswith("#"):
                transcription_factors.append(gene)

    transcription_factors = list(dict.fromkeys(transcription_factors))

    if not transcription_factors:
        raise ValueError("Transcription-factor list contains no genes")

    return transcription_factors


def get_gene_expression(
    dataset: ad.AnnData,
    gene: str,
    cell_mask: np.ndarray | None = None,
) -> np.ndarray:
    raw_gene_names = dataset.raw.var_names.astype(str)

    if gene not in raw_gene_names:
        raise ValueError(f"Gene was not found in the dataset: {gene}")

    expression = dataset.raw[:, [gene]].X

    if sparse.issparse(expression):
        expression = expression.toarray()

    values = np.asarray(
        expression,
        dtype=np.float64,
    ).reshape(-1)

    if cell_mask is not None:
        values = values[cell_mask]

    return values


def residualize_within_samples(
    values: np.ndarray,
    sample_ids: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sample_ids = np.asarray(sample_ids).astype(str)

    if values.ndim != 1:
        raise ValueError("Values must be one-dimensional")

    if len(values) != len(sample_ids):
        raise ValueError("Values and sample identifiers must have equal length")

    residuals = np.zeros_like(values)

    for sample_id in np.unique(sample_ids):
        mask = sample_ids == sample_id
        sample_values = values[mask]

        residuals[mask] = sample_values - sample_values.mean()

    return residuals


def calculate_spearman_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if first.shape != second.shape:
        raise ValueError("Correlation vectors must have matching shapes")

    if first.ndim != 1:
        raise ValueError("Correlation vectors must be one-dimensional")

    if len(first) < 3:
        return 0.0, 1.0

    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0, 1.0

    correlation, p_value = spearmanr(
        first,
        second,
    )

    if not np.isfinite(correlation):
        correlation = 0.0

    if not np.isfinite(p_value):
        p_value = 1.0

    return float(correlation), float(p_value)


def rank_regulators_by_rap(
    dataset: ad.AnnData,
    transcription_factors: list[str],
    cell_line: str,
    score_column: str = DEFAULT_SCORE_COLUMN,
    top_regulators: int = DEFAULT_TOP_REGULATORS_PER_CELL_LINE,
) -> pd.DataFrame:
    validate_dataset(dataset, score_column)

    if top_regulators <= 0:
        raise ValueError("top_regulators must be greater than zero")

    cell_mask = dataset.obs["cell_line"].astype(str).to_numpy() == cell_line

    if cell_mask.sum() < 3:
        raise ValueError(f"Too few cells were found for cell line: {cell_line}")

    sample_ids = dataset.obs.loc[cell_mask, "sample_id"].astype(str).to_numpy()

    rap_scores = dataset.obs.loc[cell_mask, score_column].astype(float).to_numpy()

    residual_rap = residualize_within_samples(
        rap_scores,
        sample_ids,
    )

    available_genes = set(dataset.raw.var_names.astype(str))

    rows = []

    for transcription_factor in transcription_factors:
        if transcription_factor not in available_genes:
            continue

        expression = get_gene_expression(
            dataset,
            transcription_factor,
            cell_mask=cell_mask,
        )

        residual_expression = residualize_within_samples(
            expression,
            sample_ids,
        )

        correlation, p_value = calculate_spearman_correlation(
            residual_expression,
            residual_rap,
        )

        rows.append(
            {
                "cell_line": cell_line,
                "transcription_factor": transcription_factor,
                "rap_correlation": correlation,
                "absolute_rap_correlation": abs(correlation),
                "rap_p_value": p_value,
                "cell_count": int(cell_mask.sum()),
            }
        )

    if not rows:
        raise ValueError("No transcription factors were found in the dataset")

    ranked = pd.DataFrame(rows).sort_values(
        [
            "absolute_rap_correlation",
            "rap_p_value",
            "transcription_factor",
        ],
        ascending=[False, True, True],
    )

    return ranked.head(top_regulators).reset_index(drop=True)


def infer_regulatory_edges(
    dataset: ad.AnnData,
    regulators: pd.DataFrame,
    signature: pd.DataFrame,
    minimum_absolute_correlation: float = (DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
) -> pd.DataFrame:
    validate_signature(signature)

    if not 0 <= minimum_absolute_correlation <= 1:
        raise ValueError("minimum_absolute_correlation must be between zero and one")

    required_regulator_columns = {
        "cell_line",
        "transcription_factor",
        "rap_correlation",
    }

    missing_columns = required_regulator_columns.difference(regulators.columns)

    if missing_columns:
        raise ValueError(
            f"Regulator table is missing required columns: {sorted(missing_columns)}"
        )

    available_genes = set(dataset.raw.var_names.astype(str))

    signature_lookup = (
        signature.drop_duplicates("gene")
        .set_index("gene")["direction"]
        .astype(str)
        .to_dict()
    )

    signature_genes = [gene for gene in signature_lookup if gene in available_genes]

    rows = []

    for cell_line, group in regulators.groupby(
        "cell_line",
        observed=True,
    ):
        cell_mask = dataset.obs["cell_line"].astype(str).to_numpy() == str(cell_line)

        sample_ids = dataset.obs.loc[cell_mask, "sample_id"].astype(str).to_numpy()

        target_expression = {}

        for target_gene in signature_genes:
            expression = get_gene_expression(
                dataset,
                target_gene,
                cell_mask=cell_mask,
            )

            target_expression[target_gene] = residualize_within_samples(
                expression,
                sample_ids,
            )

        for regulator in group.itertuples(index=False):
            transcription_factor = str(regulator.transcription_factor)

            regulator_expression = get_gene_expression(
                dataset,
                transcription_factor,
                cell_mask=cell_mask,
            )

            regulator_expression = residualize_within_samples(
                regulator_expression,
                sample_ids,
            )

            for target_gene in signature_genes:
                if transcription_factor == target_gene:
                    continue

                correlation, p_value = calculate_spearman_correlation(
                    regulator_expression,
                    target_expression[target_gene],
                )

                if abs(correlation) < minimum_absolute_correlation:
                    continue

                target_direction = signature_lookup[target_gene]

                expected_sign = 1 if target_direction == "up" else -1

                rap_consistent = (
                    np.sign(correlation) * np.sign(regulator.rap_correlation)
                    == expected_sign
                )

                rows.append(
                    {
                        "cell_line": str(cell_line),
                        "transcription_factor": (transcription_factor),
                        "target_gene": target_gene,
                        "target_direction": target_direction,
                        "tf_target_correlation": correlation,
                        "absolute_tf_target_correlation": abs(correlation),
                        "tf_target_p_value": p_value,
                        "rap_correlation": float(regulator.rap_correlation),
                        "rap_consistent": bool(rap_consistent),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "cell_line",
            "transcription_factor",
            "target_gene",
            "target_direction",
            "tf_target_correlation",
            "absolute_tf_target_correlation",
            "tf_target_p_value",
            "rap_correlation",
            "rap_consistent",
        ],
    )


def summarize_regulators(
    regulators: pd.DataFrame,
    edges: pd.DataFrame,
    minimum_targets: int = DEFAULT_MINIMUM_TARGETS,
) -> pd.DataFrame:
    if minimum_targets <= 0:
        raise ValueError("minimum_targets must be greater than zero")

    if edges.empty:
        return pd.DataFrame(
            columns=[
                "cell_line",
                "transcription_factor",
                "rap_correlation",
                "absolute_rap_correlation",
                "target_count",
                "consistent_target_count",
                "consistency_fraction",
                "mean_absolute_target_correlation",
                "regulatory_score",
            ]
        )

    edge_summary = (
        edges.groupby(
            [
                "cell_line",
                "transcription_factor",
            ],
            observed=True,
        )
        .agg(
            target_count=(
                "target_gene",
                "nunique",
            ),
            consistent_target_count=(
                "rap_consistent",
                "sum",
            ),
            mean_absolute_target_correlation=(
                "absolute_tf_target_correlation",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = regulators.merge(
        edge_summary,
        on=[
            "cell_line",
            "transcription_factor",
        ],
        how="inner",
    )

    summary = summary.loc[summary["target_count"] >= minimum_targets].copy()

    summary["consistency_fraction"] = (
        summary["consistent_target_count"] / summary["target_count"]
    )

    summary["regulatory_score"] = (
        summary["absolute_rap_correlation"]
        * summary["mean_absolute_target_correlation"]
        * summary["consistency_fraction"]
        * np.log1p(summary["target_count"])
    )

    return summary.sort_values(
        [
            "regulatory_score",
            "absolute_rap_correlation",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def create_conserved_regulator_summary(
    regulator_summary: pd.DataFrame,
) -> pd.DataFrame:
    if regulator_summary.empty:
        return pd.DataFrame(
            columns=[
                "transcription_factor",
                "cell_lines_supported",
                "mean_regulatory_score",
                "mean_rap_correlation",
                "minimum_regulatory_score",
            ]
        )

    conserved = (
        regulator_summary.groupby(
            "transcription_factor",
            observed=True,
        )
        .agg(
            cell_lines_supported=(
                "cell_line",
                "nunique",
            ),
            mean_regulatory_score=(
                "regulatory_score",
                "mean",
            ),
            mean_rap_correlation=(
                "rap_correlation",
                "mean",
            ),
            minimum_regulatory_score=(
                "regulatory_score",
                "min",
            ),
        )
        .reset_index()
    )

    return conserved.sort_values(
        [
            "cell_lines_supported",
            "mean_regulatory_score",
            "minimum_regulatory_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_regulatory_network(
    dataset: ad.AnnData,
    signature: pd.DataFrame,
    transcription_factors: list[str],
    score_column: str = DEFAULT_SCORE_COLUMN,
    top_regulators: int = DEFAULT_TOP_REGULATORS_PER_CELL_LINE,
    minimum_absolute_correlation: float = (DEFAULT_MINIMUM_ABSOLUTE_CORRELATION),
    minimum_targets: int = DEFAULT_MINIMUM_TARGETS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_dataset(dataset, score_column)
    validate_signature(signature)

    cell_lines = sorted(dataset.obs["cell_line"].astype(str).unique().tolist())

    regulator_tables = []

    for cell_line in cell_lines:
        regulator_tables.append(
            rank_regulators_by_rap(
                dataset,
                transcription_factors,
                cell_line=cell_line,
                score_column=score_column,
                top_regulators=top_regulators,
            )
        )

    regulators = pd.concat(
        regulator_tables,
        ignore_index=True,
    )

    edges = infer_regulatory_edges(
        dataset,
        regulators,
        signature,
        minimum_absolute_correlation=(minimum_absolute_correlation),
    )

    regulator_summary = summarize_regulators(
        regulators,
        edges,
        minimum_targets=minimum_targets,
    )

    conserved_summary = create_conserved_regulator_summary(regulator_summary)

    return (
        regulators,
        edges,
        regulator_summary,
        conserved_summary,
    )


def save_results(
    regulators: pd.DataFrame,
    edges: pd.DataFrame,
    regulator_summary: pd.DataFrame,
    conserved_summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    regulators.to_csv(
        output_directory / "rap_regulator_correlations.csv",
        index=False,
    )

    edges.to_csv(
        output_directory / "regulatory_network_edges.csv",
        index=False,
    )

    regulator_summary.to_csv(
        output_directory / "regulator_summary.csv",
        index=False,
    )

    conserved_summary.to_csv(
        output_directory / "conserved_regulators.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prioritize transcription factors associated with "
            "Repopulation-Associated Potential."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_scored.h5ad"),
    )

    parser.add_argument(
        "--signature",
        type=Path,
        default=Path("results/tables/repopulation_signature.csv"),
    )

    parser.add_argument(
        "--tf-list",
        type=Path,
        default=Path("data/metadata/human_transcription_factors.txt"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/tables"),
    )

    parser.add_argument(
        "--top-regulators",
        type=int,
        default=DEFAULT_TOP_REGULATORS_PER_CELL_LINE,
    )

    parser.add_argument(
        "--minimum-correlation",
        type=float,
        default=DEFAULT_MINIMUM_ABSOLUTE_CORRELATION,
    )

    parser.add_argument(
        "--minimum-targets",
        type=int,
        default=DEFAULT_MINIMUM_TARGETS,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input}")

    if not args.signature.exists():
        raise FileNotFoundError(f"Signature file does not exist: {args.signature}")

    dataset = ad.read_h5ad(args.input)
    signature = pd.read_csv(args.signature)

    transcription_factors = load_transcription_factors(args.tf_list)

    (
        regulators,
        edges,
        regulator_summary,
        conserved_summary,
    ) = run_regulatory_network(
        dataset,
        signature,
        transcription_factors,
        top_regulators=args.top_regulators,
        minimum_absolute_correlation=(args.minimum_correlation),
        minimum_targets=args.minimum_targets,
    )

    save_results(
        regulators,
        edges,
        regulator_summary,
        conserved_summary,
        args.output_directory,
    )

    print(f"Regulator-cell-line associations: {len(regulators):,}")

    print(f"Candidate regulatory edges: {len(edges):,}")

    print(f"Prioritized regulator results: {len(regulator_summary):,}")

    print("Top conserved regulators:")

    for row in conserved_summary.head(15).itertuples(index=False):
        print(
            f"{row.transcription_factor}: "
            f"{row.cell_lines_supported} cell lines, "
            f"score={row.mean_regulatory_score:.4f}, "
            f"RAP correlation="
            f"{row.mean_rap_correlation:.3f}"
        )

    print(f"Saved regulatory-network results to {args.output_directory}")


if __name__ == "__main__":
    main()
