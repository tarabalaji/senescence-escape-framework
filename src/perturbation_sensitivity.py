from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.perturbation import run_perturbation_analysis

DEFAULT_CONFIG = Path("config/project_config.json")


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_artifact(path: str | Path) -> tuple[object, list[str], str]:
    artifact = joblib.load(path)
    required = {"pipeline", "feature_columns", "model_name"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return (
        artifact["pipeline"],
        [str(column) for column in artifact["feature_columns"]],
        str(artifact["model_name"]),
    )


def top_k_overlap(reference: list[str], candidate: list[str], k: int) -> float:
    reference_set = set(reference[:k])
    candidate_set = set(candidate[:k])
    if not reference_set:
        return float("nan")
    return len(reference_set & candidate_set) / len(reference_set)


def run_sensitivity_grid(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    model_name: str,
    strengths: list[float],
    rap_effect_fractions: list[float],
    maximum_targets: int,
    comparison: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rankings = []
    reference_ranking = None

    for strength in strengths:
        for rap_fraction in rap_effect_fractions:
            _, _, ranking = run_perturbation_analysis(
                feature_table,
                target_rankings,
                pathway_results,
                comparison=comparison,
                maximum_targets=maximum_targets,
                perturbation_strength=float(strength),
                rap_effect_fraction=float(rap_fraction),
                fitted_model=model,
                model_feature_columns=feature_columns,
                model_name=model_name,
            )
            ranking = ranking.copy()
            ranking["perturbation_strength"] = float(strength)
            ranking["rap_effect_fraction"] = float(rap_fraction)
            all_rankings.append(ranking)

            if strength == 1.0 and rap_fraction == 0.25:
                reference_ranking = ranking["target"].astype(str).tolist()

    combined = pd.concat(all_rankings, ignore_index=True)

    if reference_ranking is None:
        reference_subset = combined.sort_values(
            ["perturbation_strength", "rap_effect_fraction"]
        ).iloc[:maximum_targets]
        reference_ranking = reference_subset["target"].astype(str).tolist()

    stability_rows = []
    for (strength, rap_fraction), group in combined.groupby(
        ["perturbation_strength", "rap_effect_fraction"]
    ):
        ordered = group.sort_values("perturbation_rank")
        genes = ordered["target"].astype(str).tolist()
        stability_rows.append(
            {
                "perturbation_strength": strength,
                "rap_effect_fraction": rap_fraction,
                "top_k": top_k,
                "top_k_overlap_with_reference": top_k_overlap(
                    reference_ranking, genes, top_k
                ),
                "top_target": genes[0] if genes else None,
                "top_target_effect": (
                    float(ordered.iloc[0]["mean_predicted_escape_reduction"])
                    if not ordered.empty
                    else np.nan
                ),
            }
        )

    return combined, pd.DataFrame(stability_rows)


def bootstrap_rank_stability(
    feature_table: pd.DataFrame,
    target_rankings: pd.DataFrame,
    pathway_results: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    model_name: str,
    iterations: int,
    maximum_targets: int,
    comparison: str,
    top_k: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    counts: dict[str, int] = {}
    rank_totals: dict[str, float] = {}

    for iteration in range(iterations):
        indices = rng.integers(0, len(feature_table), size=len(feature_table))
        sampled = feature_table.iloc[indices].copy()

        if "cell_id" in sampled.columns:
            sampled["cell_id"] = [
                f"{cell_id}__boot_{iteration}_{position}"
                for position, cell_id in enumerate(sampled["cell_id"].astype(str))
            ]

        _, _, ranking = run_perturbation_analysis(
            sampled,
            target_rankings,
            pathway_results,
            comparison=comparison,
            maximum_targets=maximum_targets,
            perturbation_strength=1.0,
            rap_effect_fraction=0.25,
            fitted_model=model,
            model_feature_columns=feature_columns,
            model_name=model_name,
        )

        for row in ranking.head(top_k).itertuples(index=False):
            gene = str(row.target)
            counts[gene] = counts.get(gene, 0) + 1
            rank_totals[gene] = rank_totals.get(gene, 0.0) + float(
                row.perturbation_rank
            )

    rows = []
    for gene, count in counts.items():
        rows.append(
            {
                "target": gene,
                "top_k_frequency": count / iterations,
                "mean_rank_when_selected": rank_totals[gene] / count,
                "bootstrap_iterations": iterations,
                "top_k": top_k,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["top_k_frequency", "mean_rank_when_selected"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate perturbation-strength and target-rank robustness."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Run the parameter grid without cell bootstrap resampling.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    settings = config["perturbation_sensitivity"]

    features = pd.read_csv(paths["feature_table"])
    targets = pd.read_csv(paths["target_rankings"])
    pathways = pd.read_csv(paths["pathway_results"])
    model, feature_columns, model_name = load_model_artifact(paths["model_artifact"])

    grid, stability = run_sensitivity_grid(
        features,
        targets,
        pathways,
        model,
        feature_columns,
        model_name,
        settings["strengths"],
        settings["rap_effect_fractions"],
        int(settings["maximum_targets"]),
        settings["comparison"],
        int(settings["top_k_stability"]),
    )

    output_directory = Path(paths["tables"])
    output_directory.mkdir(parents=True, exist_ok=True)
    grid.to_csv(
        output_directory / "perturbation_sensitivity_grid.csv",
        index=False,
    )
    stability.to_csv(
        output_directory / "perturbation_sensitivity_stability.csv",
        index=False,
    )

    if not args.skip_bootstrap:
        bootstrap = bootstrap_rank_stability(
            features,
            targets,
            pathways,
            model,
            feature_columns,
            model_name,
            int(settings["bootstrap_iterations"]),
            int(settings["maximum_targets"]),
            settings["comparison"],
            int(settings["top_k_stability"]),
            int(config["random_seed"]),
        )
        bootstrap.to_csv(
            output_directory / "perturbation_bootstrap_stability.csv",
            index=False,
        )

    print("Saved perturbation sensitivity and robustness outputs.")
    print(
        "Minimum top-k overlap across parameter settings: "
        f"{stability['top_k_overlap_with_reference'].min():.3f}"
    )


if __name__ == "__main__":
    main()
