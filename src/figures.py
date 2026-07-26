from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scanpy as sc
except ImportError:
    sc = None


LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PATHWAYS_DIR = RESULTS_DIR / "pathways"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT / "data" / "processed"

FIGURE_DPI = 300
TOP_N = 15

CONDITION_COLUMN_CANDIDATES = (
    "condition",
    "group",
    "state",
    "sample_condition",
    "treatment",
    "phenotype",
)

CELL_LINE_COLUMN_CANDIDATES = (
    "cell_line",
    "cellline",
    "cell_type",
    "line",
)

RAP_COLUMN_CANDIDATES = (
    "rap_score",
    "RAP_score",
    "rap",
    "repopulation_associated_program_score",
    "escape_index",
)

ESCAPE_PROBABILITY_COLUMN_CANDIDATES = (
    "transition_probability_oof",
    "transition_probability",
    "escape_probability",
    "predicted_escape_probability",
    "escape_prob",
    "rap_plus_pathways_probability",
    "probability",
)

ESCAPE_LABEL_COLUMN_CANDIDATES = (
    "escape_prone_status",
    "escape_prone",
    "escape_prone_label",
    "predicted_escape_prone",
    "escape_label",
    "is_escape_prone",
)

PSEUDOTIME_COLUMN_CANDIDATES = (
    "escape_pseudotime",
    "dpt_pseudotime",
    "pseudotime",
    "trajectory_pseudotime",
    "latent_time",
)

CELL_CYCLE_COLUMN_CANDIDATES = (
    "phase",
    "cell_cycle_phase",
    "S_score",
    "G2M_score",
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", output_path.relative_to(ROOT))


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def read_csv_candidates(paths: Sequence[Path]) -> pd.DataFrame | None:
    path = first_existing(paths)
    if path is None:
        return None

    try:
        LOGGER.info("Reading %s", path.relative_to(ROOT))
        return pd.read_csv(path)
    except Exception as error:
        LOGGER.warning("Could not read %s: %s", path, error)
        return None


def load_plotting_anndata():
    if sc is None:
        LOGGER.warning(
            "scanpy is not installed, so UMAP and trajectory figures will be skipped."
        )
        return None

    candidates = (
        DATA_DIR / "escape_prone_scored.h5ad",
        DATA_DIR / "trajectory_scored.h5ad",
        DATA_DIR / "transition_model.h5ad",
        DATA_DIR / "transition_scored.h5ad",
        DATA_DIR / "combined_reduced.h5ad",
        DATA_DIR / "combined_scored.h5ad",
        DATA_DIR / "combined_preprocessed.h5ad",
    )
    path = first_existing(candidates)
    if path is None:
        LOGGER.warning("No processed .h5ad file was found for cell-level figures.")
        return None

    LOGGER.info("Loading %s", path.relative_to(ROOT))
    try:
        adata = sc.read_h5ad(path, backed="r")
    except Exception as error:
        LOGGER.warning("Could not load %s: %s", path, error)
        return None

    if "X_umap" not in adata.obsm:
        LOGGER.warning("%s does not contain X_umap; UMAP panels will be skipped.", path)
    return adata


def sample_anndata_for_plotting(adata, maximum_cells: int = 100_000):
    if adata is None or adata.n_obs <= maximum_cells:
        return adata

    rng = np.random.default_rng(42)
    selected = np.sort(rng.choice(adata.n_obs, size=maximum_cells, replace=False))
    LOGGER.info(
        "Sampling %d of %d cells for plotting.",
        maximum_cells,
        adata.n_obs,
    )
    return adata[selected].to_memory()


def plot_umap_metadata(adata) -> None:
    if adata is None or "X_umap" not in adata.obsm:
        return

    columns = list(adata.obs.columns)
    condition = first_present(columns, CONDITION_COLUMN_CANDIDATES)
    cell_line = first_present(columns, CELL_LINE_COLUMN_CANDIDATES)
    rap_score = first_present(columns, RAP_COLUMN_CANDIDATES)
    escape_probability = first_present(
        columns,
        ESCAPE_PROBABILITY_COLUMN_CANDIDATES,
    )
    escape_label = first_present(columns, ESCAPE_LABEL_COLUMN_CANDIDATES)
    pseudotime = first_present(columns, PSEUDOTIME_COLUMN_CANDIDATES)

    selected = [
        column
        for column in (
            condition,
            cell_line,
            rap_score,
            escape_probability,
            escape_label,
            pseudotime,
        )
        if column is not None
    ]

    if not selected:
        LOGGER.warning(
            "UMAP coordinates exist, but no recognized metadata columns were found."
        )
        return

    for column in selected:
        try:
            axis = sc.pl.umap(
                adata,
                color=column,
                size=5,
                alpha=0.75,
                frameon=False,
                show=False,
                return_fig=False,
                title=column.replace("_", " ").title(),
            )
            fig = axis.figure
            save_figure(fig, f"umap_{column}.png")
        except Exception as error:
            LOGGER.warning("Could not plot UMAP for %s: %s", column, error)


def plot_trajectory(adata) -> None:
    if adata is None or "X_umap" not in adata.obsm:
        return

    pseudotime = first_present(
        list(adata.obs.columns),
        PSEUDOTIME_COLUMN_CANDIDATES,
    )
    if pseudotime is None:
        LOGGER.warning("No recognized pseudotime column was found.")
        return

    try:
        axis = sc.pl.umap(
            adata,
            color=pseudotime,
            size=5,
            alpha=0.8,
            frameon=False,
            show=False,
            return_fig=False,
            title="Senescence-to-escape pseudotime",
        )
        save_figure(axis.figure, "trajectory_pseudotime_umap.png")
    except Exception as error:
        LOGGER.warning("Could not plot trajectory UMAP: %s", error)

    condition = first_present(
        list(adata.obs.columns),
        CONDITION_COLUMN_CANDIDATES,
    )
    if condition is None:
        return

    frame = adata.obs[[condition, pseudotime]].copy()
    frame[pseudotime] = pd.to_numeric(frame[pseudotime], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        return

    order = (
        frame.groupby(condition, observed=True)[pseudotime]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    values = [
        frame.loc[frame[condition] == group, pseudotime].to_numpy() for group in order
    ]
    axis.boxplot(values, tick_labels=order, showfliers=False)
    axis.set_xlabel("Cell state")
    axis.set_ylabel("Pseudotime")
    axis.set_title("Pseudotime distribution across cell states")
    axis.tick_params(axis="x", rotation=30)
    save_figure(fig, "trajectory_pseudotime_by_condition.png")


def plot_escape_probability_distribution(adata) -> None:
    if adata is None:
        return

    columns = list(adata.obs.columns)
    probability = first_present(
        columns,
        ESCAPE_PROBABILITY_COLUMN_CANDIDATES,
    )
    if probability is None:
        LOGGER.warning("No recognized escape-probability column was found.")
        return

    condition = first_present(columns, CONDITION_COLUMN_CANDIDATES)
    frame_columns = [probability]
    if condition is not None:
        frame_columns.append(condition)

    frame = adata.obs[frame_columns].copy()
    frame[probability] = pd.to_numeric(frame[probability], errors="coerce")
    frame = frame.dropna(subset=[probability])
    if frame.empty:
        return

    fig, axis = plt.subplots(figsize=(8, 5))

    if condition is None:
        axis.hist(frame[probability], bins=40, density=True, alpha=0.8)
    else:
        groups = frame[condition].dropna().astype(str).unique()
        for group in sorted(groups):
            values = frame.loc[
                frame[condition].astype(str) == group,
                probability,
            ]
            axis.hist(
                values,
                bins=40,
                density=True,
                histtype="step",
                linewidth=2,
                label=group,
            )
        axis.legend(frameon=False)

    axis.set_xlabel("Predicted escape probability")
    axis.set_ylabel("Density")
    axis.set_title("Distribution of predicted escape probability")
    save_figure(fig, "escape_probability_distribution.png")


def select_de_comparison(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    if "comparison" in result.columns:
        preferred = result[
            result["comparison"]
            .astype(str)
            .str.contains(
                "REPOP.*TIS|TIS.*REPOP",
                case=False,
                regex=True,
                na=False,
            )
        ]
        if not preferred.empty:
            result = preferred

    return result


def detect_de_columns(data: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    columns = list(data.columns)

    gene_column = first_present(
        columns,
        ("gene", "gene_symbol", "names", "feature"),
    )
    effect_column = first_present(
        columns,
        (
            "log2_fold_change",
            "logfoldchange",
            "logfoldchanges",
            "avg_log2FC",
            "effect",
            "differential_expression_effect",
        ),
    )
    significance_column = first_present(
        columns,
        (
            "adjusted_p_value",
            "p_value_adj",
            "pvals_adj",
            "padj",
            "false_discovery_rate",
            "q_value",
            "p_value",
            "pvals",
        ),
    )
    return gene_column, effect_column, significance_column


def plot_de_volcano() -> None:
    data = read_csv_candidates(
        (
            TABLES_DIR / "differential_expression_all.csv",
            TABLES_DIR / "differential_expression_significant.csv",
            RESULTS_DIR / "differential_expression" / "differential_expression_all.csv",
            RESULTS_DIR
            / "differential_expression"
            / "differential_expression_significant.csv",
        )
    )
    if data is None or data.empty:
        LOGGER.warning("No differential-expression table was found.")
        return

    data = select_de_comparison(data)
    gene_column, effect_column, significance_column = detect_de_columns(data)

    if effect_column is None or significance_column is None:
        LOGGER.warning(
            "Could not identify fold-change and adjusted-p-value columns in the DE table."
        )
        return

    frame = data.copy()
    frame[effect_column] = pd.to_numeric(frame[effect_column], errors="coerce")
    frame[significance_column] = pd.to_numeric(
        frame[significance_column],
        errors="coerce",
    )
    frame = frame.dropna(subset=[effect_column, significance_column])
    frame = frame[frame[significance_column] >= 0]
    if frame.empty:
        return

    minimum_positive = np.nextafter(0, 1)
    frame["minus_log10_significance"] = -np.log10(
        frame[significance_column].clip(lower=minimum_positive)
    )
    frame["significant"] = (frame[significance_column] < 0.05) & (
        frame[effect_column].abs() >= 1.0
    )

    fig, axis = plt.subplots(figsize=(8, 6))
    nonsignificant = frame[~frame["significant"]]
    significant = frame[frame["significant"]]

    axis.scatter(
        nonsignificant[effect_column],
        nonsignificant["minus_log10_significance"],
        s=10,
        alpha=0.35,
        label="Other genes",
    )
    axis.scatter(
        significant[effect_column],
        significant["minus_log10_significance"],
        s=12,
        alpha=0.7,
        label="FDR < 0.05 and |effect| ≥ 1",
    )

    axis.axvline(-1.0, linestyle="--", linewidth=1)
    axis.axvline(1.0, linestyle="--", linewidth=1)
    axis.axhline(-np.log10(0.05), linestyle="--", linewidth=1)

    if gene_column is not None:
        label_frame = frame.assign(
            label_score=frame["minus_log10_significance"] * frame[effect_column].abs()
        ).nlargest(12, "label_score")
        for _, row in label_frame.iterrows():
            axis.annotate(
                str(row[gene_column]),
                (row[effect_column], row["minus_log10_significance"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    axis.set_xlabel("Differential-expression effect")
    axis.set_ylabel(r"$-\log_{10}$(adjusted p-value)")
    axis.set_title("Differential expression: repopulating versus senescent cells")
    axis.legend(frameon=False)
    save_figure(fig, "differential_expression_volcano.png")


def plot_pathway_enrichment() -> None:
    conserved = read_csv_candidates(
        (
            PATHWAYS_DIR / "conserved_pathways.csv",
            TABLES_DIR / "conserved_pathways.csv",
        )
    )
    significant = read_csv_candidates(
        (
            PATHWAYS_DIR / "pathway_results_significant.csv",
            TABLES_DIR / "pathway_results_significant.csv",
        )
    )

    data = conserved if conserved is not None and not conserved.empty else significant
    if data is None or data.empty:
        LOGGER.warning("No pathway-enrichment table was found.")
        return

    pathway_column = first_present(
        list(data.columns),
        ("pathway", "term", "name"),
    )
    score_column = first_present(
        list(data.columns),
        (
            "mean_normalized_enrichment_score",
            "normalized_enrichment_score",
            "enrichment_score",
        ),
    )

    if pathway_column is None or score_column is None:
        LOGGER.warning("Could not identify pathway and enrichment-score columns.")
        return

    frame = data.copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame.dropna(subset=[score_column])
    if frame.empty:
        return

    if "comparison" in frame.columns:
        preferred = frame[
            frame["comparison"]
            .astype(str)
            .str.contains(
                "REPOP.*TIS|TIS.*REPOP",
                case=False,
                regex=True,
                na=False,
            )
        ]
        if not preferred.empty:
            frame = preferred

    frame["absolute_score"] = frame[score_column].abs()
    frame = frame.nlargest(TOP_N, "absolute_score").sort_values(score_column)

    fig, axis = plt.subplots(figsize=(9, max(5, len(frame) * 0.42)))
    axis.barh(frame[pathway_column].astype(str), frame[score_column])
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Normalized enrichment score")
    axis.set_ylabel("Pathway")
    axis.set_title("Pathways associated with senescence escape")
    save_figure(fig, "pathway_enrichment.png")


def plot_target_priority() -> None:
    data = read_csv_candidates(
        (
            TABLES_DIR / "target_priority_rankings.csv",
            RESULTS_DIR / "target_prioritization" / "target_priority_rankings.csv",
        )
    )
    if data is None or data.empty:
        return

    required = {"gene", "priority_score"}
    if not required.issubset(data.columns):
        LOGGER.warning("Target-priority table is missing %s.", sorted(required))
        return

    frame = data.nlargest(TOP_N, "priority_score").sort_values("priority_score")
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(frame["gene"], frame["priority_score"])
    axis.set_xlabel("Integrated target-priority score")
    axis.set_ylabel("Gene")
    axis.set_title("Top targets supported across computational evidence")
    save_figure(fig, "target_priority_ranking.png")


def plot_predicted_interventions() -> None:
    data = read_csv_candidates(
        (
            TABLES_DIR / "top_predicted_interventions.csv",
            RESULTS_DIR / "perturbation" / "top_predicted_interventions.csv",
        )
    )
    if data is None or data.empty:
        return

    required = {
        "target",
        "intervention",
        "mean_predicted_escape_reduction",
    }
    if not required.issubset(data.columns):
        LOGGER.warning("Intervention table is missing %s.", sorted(required))
        return

    frame = data.nlargest(
        TOP_N,
        "mean_predicted_escape_reduction",
    ).copy()
    frame["label"] = (
        frame["target"].astype(str) + " (" + frame["intervention"].astype(str) + ")"
    )
    frame = frame.sort_values("mean_predicted_escape_reduction")

    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(frame["label"], frame["mean_predicted_escape_reduction"])
    axis.set_xlabel("Mean predicted reduction in escape probability")
    axis.set_ylabel("Simulated intervention")
    axis.set_title("Top computationally predicted interventions")
    save_figure(fig, "top_predicted_interventions.png")


def plot_baseline_model_comparison() -> None:
    data = read_csv_candidates((TABLES_DIR / "baseline_model_summary.csv",))
    if data is None or data.empty:
        return

    required = {"model", "roc_auc_mean"}
    if not required.issubset(data.columns):
        LOGGER.warning("Baseline-model summary is missing %s.", sorted(required))
        return

    frame = data.sort_values("roc_auc_mean")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(frame["model"], frame["roc_auc_mean"])
    axis.axvline(0.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Mean leave-one-cell-line-out ROC-AUC")
    axis.set_ylabel("Model")
    axis.set_title("Predictive performance of baseline feature groups")
    axis.set_xlim(0, max(1.0, float(frame["roc_auc_mean"].max()) + 0.05))
    save_figure(fig, "baseline_model_comparison.png")


def plot_model_uncertainty() -> None:
    data = read_csv_candidates((TABLES_DIR / "baseline_model_summary.csv",))
    if data is None or data.empty:
        return

    required = {"model", "roc_auc_mean", "roc_auc_std"}
    if not required.issubset(data.columns):
        return

    frame = data.sort_values("roc_auc_mean")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        frame["roc_auc_mean"],
        frame["model"],
        xerr=frame["roc_auc_std"],
        fmt="o",
        capsize=4,
    )
    axis.axvline(0.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Mean ROC-AUC ± cross-cell-line standard deviation")
    axis.set_ylabel("Model")
    axis.set_title("Model performance and cross-cell-line variability")
    axis.set_xlim(0, 1)
    save_figure(fig, "baseline_model_uncertainty.png")


def plot_cell_cycle_diagnostic(adata) -> None:
    if adata is None:
        return

    columns = list(adata.obs.columns)
    escape_probability = first_present(
        columns,
        ESCAPE_PROBABILITY_COLUMN_CANDIDATES,
    )
    phase = first_present(columns, ("phase", "cell_cycle_phase"))

    if escape_probability is None or phase is None:
        return

    frame = adata.obs[[phase, escape_probability]].copy()
    frame[escape_probability] = pd.to_numeric(
        frame[escape_probability],
        errors="coerce",
    )
    frame = frame.dropna()
    if frame.empty:
        return

    groups = sorted(frame[phase].astype(str).unique())
    values = [
        frame.loc[frame[phase].astype(str) == group, escape_probability].to_numpy()
        for group in groups
    ]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, tick_labels=groups, showfliers=False)
    axis.set_xlabel("Cell-cycle phase")
    axis.set_ylabel("Predicted escape probability")
    axis.set_title("Cell-cycle diagnostic for escape predictions")
    save_figure(fig, "escape_probability_by_cell_cycle_phase.png")


def main() -> None:
    configure_logging()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_baseline_model_comparison()
    plot_model_uncertainty()
    plot_target_priority()
    plot_predicted_interventions()
    plot_pathway_enrichment()
    plot_de_volcano()

    adata = load_plotting_anndata()
    adata = sample_anndata_for_plotting(adata)
    plot_umap_metadata(adata)
    plot_trajectory(adata)
    plot_escape_probability_distribution(adata)
    plot_cell_cycle_diagnostic(adata)

    if adata is not None and getattr(adata, "isbacked", False):
        adata.file.close()

    LOGGER.info("Figure generation complete. Output: %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
