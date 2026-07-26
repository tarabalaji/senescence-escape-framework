from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "escape_prone_scored.h5ad"
OUTPUT_PATH = ROOT / "data" / "processed" / "escape_prone_cell_cycle_scored.h5ad"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

RANDOM_STATE = 42
FIGURE_DPI = 300

# Canonical human S-phase and G2/M-phase markers commonly used by Seurat/Scanpy.
S_GENES = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]

G2M_GENES = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path.relative_to(ROOT))


def load_adata() -> sc.AnnData:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    LOGGER.info("Loading %s", INPUT_PATH.relative_to(ROOT))
    adata = sc.read_h5ad(INPUT_PATH)

    # Cell-cycle scoring should use normalized, log-transformed expression.
    # Prefer .raw when it contains the full gene set.
    if adata.raw is not None:
        LOGGER.info("Using adata.raw for cell-cycle scoring.")
        scoring = adata.raw.to_adata()
        scoring.obs = adata.obs.copy()
        scoring.obsm = adata.obsm.copy()
        return scoring

    LOGGER.warning(
        "adata.raw is absent. Assuming adata.X is already normalized and log-transformed."
    )
    return adata


def standardize_gene_names(adata: sc.AnnData) -> None:
    adata.var_names = pd.Index(adata.var_names.astype(str).str.upper())
    adata.var_names_make_unique()


def score_cell_cycle(adata: sc.AnnData) -> None:
    available = set(adata.var_names)
    s_genes = [gene for gene in S_GENES if gene in available]
    g2m_genes = [gene for gene in G2M_GENES if gene in available]

    LOGGER.info(
        "Found %d/%d S genes and %d/%d G2M genes.",
        len(s_genes),
        len(S_GENES),
        len(g2m_genes),
        len(G2M_GENES),
    )

    if len(s_genes) < 10 or len(g2m_genes) < 10:
        raise ValueError(
            "Too few cell-cycle genes were found. Check whether var_names are gene "
            "symbols and whether the expression matrix contains the full gene set."
        )

    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,
        g2m_genes=g2m_genes,
        random_state=RANDOM_STATE,
    )

    # A single continuous index is useful for correlation diagnostics.
    adata.obs["cell_cycle_activity"] = adata.obs["S_score"] + adata.obs["G2M_score"]


def copy_project_metadata(source_path: Path, adata: sc.AnnData) -> None:
    source = sc.read_h5ad(source_path, backed="r")
    try:
        for column in source.obs.columns:
            if column not in adata.obs.columns:
                adata.obs[column] = source.obs[column].reindex(adata.obs_names)
        for key in source.obsm.keys():
            if key not in adata.obsm:
                adata.obsm[key] = np.asarray(source.obsm[key])
    finally:
        source.file.close()


def create_umap_figures(adata: sc.AnnData) -> None:
    if "X_umap" not in adata.obsm:
        LOGGER.warning("X_umap is absent; skipping UMAP figures.")
        return

    for column, title, filename in (
        ("phase", "Cell-cycle phase", "umap_cell_cycle_phase.png"),
        ("S_score", "S-phase score", "umap_s_score.png"),
        ("G2M_score", "G2/M-phase score", "umap_g2m_score.png"),
    ):
        axis = sc.pl.umap(
            adata,
            color=column,
            size=5,
            alpha=0.75,
            frameon=False,
            show=False,
            title=title,
        )
        save_figure(axis.figure, filename)


def create_phase_summary(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()

    probability_column = (
        "transition_probability_oof"
        if "transition_probability_oof" in obs.columns
        else "transition_probability"
    )

    aggregations = {
        "cell_count": ("phase", "size"),
        "mean_s_score": ("S_score", "mean"),
        "mean_g2m_score": ("G2M_score", "mean"),
        "mean_rap_score": ("rap_score", "mean"),
        "mean_transition_probability": (probability_column, "mean"),
        "median_transition_probability": (probability_column, "median"),
    }

    summary = (
        obs.groupby(["cell_line", "condition", "phase"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / "cell_cycle_phase_summary.csv", index=False)
    return summary


def create_phase_probability_plot(adata: sc.AnnData) -> None:
    obs = adata.obs.copy()
    probability_column = (
        "transition_probability_oof"
        if "transition_probability_oof" in obs.columns
        else "transition_probability"
    )

    phases = [phase for phase in ("G1", "S", "G2M") if phase in set(obs["phase"])]
    values = [
        pd.to_numeric(
            obs.loc[obs["phase"] == phase, probability_column],
            errors="coerce",
        ).dropna()
        for phase in phases
    ]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, tick_labels=phases, showfliers=False)
    axis.set_xlabel("Cell-cycle phase")
    axis.set_ylabel("Out-of-fold transition probability")
    axis.set_title("Escape prediction across cell-cycle phases")
    save_figure(fig, "escape_probability_by_cell_cycle_phase.png")


def calculate_correlations(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()
    probability_column = (
        "transition_probability_oof"
        if "transition_probability_oof" in obs.columns
        else "transition_probability"
    )

    outcomes = [
        column
        for column in (
            "rap_score",
            probability_column,
            "escape_pseudotime",
        )
        if column in obs.columns
    ]
    cycle_scores = ["S_score", "G2M_score", "cell_cycle_activity"]

    rows: list[dict[str, object]] = []

    # Report both pooled and cell-line-specific relationships.
    groups = [("all", obs)]
    if "cell_line" in obs.columns:
        groups.extend(
            (str(name), group)
            for name, group in obs.groupby("cell_line", observed=True)
        )

    for group_name, group in groups:
        for outcome in outcomes:
            for cycle_score in cycle_scores:
                frame = (
                    group[[outcome, cycle_score]]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .dropna()
                )
                if len(frame) < 3:
                    continue

                rho, p_value = spearmanr(frame[outcome], frame[cycle_score])
                rows.append(
                    {
                        "group": group_name,
                        "outcome": outcome,
                        "cell_cycle_measure": cycle_score,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n_cells": len(frame),
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "cell_cycle_correlations.csv", index=False)
    return results


def scatter_with_hexbin(
    adata: sc.AnnData,
    outcome: str,
    filename: str,
    title: str,
) -> None:
    frame = (
        adata.obs[["cell_cycle_activity", outcome]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    if len(frame) > 100_000:
        frame = frame.sample(100_000, random_state=RANDOM_STATE)

    fig, axis = plt.subplots(figsize=(7, 5))
    plot = axis.hexbin(
        frame["cell_cycle_activity"],
        frame[outcome],
        gridsize=55,
        mincnt=1,
    )
    fig.colorbar(plot, ax=axis, label="Cell count")
    rho, _ = spearmanr(frame["cell_cycle_activity"], frame[outcome])
    axis.set_xlabel("Cell-cycle activity (S score + G2M score)")
    axis.set_ylabel(outcome.replace("_", " ").title())
    axis.set_title(f"{title}\nSpearman rho = {rho:.3f}")
    save_figure(fig, filename)


def make_logistic_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessing = ColumnTransformer(transformers=transformers)

    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def cell_line_holdout_auc(
    frame: pd.DataFrame,
    features: list[str],
    label_column: str,
) -> tuple[float, float, list[float]]:
    fold_scores: list[float] = []

    for held_out in sorted(frame["cell_line"].dropna().astype(str).unique()):
        train_mask = frame["cell_line"].astype(str) != held_out
        test_mask = ~train_mask

        train = frame.loc[train_mask]
        test = frame.loc[test_mask]

        if train[label_column].nunique() < 2 or test[label_column].nunique() < 2:
            LOGGER.warning(
                "Skipping held-out cell line %s due to one-class labels.", held_out
            )
            continue

        numeric_features = [feature for feature in features if feature != "phase"]
        categorical_features = [feature for feature in features if feature == "phase"]

        model = make_logistic_pipeline(numeric_features, categorical_features)
        model.fit(train[features], train[label_column])
        probability = model.predict_proba(test[features])[:, 1]
        fold_scores.append(roc_auc_score(test[label_column], probability))

    if not fold_scores:
        return np.nan, np.nan, []

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores


def create_adjusted_model_comparison(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()

    if "escape_prone_status" not in obs.columns:
        LOGGER.warning(
            "escape_prone_status is absent; skipping adjusted model comparison."
        )
        return pd.DataFrame()

    # Restrict to TIS cells because stable_tis vs escape_prone_tis is the
    # scientifically relevant within-state comparison.
    tis = obs[obs["condition"].astype(str).str.upper() == "TIS"].copy()
    tis["escape_label"] = (
        tis["escape_prone_status"].astype(str) == "escape_prone_tis"
    ).astype(int)

    model_features = {
        "cell_cycle_only": ["S_score", "G2M_score", "phase"],
        "rap_only": ["rap_score"],
        "rap_plus_cell_cycle": ["rap_score", "S_score", "G2M_score", "phase"],
        "pathways_only": [
            column for column in tis.columns if column.startswith("pathway__")
        ],
        "pathways_plus_cell_cycle": [
            *[column for column in tis.columns if column.startswith("pathway__")],
            "S_score",
            "G2M_score",
            "phase",
        ],
        "rap_pathways_cell_cycle": [
            "rap_score",
            *[column for column in tis.columns if column.startswith("pathway__")],
            "S_score",
            "G2M_score",
            "phase",
        ],
    }

    rows = []
    for model_name, features in model_features.items():
        features = [feature for feature in features if feature in tis.columns]
        mean_auc, std_auc, fold_scores = cell_line_holdout_auc(
            tis,
            features,
            "escape_label",
        )
        rows.append(
            {
                "model": model_name,
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
                "fold_scores": ";".join(f"{score:.6f}" for score in fold_scores),
                "n_features": len(features),
                "n_cells": len(tis),
            }
        )

    results = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    results.to_csv(TABLES_DIR / "cell_cycle_adjusted_models.csv", index=False)

    plot_data = results.sort_values("roc_auc_mean")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        plot_data["roc_auc_mean"],
        plot_data["model"],
        xerr=plot_data["roc_auc_std"],
        fmt="o",
        capsize=4,
    )
    axis.axvline(0.5, linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Leave-one-cell-line-out ROC-AUC")
    axis.set_ylabel("Feature set")
    axis.set_title("Does RAP/pathway signal persist beyond cell cycle?")
    save_figure(fig, "cell_cycle_adjusted_model_comparison.png")

    return results


def main() -> None:
    configure_logging()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    adata = load_adata()
    standardize_gene_names(adata)

    # If scoring came from .raw, restore project metadata and embeddings.
    copy_project_metadata(INPUT_PATH, adata)
    score_cell_cycle(adata)

    adata.write_h5ad(OUTPUT_PATH, compression="gzip")
    LOGGER.info("Saved %s", OUTPUT_PATH.relative_to(ROOT))

    create_umap_figures(adata)
    create_phase_summary(adata)
    create_phase_probability_plot(adata)
    calculate_correlations(adata)

    if "rap_score" in adata.obs.columns:
        scatter_with_hexbin(
            adata,
            "rap_score",
            "rap_vs_cell_cycle.png",
            "RAP score versus cell-cycle activity",
        )

    probability_column = (
        "transition_probability_oof"
        if "transition_probability_oof" in adata.obs.columns
        else "transition_probability"
    )
    if probability_column in adata.obs.columns:
        scatter_with_hexbin(
            adata,
            probability_column,
            "transition_probability_vs_cell_cycle.png",
            "Transition probability versus cell-cycle activity",
        )

    create_adjusted_model_comparison(adata)
    LOGGER.info("Cell-cycle diagnostics complete.")


if __name__ == "__main__":
    main()
