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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path.relative_to(ROOT))


def load_adata() -> sc.AnnData:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    LOGGER.info("Loading %s", INPUT_PATH.relative_to(ROOT))
    source = sc.read_h5ad(INPUT_PATH)

    if source.raw is None:
        LOGGER.warning(
            "adata.raw is absent. Assuming adata.X is normalized and log-transformed."
        )
        return source

    LOGGER.info("Using adata.raw for cell-cycle scoring.")
    scoring = source.raw.to_adata()
    scoring.obs = source.obs.copy()

    for key in source.obsm.keys():
        scoring.obsm[key] = np.asarray(source.obsm[key])

    return scoring


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
            "Too few canonical cell-cycle genes were found. Check that var_names "
            "contain human gene symbols."
        )

    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,
        g2m_genes=g2m_genes,
        random_state=RANDOM_STATE,
    )

    adata.obs["cell_cycle_activity"] = adata.obs["S_score"] + adata.obs["G2M_score"]


def create_umap_figures(adata: sc.AnnData) -> None:
    if "X_umap" not in adata.obsm:
        LOGGER.warning("X_umap is absent; skipping cell-cycle UMAPs.")
        return

    panels = (
        ("phase", "Cell-cycle phase", "umap_cell_cycle_phase.png"),
        ("S_score", "S-phase score", "umap_s_score.png"),
        ("G2M_score", "G2/M-phase score", "umap_g2m_score.png"),
    )

    for column, title, filename in panels:
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


def transition_probability_column(obs: pd.DataFrame) -> str:
    if "transition_probability_oof" in obs.columns:
        return "transition_probability_oof"
    if "transition_probability" in obs.columns:
        return "transition_probability"
    raise KeyError(
        "Neither transition_probability_oof nor transition_probability exists."
    )


def create_phase_summary(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()
    probability = transition_probability_column(obs)

    required = {
        "cell_line",
        "condition",
        "phase",
        "S_score",
        "G2M_score",
        "rap_score",
        probability,
    }
    missing = sorted(required.difference(obs.columns))
    if missing:
        raise KeyError(f"Missing columns for phase summary: {missing}")

    summary = (
        obs.groupby(["cell_line", "condition", "phase"], observed=True)
        .agg(
            cell_count=("phase", "size"),
            mean_s_score=("S_score", "mean"),
            mean_g2m_score=("G2M_score", "mean"),
            mean_rap_score=("rap_score", "mean"),
            mean_transition_probability=(probability, "mean"),
            median_transition_probability=(probability, "median"),
        )
        .reset_index()
    )

    path = TABLES_DIR / "cell_cycle_phase_summary.csv"
    summary.to_csv(path, index=False)
    LOGGER.info("Saved %s", path.relative_to(ROOT))
    return summary


def create_phase_probability_plot(adata: sc.AnnData) -> None:
    obs = adata.obs.copy()
    probability = transition_probability_column(obs)

    phases = [
        phase for phase in ("G1", "S", "G2M") if phase in set(obs["phase"].astype(str))
    ]
    values = [
        pd.to_numeric(
            obs.loc[obs["phase"].astype(str) == phase, probability],
            errors="coerce",
        ).dropna()
        for phase in phases
    ]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, tick_labels=phases, showfliers=False)
    axis.set_xlabel("Cell-cycle phase")
    axis.set_ylabel("Out-of-fold transition probability")
    axis.set_title("Transition probability across cell-cycle phases")
    save_figure(fig, "transition_probability_by_cell_cycle_phase.png")


def calculate_correlations(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()
    probability = transition_probability_column(obs)

    outcomes = [
        column
        for column in ("rap_score", probability, "escape_pseudotime")
        if column in obs.columns
    ]
    cycle_measures = ["S_score", "G2M_score", "cell_cycle_activity"]

    groups: list[tuple[str, pd.DataFrame]] = [("all", obs)]
    groups.extend(
        (str(name), group) for name, group in obs.groupby("cell_line", observed=True)
    )

    rows: list[dict[str, object]] = []
    for group_name, group in groups:
        for outcome in outcomes:
            for measure in cycle_measures:
                frame = (
                    group[[outcome, measure]]
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna()
                )
                if len(frame) < 3:
                    continue

                rho, p_value = spearmanr(frame[outcome], frame[measure])
                rows.append(
                    {
                        "group": group_name,
                        "outcome": outcome,
                        "cell_cycle_measure": measure,
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                        "n_cells": len(frame),
                    }
                )

    results = pd.DataFrame(rows)
    path = TABLES_DIR / "cell_cycle_correlations.csv"
    results.to_csv(path, index=False)
    LOGGER.info("Saved %s", path.relative_to(ROOT))
    return results


def create_hexbin(
    adata: sc.AnnData,
    outcome: str,
    filename: str,
    title: str,
) -> None:
    frame = (
        adata.obs[["cell_cycle_activity", outcome]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    if len(frame) > 100_000:
        frame = frame.sample(100_000, random_state=RANDOM_STATE)

    rho, _ = spearmanr(frame["cell_cycle_activity"], frame[outcome])

    fig, axis = plt.subplots(figsize=(7, 5))
    plot = axis.hexbin(
        frame["cell_cycle_activity"],
        frame[outcome],
        gridsize=55,
        mincnt=1,
    )
    fig.colorbar(plot, ax=axis, label="Cell count")
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


def evaluate_leave_one_cell_line_out(
    frame: pd.DataFrame,
    features: list[str],
    label_column: str,
) -> dict[str, object]:
    fold_auc: list[float] = []
    fold_balanced_accuracy: list[float] = []
    fold_names: list[str] = []

    cell_lines = sorted(frame["cell_line"].dropna().astype(str).unique())

    for held_out in cell_lines:
        train = frame[frame["cell_line"].astype(str) != held_out]
        test = frame[frame["cell_line"].astype(str) == held_out]

        if train[label_column].nunique() < 2 or test[label_column].nunique() < 2:
            LOGGER.warning(
                "Skipping held-out cell line %s because one split has one class.",
                held_out,
            )
            continue

        numeric_features = [feature for feature in features if feature != "phase"]
        categorical_features = [feature for feature in features if feature == "phase"]

        model = make_logistic_pipeline(
            numeric_features,
            categorical_features,
        )
        model.fit(train[features], train[label_column])

        probability = model.predict_proba(test[features])[:, 1]
        prediction = (probability >= 0.5).astype(int)

        fold_names.append(held_out)
        fold_auc.append(roc_auc_score(test[label_column], probability))
        fold_balanced_accuracy.append(
            balanced_accuracy_score(test[label_column], prediction)
        )

    if not fold_auc:
        return {
            "roc_auc_mean": np.nan,
            "roc_auc_std": np.nan,
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "fold_cell_lines": "",
            "fold_roc_auc": "",
            "fold_balanced_accuracy": "",
            "n_folds": 0,
        }

    return {
        "roc_auc_mean": float(np.mean(fold_auc)),
        "roc_auc_std": float(np.std(fold_auc)),
        "balanced_accuracy_mean": float(np.mean(fold_balanced_accuracy)),
        "balanced_accuracy_std": float(np.std(fold_balanced_accuracy)),
        "fold_cell_lines": ";".join(fold_names),
        "fold_roc_auc": ";".join(f"{value:.6f}" for value in fold_auc),
        "fold_balanced_accuracy": ";".join(
            f"{value:.6f}" for value in fold_balanced_accuracy
        ),
        "n_folds": len(fold_auc),
    }


def available_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    pathways = [column for column in frame.columns if column.startswith("pathway__")]
    regulators = [
        column for column in frame.columns if column.startswith("regulator__")
    ]

    candidates = {
        "cell_cycle_only": ["S_score", "G2M_score", "phase"],
        "rap_only": ["rap_score"],
        "pathways_only": pathways,
        "regulators_only": regulators,
        "rap_plus_cell_cycle": [
            "rap_score",
            "S_score",
            "G2M_score",
            "phase",
        ],
        "pathways_plus_cell_cycle": [
            *pathways,
            "S_score",
            "G2M_score",
            "phase",
        ],
        "rap_plus_pathways": ["rap_score", *pathways],
        "rap_plus_pathways_plus_cell_cycle": [
            "rap_score",
            *pathways,
            "S_score",
            "G2M_score",
            "phase",
        ],
        "rap_plus_pathways_plus_regulators_plus_cell_cycle": [
            "rap_score",
            *pathways,
            *regulators,
            "S_score",
            "G2M_score",
            "phase",
        ],
    }

    return {
        name: [feature for feature in features if feature in frame.columns]
        for name, features in candidates.items()
    }


def run_model_comparison(
    frame: pd.DataFrame,
    label_column: str,
    prediction_task: str,
    table_filename: str,
    figure_filename: str,
    figure_title: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name, features in available_feature_sets(frame).items():
        if not features:
            LOGGER.warning(
                "Skipping %s because it has no available features.", model_name
            )
            continue

        metrics = evaluate_leave_one_cell_line_out(
            frame,
            features,
            label_column,
        )
        rows.append(
            {
                "prediction_task": prediction_task,
                "model": model_name,
                "n_features": len(features),
                "n_cells": len(frame),
                "positive_fraction": float(frame[label_column].mean()),
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    table_path = TABLES_DIR / table_filename
    results.to_csv(table_path, index=False)
    LOGGER.info("Saved %s", table_path.relative_to(ROOT))

    plot_data = results.dropna(subset=["roc_auc_mean"]).sort_values("roc_auc_mean")

    fig, axis = plt.subplots(figsize=(9, max(5, 0.5 * len(plot_data))))
    axis.errorbar(
        plot_data["roc_auc_mean"],
        plot_data["model"],
        xerr=plot_data["roc_auc_std"],
        fmt="o",
        capsize=4,
    )
    axis.axvline(0.5, linestyle="--", linewidth=1)
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Leave-one-cell-line-out ROC-AUC")
    axis.set_ylabel("Feature set")
    axis.set_title(figure_title)
    save_figure(fig, figure_filename)

    return results


def create_observed_outcome_analysis(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()

    required = {"condition", "cell_line"}
    missing = sorted(required.difference(obs.columns))
    if missing:
        raise KeyError(f"Missing columns for observed-outcome analysis: {missing}")

    frame = obs[obs["condition"].astype(str).str.upper().isin(["TIS", "REPOP"])].copy()
    frame["observed_repopulation_label"] = (
        frame["condition"].astype(str).str.upper() == "REPOP"
    ).astype(int)

    LOGGER.info(
        "Observed-outcome task: %d cells; positive fraction %.3f.",
        len(frame),
        frame["observed_repopulation_label"].mean(),
    )

    return run_model_comparison(
        frame=frame,
        label_column="observed_repopulation_label",
        prediction_task="tis_vs_repop_observed_condition",
        table_filename="cell_cycle_adjusted_observed_outcome_models.csv",
        figure_filename="cell_cycle_adjusted_observed_outcome_comparison.png",
        figure_title=("Cell-cycle-adjusted prediction of observed repopulation"),
    )


def create_internal_outcome_analysis(adata: sc.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()

    if "escape_prone_status" not in obs.columns:
        LOGGER.warning("escape_prone_status is absent; skipping internal diagnostic.")
        return pd.DataFrame()

    frame = obs[obs["condition"].astype(str).str.upper() == "TIS"].copy()

    valid_statuses = {"stable_tis", "escape_prone_tis"}
    frame = frame[frame["escape_prone_status"].astype(str).isin(valid_statuses)].copy()

    frame["internal_escape_prone_label"] = (
        frame["escape_prone_status"].astype(str) == "escape_prone_tis"
    ).astype(int)

    LOGGER.info(
        "Internal task: %d cells; positive fraction %.3f.",
        len(frame),
        frame["internal_escape_prone_label"].mean(),
    )

    return run_model_comparison(
        frame=frame,
        label_column="internal_escape_prone_label",
        prediction_task="stable_tis_vs_escape_prone_tis_internal",
        table_filename="cell_cycle_adjusted_internal_models.csv",
        figure_filename="cell_cycle_adjusted_internal_comparison.png",
        figure_title=("Internal discrimination of escape-prone senescent cells"),
    )


def main() -> None:
    configure_logging()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    adata = load_adata()
    standardize_gene_names(adata)
    score_cell_cycle(adata)

    adata.write_h5ad(OUTPUT_PATH, compression="gzip")
    LOGGER.info("Saved %s", OUTPUT_PATH.relative_to(ROOT))

    create_umap_figures(adata)
    create_phase_summary(adata)
    create_phase_probability_plot(adata)
    calculate_correlations(adata)

    if "rap_score" in adata.obs.columns:
        create_hexbin(
            adata,
            outcome="rap_score",
            filename="rap_vs_cell_cycle.png",
            title="RAP score versus cell-cycle activity",
        )

    probability = transition_probability_column(adata.obs)
    create_hexbin(
        adata,
        outcome=probability,
        filename="transition_probability_vs_cell_cycle.png",
        title="Transition probability versus cell-cycle activity",
    )

    create_observed_outcome_analysis(adata)
    create_internal_outcome_analysis(adata)

    LOGGER.info("Cell-cycle diagnostics complete.")


if __name__ == "__main__":
    main()
