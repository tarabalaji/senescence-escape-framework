from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.transition_model import (
    build_transition_features,
    calculate_pathway_activity,
    calculate_regulator_features,
    create_prediction_summary,
    create_transition_labels,
    evaluate_predictions,
    fit_final_model,
    parse_gene_list,
    run_leave_one_cell_line_out,
    run_transition_model,
    save_results,
    select_pathway_gene_sets,
    select_regulators,
    standardize_gene_expression,
    validate_adata,
)


@pytest.fixture
def transition_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)
    genes = ["E2F1", "MKI67", "CDK1", "MYC", "TP53", "JUN", "FOS", "STAT3"]
    rows = []
    cell_lines = []
    states = []
    rap_scores = []

    for cell_line in ["MCF7", "T47D"]:
        for state in ["CTR", "TIS", "REPOP"]:
            for _ in range(10):
                expression = rng.normal(0.0, 0.3, size=len(genes))
                if state == "REPOP":
                    expression[:4] += 2.0
                    rap_score = rng.normal(2.0, 0.2)
                elif state == "TIS":
                    expression[4:] += 1.0
                    rap_score = rng.normal(0.5, 0.2)
                else:
                    rap_score = rng.normal(-1.0, 0.2)
                rows.append(expression)
                cell_lines.append(cell_line)
                states.append(state)
                rap_scores.append(rap_score)

    adata = ad.AnnData(
        X=np.asarray(rows),
        obs=pd.DataFrame(
            {
                "cell_line": cell_lines,
                "state": states,
                "rap_score": rap_scores,
            }
        ),
        var=pd.DataFrame(index=genes),
    )
    adata.obs_names = [f"cell_{index}" for index in range(adata.n_obs)]
    return adata


@pytest.fixture
def pathway_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "TIS_vs_CTR",
            ],
            "pathway": ["E2F Targets", "Cell Cycle", "DNA Repair", "Senescence"],
            "normalized_enrichment_score": [2.5, 2.0, 1.2, -1.5],
            "false_discovery_rate": [0.001, 0.010, 0.300, 0.010],
            "leading_edge_genes": [
                "E2F1;MKI67;CDK1",
                "MKI67;CDK1;MYC",
                "TP53;JUN",
                "TP53;FOS",
            ],
        }
    )


@pytest.fixture
def regulator_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regulator": ["E2F1", "MYC", "TP53", "JUN"],
            "priority_score": [0.9, 0.8, 0.4, 0.2],
        }
    )


def test_validate_adata_accepts_valid_data(transition_adata: ad.AnnData) -> None:
    validate_adata(transition_adata)


def test_validate_adata_rejects_missing_columns(transition_adata: ad.AnnData) -> None:
    broken = transition_adata.copy()
    del broken.obs["rap_score"]
    with pytest.raises(ValueError, match="missing columns"):
        validate_adata(broken)


def test_parse_gene_list() -> None:
    assert parse_gene_list("E2F1; MKI67,CDK1|MYC") == ["E2F1", "MKI67", "CDK1", "MYC"]


def test_select_pathway_gene_sets(pathway_results: pd.DataFrame) -> None:
    selected = select_pathway_gene_sets(pathway_results, maximum_pathways=2)
    assert list(selected) == ["E2F Targets", "Cell Cycle"]


def test_select_regulators(regulator_results: pd.DataFrame) -> None:
    selected = select_regulators(regulator_results, maximum_regulators=3)
    assert selected == ["E2F1", "MYC", "TP53"]


def test_standardize_gene_expression(transition_adata: ad.AnnData) -> None:
    standardized = standardize_gene_expression(transition_adata, ["E2F1", "MKI67"])
    assert standardized.shape == (transition_adata.n_obs, 2)
    np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-7)


def test_calculate_pathway_activity(transition_adata: ad.AnnData) -> None:
    activity = calculate_pathway_activity(
        transition_adata,
        {"E2F Targets": ["E2F1", "MKI67", "CDK1"]},
    )
    assert list(activity.columns) == ["pathway__e2f_targets"]


def test_calculate_regulator_features(transition_adata: ad.AnnData) -> None:
    features = calculate_regulator_features(transition_adata, ["E2F1", "MYC"])
    assert set(features.columns) == {"regulator__E2F1", "regulator__MYC"}


def test_build_transition_features(transition_adata: ad.AnnData) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={"E2F Targets": ["E2F1", "MKI67", "CDK1"]},
        regulators=["E2F1", "MYC"],
    )
    assert "rap_score" in features.columns
    assert "pathway__e2f_targets" in features.columns
    assert "regulator__E2F1" in features.columns


def test_create_transition_labels(transition_adata: ad.AnnData) -> None:
    labels = create_transition_labels(transition_adata)
    expected_positive = int((transition_adata.obs["state"] == "REPOP").sum())
    assert int(labels.sum()) == expected_positive


def test_evaluate_predictions() -> None:
    metrics = evaluate_predictions(
        labels=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_run_leave_one_cell_line_out(transition_adata: ad.AnnData) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={"E2F Targets": ["E2F1", "MKI67", "CDK1"]},
        regulators=["E2F1", "MYC"],
    )
    labels = create_transition_labels(transition_adata)
    predictions, metrics = run_leave_one_cell_line_out(
        features,
        labels,
        transition_adata.obs["cell_line"],
    )
    assert len(predictions) == transition_adata.n_obs
    assert len(metrics) == 2
    assert set(predictions["cell_line"]) == {"MCF7", "T47D"}


def test_fit_final_model(transition_adata: ad.AnnData) -> None:
    features = build_transition_features(
        transition_adata,
        pathway_gene_sets={"E2F Targets": ["E2F1", "MKI67", "CDK1"]},
        regulators=["E2F1", "MYC"],
    )
    labels = create_transition_labels(transition_adata)
    model, coefficients = fit_final_model(features, labels)
    assert hasattr(model, "predict_proba")
    assert set(coefficients.columns) == {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }


def test_run_transition_model(
    transition_adata: ad.AnnData,
    pathway_results: pd.DataFrame,
    regulator_results: pd.DataFrame,
) -> None:
    scored, metrics, coefficients, summary, feature_table = run_transition_model(
        transition_adata,
        pathway_results,
        regulator_results,
        maximum_pathways=2,
        maximum_regulators=3,
    )
    assert "transition_probability" in scored.obs.columns
    assert "transition_prediction" in scored.obs.columns
    assert not metrics.empty
    assert not coefficients.empty
    assert not summary.empty
    assert len(feature_table) == scored.n_obs


def test_create_prediction_summary(transition_adata: ad.AnnData) -> None:
    scored = transition_adata.copy()
    scored.obs["transition_probability"] = np.linspace(0, 1, scored.n_obs)
    summary = create_prediction_summary(scored)
    assert set(summary.columns) == {
        "cell_line",
        "state",
        "cell_count",
        "mean_transition_probability",
        "median_transition_probability",
        "standard_deviation",
    }


def test_save_results(tmp_path: Path, transition_adata: ad.AnnData) -> None:
    metrics = pd.DataFrame({"accuracy": [1.0]})
    coefficients = pd.DataFrame(
        {
            "feature": ["rap_score"],
            "coefficient": [1.0],
            "absolute_coefficient": [1.0],
        }
    )
    summary = pd.DataFrame({"cell_line": ["MCF7"], "state": ["REPOP"]})
    feature_table = pd.DataFrame({"cell_id": ["cell_0"], "rap_score": [1.0]})
    output_dataset = tmp_path / "transition_model.h5ad"

    save_results(
        transition_adata,
        metrics,
        coefficients,
        summary,
        feature_table,
        output_dataset,
        tmp_path,
    )

    assert output_dataset.exists()
    assert (tmp_path / "transition_model_metrics.csv").exists()
    assert (tmp_path / "transition_model_coefficients.csv").exists()
    assert (tmp_path / "transition_prediction_summary.csv").exists()
    assert (tmp_path / "transition_feature_table.csv").exists()
