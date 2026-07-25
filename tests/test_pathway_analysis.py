from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.pathway_analysis import (
    create_conserved_pathway_summary,
    create_pathway_summary,
    create_ranking_metric,
    filter_significant_pathways,
    normalize_gsea_results,
    run_pathway_analysis,
    run_preranked_gsea,
    save_results,
    validate_differential_expression,
)


@pytest.fixture
def differential_expression_results() -> pd.DataFrame:
    rows = []

    genes = [
        "E2F1",
        "MYC",
        "TP53",
        "JUN",
        "FOS",
        "STAT3",
    ]

    for cell_line in ["MCF7", "T47D"]:
        for comparison in [
            "TIS_vs_CTR",
            "REPOP_vs_TIS",
            "REPOP_vs_CTR",
        ]:
            for gene_index, gene in enumerate(genes):
                effect = (gene_index + 1) * 0.5

                if comparison == "TIS_vs_CTR":
                    effect *= -1

                rows.append(
                    {
                        "cell_line": cell_line,
                        "comparison": comparison,
                        "gene": gene,
                        "score": effect * 2,
                        "log2_fold_change": effect,
                        "adjusted_p_value": 0.01,
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture
def mock_gsea_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Term": [
                "E2F Targets",
                "DNA Repair",
                "Interferon Response",
            ],
            "ES": [
                0.6,
                -0.5,
                0.3,
            ],
            "NES": [
                1.8,
                -1.5,
                0.8,
            ],
            "NOM p-val": [
                0.001,
                0.01,
                0.2,
            ],
            "FDR q-val": [
                0.01,
                0.04,
                0.3,
            ],
            "FWER p-val": [
                0.02,
                0.08,
                0.5,
            ],
            "Tag %": [
                "20/50",
                "15/40",
                "10/30",
            ],
            "Gene %": [
                "10%",
                "15%",
                "20%",
            ],
            "Lead_genes": [
                "E2F1;MYC",
                "TP53",
                "STAT3",
            ],
        }
    )


def test_validate_differential_expression_accepts_valid_data(
    differential_expression_results: pd.DataFrame,
) -> None:
    validate_differential_expression(differential_expression_results)


def test_validate_differential_expression_rejects_missing_columns() -> None:
    results = pd.DataFrame(
        {
            "gene": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_differential_expression(results)


def test_create_ranking_metric(
    differential_expression_results: pd.DataFrame,
) -> None:
    subset = differential_expression_results.loc[
        (differential_expression_results["cell_line"] == "MCF7")
        & (differential_expression_results["comparison"] == "REPOP_vs_TIS")
    ]

    ranking = create_ranking_metric(subset)

    assert set(ranking.columns) == {
        "gene",
        "ranking_metric",
    }

    assert len(ranking) == 6

    assert ranking["ranking_metric"].is_monotonic_decreasing


def test_create_ranking_metric_removes_duplicate_genes() -> None:
    results = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "MCF7",
                "MCF7",
            ],
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
            ],
            "gene": [
                "E2F1",
                "E2F1",
                "TP53",
            ],
            "score": [
                2.0,
                -5.0,
                1.5,
            ],
            "log2_fold_change": [
                1.0,
                -2.0,
                0.8,
            ],
            "adjusted_p_value": [
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    ranking = create_ranking_metric(results)

    assert len(ranking) == 2

    e2f1_score = ranking.loc[
        ranking["gene"] == "E2F1",
        "ranking_metric",
    ].iloc[0]

    assert e2f1_score == -5.0


def test_normalize_gsea_results(
    mock_gsea_results: pd.DataFrame,
) -> None:
    normalized = normalize_gsea_results(
        mock_gsea_results,
        cell_line="MCF7",
        comparison="REPOP_vs_TIS",
        gene_set_library="MSigDB_Hallmark_2020",
    )

    assert len(normalized) == 3

    assert set(normalized["direction"]) == {
        "positive",
        "negative",
    }

    assert set(normalized["cell_line"]) == {
        "MCF7",
    }

    assert normalized["normalized_enrichment_score"].max() == 1.8


def test_run_preranked_gsea(
    monkeypatch: pytest.MonkeyPatch,
    differential_expression_results: pd.DataFrame,
    mock_gsea_results: pd.DataFrame,
) -> None:
    def fake_prerank(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(res2d=mock_gsea_results)

    monkeypatch.setattr(
        "src.pathway_analysis.gp.prerank",
        fake_prerank,
    )

    subset = differential_expression_results.loc[
        (differential_expression_results["cell_line"] == "MCF7")
        & (differential_expression_results["comparison"] == "REPOP_vs_TIS")
    ]

    ranking = create_ranking_metric(subset)

    results = run_preranked_gsea(
        ranking,
        gene_set_library="MOCK_LIBRARY",
        cell_line="MCF7",
        comparison="REPOP_vs_TIS",
        permutations=10,
        threads=1,
    )

    assert len(results) == 3

    assert set(results["gene_set_library"]) == {
        "MOCK_LIBRARY",
    }


def test_run_pathway_analysis(
    monkeypatch: pytest.MonkeyPatch,
    differential_expression_results: pd.DataFrame,
    mock_gsea_results: pd.DataFrame,
) -> None:
    def fake_prerank(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(res2d=mock_gsea_results)

    monkeypatch.setattr(
        "src.pathway_analysis.gp.prerank",
        fake_prerank,
    )

    results = run_pathway_analysis(
        differential_expression_results,
        gene_set_libraries=("MOCK_LIBRARY",),
        permutations=10,
        threads=1,
    )

    assert len(results) == 18

    assert set(results["cell_line"]) == {
        "MCF7",
        "T47D",
    }

    assert set(results["comparison"]) == {
        "TIS_vs_CTR",
        "REPOP_vs_TIS",
        "REPOP_vs_CTR",
    }


def test_filter_significant_pathways(
    mock_gsea_results: pd.DataFrame,
) -> None:
    normalized = normalize_gsea_results(
        mock_gsea_results,
        cell_line="MCF7",
        comparison="REPOP_vs_TIS",
        gene_set_library="Hallmark",
    )

    significant = filter_significant_pathways(
        normalized,
        maximum_false_discovery_rate=0.05,
        minimum_absolute_nes=1.0,
    )

    assert set(significant["pathway"]) == {
        "E2F Targets",
        "DNA Repair",
    }


def test_filter_significant_pathways_rejects_invalid_fdr(
    mock_gsea_results: pd.DataFrame,
) -> None:
    normalized = normalize_gsea_results(
        mock_gsea_results,
        cell_line="MCF7",
        comparison="REPOP_vs_TIS",
        gene_set_library="Hallmark",
    )

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        filter_significant_pathways(
            normalized,
            maximum_false_discovery_rate=0,
        )


def test_create_conserved_pathway_summary() -> None:
    results = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "T47D",
                "MCF7",
            ],
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
            ],
            "gene_set_library": [
                "Hallmark",
                "Hallmark",
                "Hallmark",
            ],
            "pathway": [
                "E2F Targets",
                "E2F Targets",
                "DNA Repair",
            ],
            "normalized_enrichment_score": [
                1.8,
                1.6,
                -1.5,
            ],
            "false_discovery_rate": [
                0.01,
                0.02,
                0.03,
            ],
            "direction": [
                "positive",
                "positive",
                "negative",
            ],
        }
    )

    conserved = create_conserved_pathway_summary(
        results,
        required_cell_lines=2,
    )

    assert len(conserved) == 1

    assert conserved["pathway"].iloc[0] == "E2F Targets"

    assert conserved["cell_lines_supported"].iloc[0] == 2


def test_create_pathway_summary() -> None:
    results = pd.DataFrame(
        {
            "cell_line": [
                "MCF7",
                "MCF7",
                "MCF7",
            ],
            "comparison": [
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
                "REPOP_vs_TIS",
            ],
            "gene_set_library": [
                "Hallmark",
                "Hallmark",
                "Hallmark",
            ],
            "pathway": [
                "A",
                "B",
                "C",
            ],
            "direction": [
                "positive",
                "positive",
                "negative",
            ],
        }
    )

    summary = create_pathway_summary(results)

    assert summary["significant_pathways"].iloc[0] == 3

    assert summary["positively_enriched"].iloc[0] == 2

    assert summary["negatively_enriched"].iloc[0] == 1


def test_save_results(
    tmp_path: Path,
) -> None:
    all_results = pd.DataFrame(
        {
            "pathway": ["A"],
        }
    )

    significant = pd.DataFrame(
        {
            "pathway": ["A"],
        }
    )

    conserved = pd.DataFrame(
        {
            "pathway": ["A"],
        }
    )

    summary = pd.DataFrame(
        {
            "significant_pathways": [1],
        }
    )

    save_results(
        all_results,
        significant,
        conserved,
        summary,
        tmp_path,
    )

    assert (tmp_path / "pathway_results_all.csv").exists()

    assert (tmp_path / "pathway_results_significant.csv").exists()

    assert (tmp_path / "conserved_pathways.csv").exists()

    assert (tmp_path / "pathway_summary.csv").exists()
