from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite

from src.load_data import (
    discover_samples,
    load_dataset,
    load_sample,
    parse_sample_metadata,
    save_dataset,
)

SAMPLE_NAMES = [
    f"GSM{i:07d}_{cell_line}-{condition}-{replicate}"
    for i, (cell_line, condition, replicate) in enumerate(
        [
            ("MCF7", "ctr", "1"),
            ("MCF7", "ctr", "2"),
            ("MCF7", "tis", "1"),
            ("MCF7", "tis", "2"),
            ("MCF7", "repop", "1"),
            ("MCF7", "repop", "2"),
            ("T47D", "ctr", "1"),
            ("T47D", "ctr", "2"),
            ("T47D", "tis", "1"),
            ("T47D", "tis", "2"),
            ("T47D", "repop", "1"),
            ("T47D", "repop", "2"),
        ],
        start=1,
    )
]


def write_gzip_text(path: Path, content: str) -> None:
    with gzip.open(path, "wt") as file:
        file.write(content)


def create_sample_files(
    directory: Path,
    sample_id: str,
    counts: np.ndarray | None = None,
) -> None:
    if counts is None:
        counts = np.array(
            [
                [1, 0, 3],
                [0, 2, 1],
            ],
            dtype=np.int32,
        )

    matrix_path = directory / f"{sample_id}_matrix.mtx.gz"
    temporary_matrix_path = directory / f"{sample_id}_matrix.mtx"

    mmwrite(
        temporary_matrix_path,
        sparse.coo_matrix(counts),
    )

    with open(temporary_matrix_path, "rb") as source:
        with gzip.open(matrix_path, "wb") as destination:
            destination.write(source.read())

    temporary_matrix_path.unlink()

    write_gzip_text(
        directory / f"{sample_id}_features.tsv.gz",
        "ENSG000001\tGENE1\tGene Expression\nENSG000002\tGENE2\tGene Expression\n",
    )

    write_gzip_text(
        directory / f"{sample_id}_barcodes.tsv.gz",
        "CELL_A\nCELL_B\nCELL_C\n",
    )


@pytest.fixture
def complete_dataset_directory(tmp_path: Path) -> Path:
    for sample_name in SAMPLE_NAMES:
        create_sample_files(tmp_path, sample_name)

    return tmp_path


def test_parse_sample_metadata() -> None:
    metadata = parse_sample_metadata("GSM8595876_MCF7-ctr-1")

    assert metadata == {
        "sample_id": "GSM8595876_MCF7-ctr-1",
        "accession": "GSM8595876",
        "cell_line": "MCF7",
        "condition": "CTR",
        "replicate": "1",
    }


def test_parse_sample_metadata_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        parse_sample_metadata("invalid_sample_name")


def test_discover_samples(
    complete_dataset_directory: Path,
) -> None:
    sample_ids = discover_samples(complete_dataset_directory)

    assert len(sample_ids) == 12
    assert set(sample_ids) == set(SAMPLE_NAMES)


def test_load_sample(tmp_path: Path) -> None:
    sample_id = "GSM8595876_MCF7-ctr-1"
    create_sample_files(tmp_path, sample_id)

    sample_data = load_sample(tmp_path, sample_id)

    assert sample_data.shape == (3, 2)
    assert sparse.issparse(sample_data.X)

    assert set(sample_data.obs.columns) == {
        "sample_id",
        "accession",
        "cell_line",
        "condition",
        "replicate",
    }

    assert sample_data.obs["cell_line"].unique().tolist() == ["MCF7"]
    assert sample_data.obs["condition"].unique().tolist() == ["CTR"]
    assert sample_data.obs["replicate"].unique().tolist() == ["1"]

    assert sample_data.var_names.tolist() == [
        "GENE1",
        "GENE2",
    ]


def test_load_sample_preserves_counts(tmp_path: Path) -> None:
    sample_id = "GSM8595876_MCF7-ctr-1"

    counts = np.array(
        [
            [1, 0, 3],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )

    create_sample_files(tmp_path, sample_id, counts)

    sample_data = load_sample(tmp_path, sample_id)

    expected_cell_by_gene = counts.transpose()

    np.testing.assert_array_equal(
        sample_data.X.toarray(),
        expected_cell_by_gene,
    )


def test_load_dataset(
    complete_dataset_directory: Path,
) -> None:
    dataset = load_dataset(complete_dataset_directory)

    assert dataset.n_obs == 36
    assert dataset.n_vars == 2
    assert dataset.uns["dataset_accession"] == "GSE280381"
    assert dataset.uns["sample_count"] == 12

    assert dataset.obs["sample_id"].nunique() == 12
    assert set(dataset.obs["cell_line"].astype(str)) == {
        "MCF7",
        "T47D",
    }
    assert set(dataset.obs["condition"].astype(str)) == {
        "CTR",
        "TIS",
        "REPOP",
    }
    assert set(dataset.obs["replicate"].astype(str)) == {
        "1",
        "2",
    }

    assert dataset.obs_names.is_unique


def test_load_dataset_rejects_incomplete_dataset(
    tmp_path: Path,
) -> None:
    create_sample_files(
        tmp_path,
        "GSM8595876_MCF7-ctr-1",
    )

    with pytest.raises(ValueError, match="Expected 12 samples"):
        load_dataset(tmp_path)


def test_load_dataset_can_skip_expected_sample_validation(
    tmp_path: Path,
) -> None:
    create_sample_files(
        tmp_path,
        "GSM8595876_MCF7-ctr-1",
    )

    dataset = load_dataset(
        tmp_path,
        validate_expected_samples=False,
    )

    assert dataset.n_obs == 3
    assert dataset.n_vars == 2
    assert dataset.uns["sample_count"] == 1


def test_load_sample_rejects_missing_files(
    tmp_path: Path,
) -> None:
    sample_id = "GSM8595876_MCF7-ctr-1"

    write_gzip_text(
        tmp_path / f"{sample_id}_features.tsv.gz",
        "ENSG000001\tGENE1\tGene Expression\n",
    )

    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path, sample_id)


def test_save_dataset(
    complete_dataset_directory: Path,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(complete_dataset_directory)
    output_path = tmp_path / "processed" / "combined_raw.h5ad"

    save_dataset(dataset, output_path)

    assert output_path.exists()


def test_metadata_has_two_replicates_per_group(
    complete_dataset_directory: Path,
) -> None:
    dataset = load_dataset(complete_dataset_directory)

    sample_metadata = dataset.obs[
        ["sample_id", "cell_line", "condition", "replicate"]
    ].drop_duplicates()

    group_counts = sample_metadata.groupby(
        ["cell_line", "condition"],
        observed=True,
    ).size()

    assert (group_counts == 2).all()
