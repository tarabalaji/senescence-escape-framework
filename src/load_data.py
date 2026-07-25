from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse
from scipy.io import mmread

SAMPLE_PATTERN = re.compile(
    r"^(?P<accession>GSM\d+)_"
    r"(?P<cell_line>MCF7|T47D)-"
    r"(?P<condition>ctr|tis|repop)-"
    r"(?P<replicate>\d+)$",
    re.IGNORECASE,
)

EXPECTED_CELL_LINES = {"MCF7", "T47D"}
EXPECTED_CONDITIONS = {"CTR", "TIS", "REPOP"}
EXPECTED_REPLICATES = {"1", "2"}
EXPECTED_SAMPLE_COUNT = 12


def discover_samples(raw_data_dir: str | Path) -> list[str]:
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    matrix_files = sorted(raw_data_dir.glob("*_matrix.mtx.gz"))

    if not matrix_files:
        raise FileNotFoundError(
            f"No '_matrix.mtx.gz' files were found in {raw_data_dir}"
        )

    sample_ids = [
        file_path.name.removesuffix("_matrix.mtx.gz") for file_path in matrix_files
    ]

    return sample_ids


def parse_sample_metadata(sample_id: str) -> dict[str, str]:
    match = SAMPLE_PATTERN.fullmatch(sample_id)

    if match is None:
        raise ValueError(
            "Sample name does not match the expected format "
            f"'GSMXXXXXXX_CELL-LINE-condition-replicate': {sample_id}"
        )

    metadata = match.groupdict()

    return {
        "sample_id": sample_id,
        "accession": metadata["accession"].upper(),
        "cell_line": metadata["cell_line"].upper(),
        "condition": metadata["condition"].upper(),
        "replicate": metadata["replicate"],
    }


def validate_sample_files(
    raw_data_dir: str | Path,
    sample_id: str,
) -> dict[str, Path]:
    raw_data_dir = Path(raw_data_dir)

    sample_files = {
        "matrix": raw_data_dir / f"{sample_id}_matrix.mtx.gz",
        "features": raw_data_dir / f"{sample_id}_features.tsv.gz",
        "barcodes": raw_data_dir / f"{sample_id}_barcodes.tsv.gz",
    }

    missing_files = [
        str(file_path) for file_path in sample_files.values() if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(
            f"Sample {sample_id} is missing required files:\n{missing_text}"
        )

    return sample_files


def make_unique(values: pd.Series) -> list[str]:
    counts: dict[str, int] = {}
    unique_values: list[str] = []

    for value in values.astype(str):
        count = counts.get(value, 0)

        if count == 0:
            unique_values.append(value)
        else:
            unique_values.append(f"{value}-{count}")

        counts[value] = count + 1

    return unique_values


def load_sample(
    raw_data_dir: str | Path,
    sample_id: str,
) -> ad.AnnData:
    metadata = parse_sample_metadata(sample_id)
    sample_files = validate_sample_files(raw_data_dir, sample_id)

    count_matrix = mmread(sample_files["matrix"])

    if not sparse.issparse(count_matrix):
        count_matrix = sparse.coo_matrix(count_matrix)

    count_matrix = count_matrix.tocsr().transpose().tocsr()

    features = pd.read_csv(
        sample_files["features"],
        sep="\t",
        header=None,
        compression="gzip",
    )

    barcodes = pd.read_csv(
        sample_files["barcodes"],
        sep="\t",
        header=None,
        compression="gzip",
    )

    if features.shape[1] < 2:
        raise ValueError(
            f"Feature file for {sample_id} must contain at least two columns"
        )

    gene_ids = features.iloc[:, 0].astype(str)
    gene_names = features.iloc[:, 1].astype(str)
    barcode_values = barcodes.iloc[:, 0].astype(str)

    if count_matrix.shape[0] != len(barcode_values):
        raise ValueError(
            f"Barcode count does not match matrix rows for {sample_id}: "
            f"{len(barcode_values)} barcodes and "
            f"{count_matrix.shape[0]} matrix rows"
        )

    if count_matrix.shape[1] != len(gene_names):
        raise ValueError(
            f"Feature count does not match matrix columns for {sample_id}: "
            f"{len(gene_names)} features and "
            f"{count_matrix.shape[1]} matrix columns"
        )

    unique_gene_names = make_unique(gene_names)

    observation_names = [f"{sample_id}_{barcode}" for barcode in barcode_values]

    obs = pd.DataFrame(index=observation_names)
    obs.index.name = "cell_id"

    for field, value in metadata.items():
        obs[field] = value

    var = pd.DataFrame(
        {
            "gene_id": gene_ids.to_numpy(),
            "gene_symbol": gene_names.to_numpy(),
        },
        index=unique_gene_names,
    )
    var.index.name = "gene"

    sample_data = ad.AnnData(
        X=count_matrix,
        obs=obs,
        var=var,
    )

    return sample_data


def validate_dataset_metadata(metadata_table: pd.DataFrame) -> None:
    sample_count = metadata_table["sample_id"].nunique()

    if sample_count != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples, but found {sample_count}"
        )

    observed_cell_lines = set(metadata_table["cell_line"])
    observed_conditions = set(metadata_table["condition"])
    observed_replicates = set(metadata_table["replicate"])

    if observed_cell_lines != EXPECTED_CELL_LINES:
        raise ValueError(
            f"Expected cell lines {EXPECTED_CELL_LINES}, "
            f"but found {observed_cell_lines}"
        )

    if observed_conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Expected conditions {EXPECTED_CONDITIONS}, "
            f"but found {observed_conditions}"
        )

    if observed_replicates != EXPECTED_REPLICATES:
        raise ValueError(
            f"Expected replicates {EXPECTED_REPLICATES}, "
            f"but found {observed_replicates}"
        )

    group_counts = (
        metadata_table[["cell_line", "condition", "replicate", "sample_id"]]
        .drop_duplicates()
        .groupby(["cell_line", "condition"])
        .size()
    )

    if not (group_counts == 2).all():
        raise ValueError(
            "Each cell-line and condition combination must contain "
            "exactly two biological replicates"
        )


def load_dataset(
    raw_data_dir: str | Path,
    validate_expected_samples: bool = True,
) -> ad.AnnData:
    sample_ids = discover_samples(raw_data_dir)

    metadata_table = pd.DataFrame(
        [parse_sample_metadata(sample_id) for sample_id in sample_ids]
    )

    if validate_expected_samples:
        validate_dataset_metadata(metadata_table)

    sample_objects = {
        sample_id: load_sample(raw_data_dir, sample_id) for sample_id in sample_ids
    }

    combined_data = ad.concat(
        sample_objects,
        join="outer",
        merge="same",
        label=None,
        index_unique=None,
        fill_value=0,
    )

    combined_data.obs["cell_line"] = combined_data.obs["cell_line"].astype("category")

    combined_data.obs["condition"] = combined_data.obs["condition"].astype(
        pd.CategoricalDtype(
            categories=["CTR", "TIS", "REPOP"],
            ordered=True,
        )
    )

    combined_data.obs["replicate"] = combined_data.obs["replicate"].astype("category")

    combined_data.uns["dataset_accession"] = "GSE280381"
    combined_data.uns["sample_count"] = len(sample_ids)

    return combined_data


def save_dataset(
    dataset: ad.AnnData,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_h5ad(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and combine the GSE280381 single-cell dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/GSE280381_RAW"),
        help="Directory containing the raw GSE280381 files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/combined_raw.h5ad"),
        help="Path for the combined AnnData file.",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.input)
    save_dataset(dataset, args.output)

    print(f"Loaded {dataset.n_obs:,} cells")
    print(f"Loaded {dataset.n_vars:,} genes")
    print(f"Loaded {dataset.uns['sample_count']} samples")
    print(f"Saved combined dataset to {args.output}")


if __name__ == "__main__":
    main()
