from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_CONFIG = Path("config/project_config.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_feature_order(
    feature_table: pd.DataFrame,
    artifact: dict,
) -> dict:
    expected = [str(column) for column in artifact["feature_columns"]]
    missing = [column for column in expected if column not in feature_table.columns]
    ordered_available = [
        column for column in feature_table.columns if column in expected
    ]

    return {
        "expected_feature_count": len(expected),
        "missing_features": missing,
        "feature_order_matches": ordered_available == expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reproducibility audit for science-fair judging."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    tables_directory = Path(paths["tables"])

    required_files = [
        Path(paths["feature_table"]),
        Path(paths["target_rankings"]),
        Path(paths["pathway_results"]),
        Path(paths["model_artifact"]),
        tables_directory / "baseline_model_summary.csv",
        tables_directory / "validation_summary.csv",
        tables_directory / "top_predicted_interventions.csv",
    ]

    file_checks = []
    for path in required_files:
        file_checks.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )

    artifact_path = Path(paths["model_artifact"])
    feature_path = Path(paths["feature_table"])
    model_check = {}

    if artifact_path.exists() and feature_path.exists():
        artifact = joblib.load(artifact_path)
        features = pd.read_csv(feature_path)
        model_check = verify_model_feature_order(features, artifact)
        model_check.update(
            {
                "model_name": artifact.get("model_name"),
                "training_cell_count": artifact.get("training_cell_count"),
                "random_state": artifact.get("random_state"),
            }
        )

    report = {
        "project_name": config.get("project_name"),
        "configured_random_seed": config.get("random_seed"),
        "all_required_files_exist": all(row["exists"] for row in file_checks),
        "file_checks": file_checks,
        "model_check": model_check,
    }

    output = tables_directory / "reproducibility_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved reproducibility audit to {output}")
    print(f"All required files exist: {report['all_required_files_exist']}")
    if model_check:
        print(
            "Saved-model feature order matches table: "
            f"{model_check['feature_order_matches']}"
        )


if __name__ == "__main__":
    main()
