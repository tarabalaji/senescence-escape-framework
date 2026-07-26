from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CONFIG = Path("config/project_config.json")


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def run_module(module: str) -> dict:
    command = [sys.executable, "-m", f"src.{module}"]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")

    return {
        "module": module,
        "return_code": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def run_pipeline(
    modules: list[str],
    skip: set[str] | None = None,
    stop_on_error: bool = True,
) -> list[dict]:
    skip = skip or set()
    results = []

    for module in modules:
        if module in skip:
            results.append(
                {
                    "module": module,
                    "return_code": None,
                    "elapsed_seconds": 0.0,
                    "status": "skipped",
                }
            )
            print(f"Skipping {module}")
            continue

        print(f"\n=== Running {module} ===")
        result = run_module(module)
        results.append(result)

        if result["status"] == "failed" and stop_on_error:
            break

    return results


def save_run_summary(results: list[dict], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SenEscape analysis pipeline in a reproducible order."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--include-load-data",
        action="store_true",
        help="Run load_data instead of skipping it.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only the named modules, in the order provided.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a module fails.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/pipeline_run_summary.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline_config = config["pipeline"]
    modules = args.only or pipeline_config["modules"]
    skip = set(pipeline_config.get("skip_by_default", []))

    if args.include_load_data:
        skip.discard("load_data")

    results = run_pipeline(
        modules,
        skip=skip,
        stop_on_error=not args.continue_on_error,
    )
    save_run_summary(results, args.summary)

    failed = [result for result in results if result["status"] == "failed"]
    passed = [result for result in results if result["status"] == "passed"]
    skipped = [result for result in results if result["status"] == "skipped"]

    print(
        f"\nPipeline summary: {len(passed)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    print(f"Saved run summary to {args.summary}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
