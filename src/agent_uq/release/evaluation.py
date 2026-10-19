import argparse
import csv
import json
import math
from pathlib import Path

import agent_uq.uq.aggregators.builtin
from agent_uq.evaluation.metric_policy import _build_arrays_for_row
from agent_uq.metrics import safe_metric_call
from agent_uq.records import list_record_paths, read_scored
from agent_uq.release.protocol import METHODS, load_estimator_config, load_runs

FIELDS = (
    "run_id", "model", "dataset", "estimator_config", "estimator", "aggregator",
    "auroc", "n_total", "n_valid", "n_excluded", "status",
)


def evaluate_records(records, *, run_id, estimator_config, dataset=None):
    rows = []
    first = records[0] if records else None
    for estimator in METHODS:
        values, targets, excluded, target_type = _build_arrays_for_row(records, estimator, "mean")
        value = safe_metric_call("ROCAUC", values, targets, target_type)
        if value is not None and not math.isfinite(value):
            value = None
        if not values:
            status = "no_valid_records"
        elif len(set(targets)) < 2:
            status = "single_class"
        elif value is None:
            status = "metric_unavailable"
        else:
            status = "ok"
        rows.append(dict(
            run_id=run_id,
            model=first.model if first else "",
            dataset=dataset if dataset is not None else (first.dataset_key if first else ""),
            estimator_config=estimator_config,
            estimator=estimator,
            aggregator="mean",
            auroc=value,
            n_total=len(records),
            n_valid=len(values),
            n_excluded=excluded,
            status=status,
        ))
    return rows


def evaluate_leaf(leaf: Path, *, run_id, estimator_config, dataset=None):
    records = [read_scored(path) for path in list_record_paths(leaf / "scored", reject_duplicates=True)]
    return evaluate_records(records, run_id=run_id, estimator_config=estimator_config, dataset=dataset)


def write_rows(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "auroc.csv"
    json_path = output_dir / "auroc.json"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--estimator-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    load_estimator_config(args.estimator_config)
    run = next((run for run in load_runs() if run["run_id"] == args.run_id), None)
    if run is None:
        parser.error(f"unknown run ID: {args.run_id}")
    rows = evaluate_leaf(
        args.leaf,
        run_id=args.run_id,
        estimator_config=args.estimator_config,
        dataset=run["dataset"],
    )
    write_rows(rows, args.output_dir)


if __name__ == "__main__":
    main()
