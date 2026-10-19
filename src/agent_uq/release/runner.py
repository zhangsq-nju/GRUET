import argparse
import json
from pathlib import Path

from agent_uq.records import list_record_paths, read_raw, read_scored
from agent_uq.release.evaluation import evaluate_leaf, write_rows
from agent_uq.release.protocol import (
    ESTIMATOR_CONFIGS, GUTQ_REVISION, METHODS, REASONING_METHODS,
    load_estimator_config, load_models, load_runs,
)
from agent_uq.release.reasoning import read_bindings, validate_terminal_coverage
from agent_uq.release.stages import generate_leaf, invoke, native_command, score_leaf


def bind_output_protocol(output_root: Path, estimator_config: str) -> None:
    config = load_estimator_config(estimator_config)
    expected = dict(estimator_config=estimator_config, gutq_revision=GUTQ_REVISION,
                    algorithm={key: config[key] for key in (
                        "graph_type", "merge_way", "sim_way", "aggregation", "weight",
                        "token_level", "step_aggregation", "beta", "alpha", "window_size",
                        "action_similarity_model_name", "action_similarity_threshold")})
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError("output root belongs to a different global configuration")
    path.write_text(json.dumps(expected, indent=2) + "\n")


def _validate_scores(leaf: Path, methods: tuple[str, ...]) -> None:
    raw_paths = list_record_paths(leaf / "raw", reject_duplicates=True)
    scored_records = {record.record_id: record
                      for path in list_record_paths(leaf / "scored", reject_duplicates=True)
                      for record in [read_scored(path)]}
    if not raw_paths:
        raise ValueError("incomplete scoring: raw/scored record coverage differs")
    for raw_path in raw_paths:
        raw = read_raw(raw_path)
        record_id = raw.record_id
        scored = scored_records.pop(record_id, None)
        if scored is None:
            raise ValueError("incomplete scoring: raw/scored record coverage differs")
        source = Path(scored.raw_path)
        if not source.is_absolute():
            source = leaf / source
        if (source.resolve() != raw_path.resolve()
                or [turn.turn_idx for turn in scored.turns] != [turn.turn_idx for turn in raw.turns]):
            raise ValueError(f"incomplete scoring: source/turn coverage differs for {record_id}")
        for method in methods:
            metadata = scored.estimators.get(method, {})
            if not isinstance(metadata.get("params"), dict):
                raise ValueError(f"incomplete scoring: missing {method} metadata for {record_id}")
            for turn in scored.turns:
                if method not in turn.uncertainty_estimates:
                    raise ValueError(f"incomplete scoring: missing {method} result for {record_id}/{turn.turn_idx}")
    if scored_records:
        raise ValueError("incomplete scoring: raw/scored record coverage differs")


def run_one(run: dict, *, output_root: Path, models: dict,
            estimator_config: str = "gut_up",
            max_records: int | None = None) -> list[dict]:
    output_root = output_root.resolve()
    bind_output_protocol(output_root, estimator_config)
    leaf = generate_leaf(run, output_root=output_root, models=models, max_records=max_records)
    ordinary = tuple(method for method in METHODS if method not in REASONING_METHODS)
    score_leaf(run, leaf, models=models, methods=ordinary, estimator_config=estimator_config)
    _validate_scores(leaf, ordinary)
    root = output_root / run["run_id"] / "reasoning"
    structured_root = root / "structured"
    runtime_path = root / "runtime.json"
    invoke(native_command(run["environment"], "agent_uq.release.reasoning", [
        "prepare", "--leaf", str(leaf), "--run-id", run["run_id"], "--root", str(root),
        "--source-environment", run["environment"],
    ]))
    runtime_arguments = ["runtime", "--output", str(runtime_path)]
    if "gpu_memory_utilization" in models[run["model"]]:
        runtime_arguments.extend(["--gpu-memory-utilization", str(models[run["model"]]["gpu_memory_utilization"])])
    invoke(native_command("gruet", "agent_uq.release.reasoning", runtime_arguments))
    common = ["--artifact-root", str(structured_root), "--runtime-json", str(runtime_path), "--prepared-only"]
    for action in (["preflight"], ["produce", "--phase", "topo"], ["produce", "--phase", "cot"]):
        invoke(native_command("gruet", "agent_uq.cli.reasoning_contenders", action + common))
    validate_terminal_coverage(structured_root, run_id=run["run_id"])
    bindings = read_bindings(structured_root, run_id=run["run_id"])
    score_leaf(run, leaf, models=models, methods=REASONING_METHODS,
               estimator_config=estimator_config, bindings=bindings)
    _validate_scores(leaf, METHODS)
    rows = evaluate_leaf(leaf, run_id=run["run_id"], estimator_config=estimator_config,
                         dataset=run["dataset"])
    write_rows(rows, output_root / run["run_id"] / "results")
    return rows


def main(argv=None):
    runs = load_runs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=["all", *(run["run_id"] for run in runs)])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--estimator-config", choices=ESTIMATOR_CONFIGS, default="gut_up")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    selected = runs if args.run == "all" else [run for run in runs if run["run_id"] == args.run]
    models, rows = load_models(args.models), []
    for run in selected:
        rows.extend(run_one(run, output_root=output_root, models=models,
                            estimator_config=args.estimator_config, max_records=args.max_records))
    write_rows(rows, output_root / "results")


if __name__ == "__main__":
    main()
