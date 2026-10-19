import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from agent_uq.records import list_record_paths, read_scored
from agent_uq.release.protocol import CONF, REASONING_METHODS
from agent_uq.uq.base import EstimatorInfrastructureError
from agent_uq.uq.reasoning_contender_cache import content_identity, read_result, write_json_atomic
from agent_uq.uq.reasoning_contender_inputs import enrich_preparation_manifest, prepare_inventory
from agent_uq.uq.reasoning_contender_runtime import _manifest


def build_inventory(leaf: Path, *, run_id: str) -> dict:
    records = []
    for path in list_record_paths(leaf / "scored", reject_duplicates=True):
        scored = read_scored(path)
        raw_path = Path(scored.raw_path)
        if not raw_path.is_absolute():
            raw_path = leaf / raw_path
        records.append(dict(record_id=scored.record_id, raw_path=str(raw_path.resolve()),
                            scored_path=str(path.resolve()),
                            turn_indices=[turn.turn_idx for turn in scored.turns]))
    return dict(runs=[dict(run_id=run_id, records=records)], totals=dict(
        runs=1, records=len(records), turns=sum(len(row["turn_indices"]) for row in records)))


def prepare_leaf(leaf: Path, *, run_id: str, root: Path, source_environment: str | None = None) -> Path:
    manifest = prepare_inventory(build_inventory(leaf, run_id=run_id), root / "prepared",
                                 source_environment=source_environment)
    enriched = root / "structured"
    enrich_preparation_manifest(manifest, enriched)
    return enriched


def _defaults() -> dict:
    return OmegaConf.to_container(OmegaConf.load(CONF / "release" / "reasoning.yaml"), resolve=True)


def build_runtime(auxiliary_model: Path, bert_snapshot: Path, *,
                  max_model_len: int = 65536, batch_size: int = 1,
                  gpu_memory_utilization: float | None = None) -> dict:
    defaults = _defaults()
    llm = dict(defaults["llm_kwargs"], model=str(auxiliary_model), tokenizer=str(auxiliary_model),
               max_model_len=max_model_len)
    if gpu_memory_utilization is not None:
        llm["gpu_memory_utilization"] = gpu_memory_utilization
    return dict(schema="reasoning_runtime_v1", enable_thinking=defaults["enable_thinking"],
                batch_size=batch_size, topo_num_samples=defaults["topo_num_samples"],
                bert_revision=defaults["bert_revision"], bert_snapshot=str(bert_snapshot),
                phases={phase: dict(llm_kwargs=dict(llm)) for phase in ("topo", "cot")})


def read_bindings(root: Path, *, run_id: str) -> dict[str, dict[str, str]]:
    manifest = json.loads((root / "runtime_identity_manifest.json").read_text())
    binding = dict(artifact_dir=manifest["artifact_dirs"][run_id],
                   artifact_identity=manifest["identity"])
    return {method: dict(binding) for method in REASONING_METHODS}


def validate_terminal_coverage(root: Path, *, run_id: str) -> dict:
    manifest = _manifest(root, prepared_only=True, require_structured=True)
    identity_manifest = json.loads((root / "runtime_identity_manifest.json").read_text())
    identity = identity_manifest["identity"]
    runtime_path = root / "runtime" / identity / "effective_runtime.json"
    if identity_manifest["prepared_inventory_identity"] != manifest["prepared_inventory_identity"]:
        raise EstimatorInfrastructureError("runtime identity does not match prepared inventory")
    try:
        runtime = json.loads(runtime_path.read_text())
    except (OSError, ValueError) as error:
        raise EstimatorInfrastructureError("runtime identity manifest has no readable effective runtime") from error
    if (content_identity(runtime) != identity_manifest["runtime_identity"]
            or content_identity(dict(prepared_inventory_identity=manifest["prepared_inventory_identity"],
                                     runtime=runtime)) != identity):
        raise EstimatorInfrastructureError("runtime identity does not match effective runtime")
    references = [reference for reference in manifest["records"] if reference["run_id"] == run_id]
    if not references:
        raise ValueError(f"unknown prepared run_id: {run_id}")
    directory = identity_manifest["artifact_dirs"][run_id]
    missing = dict.fromkeys(REASONING_METHODS, 0)
    completed = 0
    for reference in references:
        for turn_idx in reference["turn_indices"]:
            result = read_result(directory, reference["record_id"], turn_idx, expected_identity=identity)
            if result["input_hashes"]["input_identity"] != reference["input_identities"][str(turn_idx)]:
                raise EstimatorInfrastructureError("terminal input identity does not match preparation")
            for method in REASONING_METHODS:
                missing[method] += int(result["estimates"][method]["scalar"] is None)
            completed += 1
    return dict(run_id=run_id, identity=identity, records=len(references),
                turns=sum(len(reference["turn_indices"]) for reference in references),
                completed_results=completed, missing_results=missing)


def main(argv=None):
    defaults = _defaults()
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--leaf", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--source-environment", choices=("gruet", "gruet-gemma"))
    runtime = commands.add_parser("runtime")
    runtime.add_argument("--output", type=Path, required=True)
    runtime.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    runtime.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    runtime.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        root = prepare_leaf(args.leaf, run_id=args.run_id, root=args.root,
                            source_environment=args.source_environment)
        print(json.dumps(dict(artifact_root=str(root))))
    else:
        from huggingface_hub import snapshot_download

        auxiliary = snapshot_download(defaults["auxiliary_model"], revision=defaults["auxiliary_revision"])
        bert = snapshot_download(defaults["bert_model"], revision=defaults["bert_revision"])
        result = build_runtime(Path(auxiliary), Path(bert), max_model_len=args.max_model_len,
                               batch_size=args.batch_size, gpu_memory_utilization=args.gpu_memory_utilization)
        write_json_atomic(args.output, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
