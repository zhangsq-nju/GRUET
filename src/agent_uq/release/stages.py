import json
import os
import subprocess
from pathlib import Path

from agent_uq.release.protocol import REASONING_METHODS

ROOT = Path(__file__).resolve().parents[3]


def native_command(environment: str, module: str, arguments: list[str]) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", environment,
            "python", "-m", module, *arguments]


def invoke(argv: list[str], *, gpu_memory_utilization: float | None = None,
           source_model: dict | None = None) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "vendor" / "SWE-agent")))
    if gpu_memory_utilization is not None:
        environment["AGENT_UQ_AUXILIARY_VLLM_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
    if source_model is not None:
        environment["AGENT_UQ_AUXILIARY_SOURCE_MODEL"] = json.dumps(source_model)
    subprocess.run(argv, cwd=ROOT, env=environment, check=True)


def override(key, value):
    return f"{key}={json.dumps(value)}"


def generate_leaf(run: dict, *, output_root: Path, models: dict,
                  max_records: int | None = None) -> Path:
    from agent_uq.swe_imports import resolve_swe_import_leaf

    model = models[run["model"]]
    root = (output_root / run["run_id"]).resolve()
    arguments = [override("output_dir", str(root))]
    if run["kind"] == "intercode":
        module = "agent_uq.cli.run"
        arguments.extend([
            f"intercode_run={run['hydra_config']}",
            override("intercode_run.output_root", str(root)),
            override("model.model_path", model["model_path"]),
            override("model.tokenizer_path", model["tokenizer_path"]),
            override("model.tensor_parallel_size", model["tensor_parallel_size"]),
        ])
        if max_records is not None:
            arguments.append(override("max_records", max_records))
    else:
        module = "agent_uq.cli.swe_workflow"
        arguments.extend([
            f"swe_run={run['hydra_config']}",
            override("swe_run.output_root", str(root)),
            override("launch.model.name", model["model_path"]),
            override("launch.model.tokenizer_path", model["tokenizer_path"]),
            override("launch.model.tensor_parallel_size", model["tensor_parallel_size"]),
        ])
        if max_records is not None:
            arguments.append(override("source.slice", f":{max_records}"))
    if "gpu_memory_utilization" in model:
        key = "model.gpu_memory_utilization" if run["kind"] == "intercode" else "launch.model.gpu_memory_utilization"
        arguments.append(override(key, model["gpu_memory_utilization"]))
    invoke(native_command(run["environment"], module, arguments))
    return root if run["kind"] == "intercode" else resolve_swe_import_leaf(root / "import", base_dir=ROOT)


def score_leaf(run: dict, leaf: Path, *, models: dict, methods: tuple[str, ...],
               estimator_config: str, bindings: dict | None = None) -> None:
    model = models[run["model"]]
    for method in methods:
        selected = estimator_config if method == "gut_up" else method
        arguments = [
            f"model={run['model']}", f"estimator={selected}",
            override("output_dir", str(leaf)),
            override("model.model_path", model["model_path"]),
            override("model.tokenizer_path", model["tokenizer_path"]),
            override("model.tensor_parallel_size", model["tensor_parallel_size"]),
        ]
        if method in REASONING_METHODS:
            if bindings is None or not {"artifact_dir", "artifact_identity"}.issubset(bindings.get(method, {})):
                raise ValueError(f"missing generated artifacts for {method}")
            arguments.extend(override(f"estimator.{key}", bindings[method][key])
                             for key in ("artifact_dir", "artifact_identity"))
        if "gpu_memory_utilization" in model:
            arguments.append(override("model.gpu_memory_utilization", model["gpu_memory_utilization"]))
        if method in ("p_true", "verbalized_2s"):
            invoke(native_command(run["environment"], "agent_uq.cli.score", arguments),
                   source_model=dict(model_ref=run["model"], **model),
                   gpu_memory_utilization=model.get("gpu_memory_utilization"))
        elif "gpu_memory_utilization" in model:
            invoke(native_command(run["environment"], "agent_uq.cli.score", arguments),
                   gpu_memory_utilization=model["gpu_memory_utilization"])
        else:
            invoke(native_command(run["environment"], "agent_uq.cli.score", arguments))
