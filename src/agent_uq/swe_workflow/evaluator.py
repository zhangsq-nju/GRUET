from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Sequence

from agent_uq.swe_workflow.common import require_path, sha256_file


@dataclass(frozen=True)
class PreparedEvaluatorInputs:
    evaluator_report_path: pathlib.Path
    prediction_source_path: pathlib.Path | None
    runtime_byproducts: dict[str, pathlib.Path]

def _require_swebench_harness_cfg(evaluation_cfg: Any) -> Any:
    cfg = evaluation_cfg.get("swebench_harness")
    if cfg is None:
        raise ValueError("evaluation.swebench_harness must be configured")
    return cfg


def _str_or_auto(value: Any) -> str:
    return "auto" if value in (None, "") else str(value)


def _bool_cli(value: Any) -> str:
    return "true" if bool(value) else "false"


def assert_current_python_has_swebench() -> None:
    if importlib.util.find_spec("swebench") is None:
        raise ModuleNotFoundError(
            "evaluation.mode='swebench_harness' requires the current Python "
            "environment to install swebench"
        )


def resolve_swebench_harness_dataset_name(*, source_cfg: Any, evaluation_cfg: Any) -> str:
    harness_cfg = _require_swebench_harness_cfg(evaluation_cfg)
    configured = _str_or_auto(harness_cfg.get("dataset_name"))
    if configured != "auto":
        return configured
    if str(source_cfg.kind) != "swe_bench":
        raise ValueError(
            "evaluation.swebench_harness.dataset_name must be set explicitly "
            "when source.kind is not 'swe_bench'"
        )
    mapping = {
        "full": "SWE-bench/SWE-bench",
        "lite": "SWE-bench/SWE-bench_Lite",
        "verified": "SWE-bench/SWE-bench_Verified",
    }
    subset = str(source_cfg.subset)
    if subset not in mapping:
        raise ValueError(
            "evaluation.swebench_harness.dataset_name cannot be auto-resolved "
            f"for swe_bench subset {subset!r}"
        )
    return mapping[subset]


def resolve_swebench_harness_split(*, source_cfg: Any, evaluation_cfg: Any) -> str:
    harness_cfg = _require_swebench_harness_cfg(evaluation_cfg)
    configured = _str_or_auto(harness_cfg.get("split"))
    if configured != "auto":
        return configured
    if hasattr(source_cfg, "get") and source_cfg.get("split") not in (None, ""):
        return str(source_cfg.split)
    return "test"


def derive_swebench_harness_run_id(
    *,
    prediction_source_path: str | pathlib.Path,
    target_record_ids: Sequence[str] | None,
    dataset_name: str,
    split: str,
    harness_cfg: Any,
) -> str:
    payload = {
        "prediction_sha256": sha256_file(prediction_source_path),
        "target_record_ids": [str(record_id) for record_id in (target_record_ids or [])],
        "dataset_name": str(dataset_name),
        "split": str(split),
        "max_workers": int(harness_cfg.max_workers),
        "timeout": int(harness_cfg.timeout),
        "cache_level": str(harness_cfg.cache_level),
        "namespace": None if harness_cfg.get("namespace") in (None, "") else str(harness_cfg.namespace),
        "instance_image_tag": str(harness_cfg.instance_image_tag),
        "env_image_tag": str(harness_cfg.env_image_tag),
        "rewrite_reports": bool(harness_cfg.rewrite_reports),
        "modal": bool(harness_cfg.modal),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"swebench_harness_{digest}"


def _find_swebench_harness_report(run_dir: pathlib.Path, *, run_id: str) -> pathlib.Path:
    reports = sorted(run_dir.glob(f"*.{run_id}.json"))
    if len(reports) != 1:
        raise ValueError(
            f"expected exactly one SWE-bench harness report under {run_dir} "
            f"for run_id={run_id!r}, found {len(reports)}"
        )
    return reports[0]

def submit_predictions_to_sb_cli(
    *,
    evaluation_cfg: Any,
    source_cfg: Any,
    prediction_source_path: str | pathlib.Path,
    target_record_ids: Sequence[str] | None = None,
    output_dir_override: str | pathlib.Path | None = None,
) -> pathlib.Path:
    if shutil.which("sb-cli") is None:
        raise FileNotFoundError("sb-cli is required for evaluation.mode='sb_cli_submit'")

    subset = resolve_sb_cli_subset(source_cfg=source_cfg, evaluation_cfg=evaluation_cfg)
    split = resolve_sb_cli_split(source_cfg=source_cfg, evaluation_cfg=evaluation_cfg)
    api_key_env = str(evaluation_cfg.sb_cli.api_key_env)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"{api_key_env} is required for evaluation.mode='sb_cli_submit'")

    base_output_dir = (
        pathlib.Path(output_dir_override)
        if output_dir_override is not None
        else require_path(evaluation_cfg.sb_cli.output_dir, field_name="evaluation.sb_cli.output_dir")
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = pathlib.Path(tempfile.mkdtemp(prefix="submit-", dir=str(base_output_dir)))
    cmd = [
        "sb-cli",
        "submit",
        subset,
        split,
        "--predictions_path",
        str(prediction_source_path),
        "--output_dir",
        str(output_dir),
        "--api_key",
        api_key,
    ]
    if target_record_ids:
        cmd.extend(["--instance_ids", ",".join(str(record_id) for record_id in target_record_ids)])
    subprocess.run(cmd, check=True)
    expected_report = output_dir / f"{subset}__{split}__{pathlib.Path(prediction_source_path).parent.name}.json"
    if expected_report.is_file():
        return expected_report
    reports = sorted(
        path for path in output_dir.glob("*.json") if not path.name.endswith(".response.json")
    )
    if len(reports) != 1:
        raise ValueError(f"expected exactly one sb-cli report under {output_dir}, found {len(reports)}")
    return reports[0]


def resolve_sb_cli_subset(*, source_cfg: Any, evaluation_cfg: Any) -> str:
    configured = str(evaluation_cfg.sb_cli.subset)
    if configured != "auto":
        return configured
    if str(source_cfg.kind) != "swe_bench":
        raise ValueError("evaluation.sb_cli.subset must be set explicitly when source.kind is not 'swe_bench'")
    mapping = {
        "full": "swe-bench",
        "lite": "swe-bench_lite",
        "verified": "swe-bench_verified",
        "multimodal": "swe-bench_multimodal",
    }
    subset = str(source_cfg.subset)
    if subset not in mapping:
        raise ValueError(f"unsupported swe_bench subset for sb-cli submission: {subset!r}")
    return mapping[subset]


def resolve_sb_cli_split(*, source_cfg: Any, evaluation_cfg: Any) -> str:
    configured = str(evaluation_cfg.sb_cli.split)
    if configured != "auto":
        return configured
    if str(source_cfg.kind) != "swe_bench":
        raise ValueError("evaluation.sb_cli.split must be set explicitly when source.kind is not 'swe_bench'")
    return str(source_cfg.split)


def copy_if_needed(src: str | pathlib.Path, dst: str | pathlib.Path) -> None:
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst)
    if src_path.resolve() == dst_path.resolve():
        return
    shutil.copy2(src_path, dst_path)
