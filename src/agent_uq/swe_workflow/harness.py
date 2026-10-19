from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Sequence

from agent_uq.swe_workflow.bootstrap import _docker_image_exists
from agent_uq.swe_workflow.common import ensure_runtime_dir, optional_path, require_path, sha256_file
from agent_uq.swe_workflow.evaluator import (
    PreparedEvaluatorInputs,
    _bool_cli,
    _find_swebench_harness_report,
    _require_swebench_harness_cfg,
    assert_current_python_has_swebench,
    copy_if_needed,
    derive_swebench_harness_run_id,
    resolve_swebench_harness_dataset_name,
    resolve_swebench_harness_split,
    submit_predictions_to_sb_cli,
)
from agent_uq.swe_workflow.import_plan import normalize_evaluator_report


def _tag_bootstrap_images_for_swebench_harness(prediction_source_path: pathlib.Path) -> None:
    image_map_path = prediction_source_path.parent / "runtime" / "image_map.json"
    if not image_map_path.is_file():
        return

    image_map = json.loads(image_map_path.read_text(encoding="utf-8"))
    for harness_image, bootstrap_image in image_map.items():
        harness_image = str(harness_image)
        bootstrap_image = str(bootstrap_image)
        if _docker_image_exists(harness_image):
            continue
        if not _docker_image_exists(bootstrap_image):
            continue
        subprocess.run(["docker", "tag", bootstrap_image, harness_image], check=True)


def _write_swebench_raw_cache_sitecustomize(
    *,
    run_dir: pathlib.Path,
    prediction_source_path: pathlib.Path,
) -> bool:
    runtime_dir = prediction_source_path.parent / "runtime"
    instances_path = runtime_dir / "patched_instances.json"
    image_map_path = runtime_dir / "image_map.json"
    if not instances_path.is_file() or not image_map_path.is_file():
        return False

    cache_dir = runtime_dir / "swebench_raw_cache"
    sitecustomize_path = run_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        f"""
import json
import pathlib
import subprocess
import urllib.parse

import requests

_IMAGE_MAP_PATH = pathlib.Path({str(image_map_path)!r})
_INSTANCES_PATH = pathlib.Path({str(instances_path)!r})
_RAW_CACHE_DIR = pathlib.Path({str(cache_dir)!r})
_ORIGINAL_GET = requests.get
_IMAGE_INDEX = None


class _CachedRawResponse:
    status_code = 200

    def __init__(self, text):
        self.text = text
        self.content = text.encode("utf-8")


def _repo_from_source_image(source_image):
    image_name = source_image.rsplit("/", 1)[-1].split(":", 1)[0]
    prefix = "sweb.eval.x86_64."
    if not image_name.startswith(prefix):
        return None
    repo_issue = image_name[len(prefix):].replace("_1776_", "/")
    repo, sep, _issue = repo_issue.rpartition("-")
    return repo if sep else None


def _append_unique(values, value):
    if value not in values:
        values.append(value)


def _load_image_index():
    global _IMAGE_INDEX
    if _IMAGE_INDEX is not None:
        return _IMAGE_INDEX
    image_map = json.loads(_IMAGE_MAP_PATH.read_text(encoding="utf-8"))
    instances = json.loads(_INSTANCES_PATH.read_text(encoding="utf-8"))
    by_repo_commit = {{}}
    by_repo = {{}}
    for instance in instances:
        source_image = str(instance.get("source_image_name") or "")
        repo = instance.get("repo") or _repo_from_source_image(source_image)
        if not repo:
            continue
        repo = str(repo)
        image_name = str(instance.get("image_name") or image_map.get(source_image) or "")
        if not image_name:
            continue
        commit = str(instance.get("environment_setup_commit") or instance["base_commit"])
        by_repo_commit[(repo, commit)] = image_name
        _append_unique(by_repo.setdefault(repo, []), image_name)
    _IMAGE_INDEX = by_repo_commit, by_repo
    return _IMAGE_INDEX


def _cached_raw_path(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "raw.githubusercontent.com":
        return None
    parts = pathlib.PurePosixPath(parsed.path.lstrip("/")).parts
    if len(parts) < 4:
        return None
    repo = "/".join(parts[:2])
    commit = parts[2]
    filepath = pathlib.Path(*parts[3:])
    return repo, commit, filepath, _RAW_CACHE_DIR / repo / commit / filepath


def _get(url, *args, **kwargs):
    raw_info = _cached_raw_path(url)
    if raw_info is None:
        return _ORIGINAL_GET(url, *args, **kwargs)

    repo, commit, filepath, raw_path = raw_info
    if raw_path.is_file():
        return _CachedRawResponse(raw_path.read_text(encoding="utf-8"))
    by_repo_commit, by_repo = _load_image_index()
    image_names = []
    image_name = by_repo_commit.get((repo, commit))
    if image_name:
        image_names.append(image_name)
    for fallback_image_name in by_repo.get(repo, []):
        _append_unique(image_names, fallback_image_name)
    for image_name in image_names:
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", image_name, f"/testbed/{{filepath.as_posix()}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(result.stdout, encoding="utf-8")
            return _CachedRawResponse(result.stdout)
    return _ORIGINAL_GET(url, *args, **kwargs)


requests.get = _get
""".lstrip(),
        encoding="utf-8",
    )
    return True


def prepare_evaluator_inputs(
    *,
    evaluation_cfg: Any,
    source_cfg: Any,
    workflow_output_dir: str | pathlib.Path,
    target_record_ids: Sequence[str] | None,
    launch_output_dir: str | pathlib.Path | None = None,
    report_path_override: str | pathlib.Path | None = None,
    prediction_source_path_override: str | pathlib.Path | None = None,
    normalized_report_path_override: str | pathlib.Path | None = None,
    sb_cli_output_dir_override: str | pathlib.Path | None = None,
) -> PreparedEvaluatorInputs:
    runtime_dir = ensure_runtime_dir(workflow_output_dir)
    evaluation_mode = str(evaluation_cfg.mode)
    runtime_byproducts: dict[str, pathlib.Path] = {}

    if evaluation_mode == "existing_report":
        report_source_path = (
            pathlib.Path(report_path_override)
            if report_path_override is not None
            else require_path(evaluation_cfg.report_path, field_name="evaluation.report_path")
        )
        prediction_source_path = (
            pathlib.Path(prediction_source_path_override)
            if prediction_source_path_override is not None
            else optional_path(evaluation_cfg.prediction_source_path)
        )
    elif evaluation_mode == "launch_results":
        if launch_output_dir is None:
            raise ValueError("evaluation.mode='launch_results' requires launch output")
        report_source_path = (
            pathlib.Path(report_path_override)
            if report_path_override is not None
            else pathlib.Path(launch_output_dir) / "results.json"
        )
        prediction_source_path = (
            pathlib.Path(prediction_source_path_override)
            if prediction_source_path_override is not None
            else pathlib.Path(launch_output_dir) / "preds.json"
        )
    elif evaluation_mode == "sb_cli_submit":
        if launch_output_dir is None:
            raise ValueError("evaluation.mode='sb_cli_submit' requires launch output")
        prediction_source_path = (
            pathlib.Path(prediction_source_path_override)
            if prediction_source_path_override is not None
            else pathlib.Path(launch_output_dir) / "preds.json"
        )
        report_source_path = submit_predictions_to_sb_cli(
            evaluation_cfg=evaluation_cfg,
            source_cfg=source_cfg,
            prediction_source_path=prediction_source_path,
            target_record_ids=target_record_ids,
            output_dir_override=sb_cli_output_dir_override,
        )
        runtime_byproducts["sb_cli_report_path"] = report_source_path
        runtime_byproducts["sb_cli_report_dir"] = report_source_path.parent
    elif evaluation_mode == "swebench_harness":
        if launch_output_dir is None:
            raise ValueError("evaluation.mode='swebench_harness' requires launch output")
        prediction_source_path = (
            pathlib.Path(prediction_source_path_override)
            if prediction_source_path_override is not None
            else pathlib.Path(launch_output_dir) / "preds.json"
        )
        report_source_path = run_predictions_with_swebench_harness(
            evaluation_cfg=evaluation_cfg,
            source_cfg=source_cfg,
            prediction_source_path=prediction_source_path,
            target_record_ids=target_record_ids,
        )
        runtime_byproducts["swebench_harness_report_path"] = report_source_path
        runtime_byproducts["swebench_harness_report_dir"] = report_source_path.parent
    else:
        raise ValueError(f"unsupported SWE evaluation mode: {evaluation_mode!r}")

    normalized_report_path = (
        pathlib.Path(normalized_report_path_override)
        if normalized_report_path_override is not None
        else require_path(
            evaluation_cfg.normalized_report_path,
            field_name="evaluation.normalized_report_path",
        )
    )
    normalized_report_path.parent.mkdir(parents=True, exist_ok=True)
    copy_if_needed(report_source_path, normalized_report_path)
    if target_record_ids:
        normalize_evaluator_report(normalized_report_path, target_record_ids=target_record_ids)
    runtime_byproducts["normalized_report_path"] = normalized_report_path
    return PreparedEvaluatorInputs(
        evaluator_report_path=normalized_report_path,
        prediction_source_path=prediction_source_path,
        runtime_byproducts=runtime_byproducts,
    )


def run_predictions_with_swebench_harness(
    *,
    evaluation_cfg: Any,
    source_cfg: Any,
    prediction_source_path: str | pathlib.Path,
    target_record_ids: Sequence[str] | None = None,
) -> pathlib.Path:
    assert_current_python_has_swebench()
    harness_cfg = _require_swebench_harness_cfg(evaluation_cfg)
    prediction_source_path = pathlib.Path(prediction_source_path)
    if not prediction_source_path.is_absolute():
        prediction_source_path = prediction_source_path.resolve()
    dataset_name = resolve_swebench_harness_dataset_name(
        source_cfg=source_cfg,
        evaluation_cfg=evaluation_cfg,
    )
    split = resolve_swebench_harness_split(
        source_cfg=source_cfg,
        evaluation_cfg=evaluation_cfg,
    )
    run_id = derive_swebench_harness_run_id(
        prediction_source_path=prediction_source_path,
        target_record_ids=target_record_ids,
        dataset_name=dataset_name,
        split=split,
        harness_cfg=harness_cfg,
    )
    base_output_dir = require_path(
        harness_cfg.output_dir,
        field_name="evaluation.swebench_harness.output_dir",
    )
    if not base_output_dir.is_absolute():
        base_output_dir = base_output_dir.resolve()
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _tag_bootstrap_images_for_swebench_harness(prediction_source_path)
    use_raw_cache = _write_swebench_raw_cache_sitecustomize(
        run_dir=run_dir,
        prediction_source_path=prediction_source_path,
    )
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(prediction_source_path),
        "--max_workers",
        str(int(harness_cfg.max_workers)),
        "--open_file_limit",
        str(int(harness_cfg.open_file_limit)),
        "--timeout",
        str(int(harness_cfg.timeout)),
        "--force_rebuild",
        _bool_cli(harness_cfg.force_rebuild),
        "--cache_level",
        str(harness_cfg.cache_level),
        "--clean",
        _bool_cli(harness_cfg.clean),
        "--run_id",
        run_id,
        "--namespace",
        ("none" if harness_cfg.get("namespace") in (None, "") else str(harness_cfg.namespace)),
        "--instance_image_tag",
        str(harness_cfg.instance_image_tag),
        "--env_image_tag",
        str(harness_cfg.env_image_tag),
        "--rewrite_reports",
        _bool_cli(harness_cfg.rewrite_reports),
        "--modal",
        _bool_cli(harness_cfg.modal),
    ]
    if target_record_ids:
        cmd.extend(["--instance_ids", *[str(record_id) for record_id in target_record_ids]])
    env = None
    if use_raw_cache:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(run_dir)
            if not existing_pythonpath
            else os.pathsep.join([str(run_dir), existing_pythonpath])
        )
    run_kwargs: dict[str, Any] = {"check": True, "cwd": run_dir}
    if env is not None:
        run_kwargs["env"] = env
    subprocess.run(cmd, **run_kwargs)
    return _find_swebench_harness_report(run_dir, run_id=run_id)


__all__ = [
    "prepare_evaluator_inputs",
    "run_predictions_with_swebench_harness",
    "_tag_bootstrap_images_for_swebench_harness",
    "_write_swebench_raw_cache_sitecustomize",
]
