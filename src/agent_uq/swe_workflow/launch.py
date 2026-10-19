from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from agent_uq.swe_workflow.common import optional_path, require_path


@dataclass(frozen=True)
class ResolvedLaunchSource:
    target_record_ids: list[str]
    launch_args: list[str]
    runtime_byproducts: dict[str, pathlib.Path]

def build_native_source_launch_args(
    source_cfg: Any,
    *,
    evaluation_cfg: Any,
    launch_cfg: Any,
) -> list[str]:
    source_kind = str(source_cfg.kind)
    if source_kind == "swe_bench":
        args = [
            "--instances.type",
            "swe_bench",
            "--instances.subset",
            str(source_cfg.subset),
            "--instances.split",
            str(source_cfg.split),
            "--instances.evaluate="
            + ("True" if str(evaluation_cfg.mode) == "launch_results" else "False"),
        ]
        path_override = optional_path(source_cfg.get("path_override"))
        if path_override is not None:
            args.extend(["--instances.path_override", str(path_override)])
        if str(source_cfg.filter) != ".*":
            args.extend(["--instances.filter", str(source_cfg.filter)])
        if str(source_cfg.slice) != "":
            args.extend(["--instances.slice", str(source_cfg.slice)])
        if bool(source_cfg.shuffle):
            args.extend(["--instances.shuffle=True"])
    elif source_kind == "file":
        args = build_file_source_launch_args(
            instances_path=require_path(source_cfg.path, field_name="source.path"),
            launch_cfg=launch_cfg,
        )
        if str(source_cfg.filter) != ".*":
            args.extend(["--instances.filter", str(source_cfg.filter)])
        if str(source_cfg.slice) != "":
            args.extend(["--instances.slice", str(source_cfg.slice)])
        if bool(source_cfg.shuffle):
            args.extend(["--instances.shuffle=True"])
        return args
    else:
        raise ValueError(f"unsupported launch source kind: {source_kind!r}")

    args.extend(build_deployment_launch_args(launch_cfg))
    return args


def build_file_source_launch_args(*, instances_path: str | pathlib.Path, launch_cfg: Any) -> list[str]:
    args = [
        "--instances.type",
        "file",
        "--instances.path",
        str(instances_path),
    ]
    args.extend(build_deployment_launch_args(launch_cfg))
    return args


def build_deployment_launch_args(launch_cfg: Any) -> list[str]:
    python_standalone_dir = getattr(launch_cfg.deployment, "python_standalone_dir", None)
    if python_standalone_dir is None:
        return []
    return ["--instances.deployment.python_standalone_dir", str(python_standalone_dir)]


def resolve_selected_simple_batch_instances(source_cfg: Any) -> list[dict[str, Any]]:
    from sweagent.run.batch_instances import InstancesFromFile, SWEBenchInstances
    from swerex.deployment.config import DockerDeploymentConfig

    deployment = DockerDeploymentConfig(image="python:3.11")
    source_kind = str(source_cfg.kind)
    if source_kind == "swe_bench":
        source = SWEBenchInstances(
            subset=str(source_cfg.subset),
            path_override=source_cfg.get("path_override"),
            split=str(source_cfg.split),
            filter=str(source_cfg.filter),
            slice=str(source_cfg.slice),
            shuffle=bool(source_cfg.shuffle),
            evaluate=False,
            deployment=deployment,
        )
    elif source_kind == "file":
        source = InstancesFromFile(
            path=require_path(source_cfg.path, field_name="source.path"),
            filter=str(source_cfg.filter),
            slice=str(source_cfg.slice),
            shuffle=bool(source_cfg.shuffle),
            deployment=deployment,
        )
    else:
        raise ValueError(f"unsupported source.kind for launch selection: {source_kind!r}")
    return [_batch_instance_to_simple_dict(instance) for instance in source.get_instance_configs()]


def _batch_instance_to_simple_dict(instance: Any) -> dict[str, Any]:
    payload = instance.model_dump(mode="python")
    env = dict(payload.get("env") or {})
    problem = dict(payload.get("problem_statement") or {})
    repo = dict(env.get("repo") or {})
    deployment = dict(env.get("deployment") or {})
    extra_fields = dict(problem.get("extra_fields") or {})
    if "issue_images" in problem:
        extra_fields["issue_images"] = problem["issue_images"]

    repo_name = ""
    if "repo_name" in repo:
        repo_name = str(repo["repo_name"])
    elif "github_url" in repo:
        repo_name = str(repo["github_url"])
    elif "path" in repo:
        repo_name = str(repo["path"])

    return {
        "image_name": str(deployment.get("image") or ""),
        "problem_statement": str(problem.get("text") or ""),
        "instance_id": str(problem.get("id") or ""),
        "repo_name": repo_name,
        "base_commit": str(repo.get("base_commit") or "HEAD"),
        "extra_fields": extra_fields,
    }
