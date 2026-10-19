from __future__ import annotations

import logging
import os
import pathlib
import subprocess
from typing import Callable

from omegaconf import DictConfig

from agent_uq.swe_workflow.common import REPO_ROOT
from agent_uq.swe_workflow.config_validation import (
    _effective_launch_candidate_sampling,
    _effective_launch_sampling,
    _launch_agent_max_steps,
    _launch_extra_args,
    _launch_model_chat_template_kwargs,
    _launch_model_completion_kwargs,
    _serialize_completion_kwargs_for_cli,
    _validate_launch_extra_args,
)
from agent_uq.swe_workflow.import_plan import ResolvedAuthoritativeSWEConfig
from agent_uq.swe_workflow.launch import ResolvedLaunchSource

log = logging.getLogger("agent_uq.swe_workflow.launch_command")


BuildLaunchCommand = Callable[
    [DictConfig, ResolvedLaunchSource, ResolvedAuthoritativeSWEConfig],
    tuple[list[str], pathlib.Path],
]


def _build_launch_command(
    cfg: DictConfig,
    resolved_source: ResolvedLaunchSource,
    authoritative: ResolvedAuthoritativeSWEConfig,
    *,
    launch_output_dir_override: str | pathlib.Path | None = None,
) -> tuple[list[str], pathlib.Path]:
    launch_output_dir = (
        pathlib.Path(launch_output_dir_override)
        if launch_output_dir_override is not None
        else pathlib.Path(cfg.launch.output_dir)
    )
    launch_output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = _launch_extra_args(cfg)
    chat_template_kwargs = _launch_model_chat_template_kwargs(cfg)
    completion_kwargs = _launch_model_completion_kwargs(cfg)
    agent_max_steps = _launch_agent_max_steps(cfg)
    effective_sampling = _effective_launch_sampling(cfg)
    effective_candidate_sampling = _effective_launch_candidate_sampling(cfg)
    _validate_launch_extra_args(
        extra_args,
        chat_template_kwargs=chat_template_kwargs,
        agent_max_steps=agent_max_steps,
    )
    cmd = [
        str(cfg.launch.python_executable),
        "-m",
        str(cfg.launch.run_module),
        "run-batch",
        "--config",
        str(authoritative.config_path),
        "--output_dir",
        str(launch_output_dir),
        "--num_workers",
        str(cfg.launch.num_workers),
        *resolved_source.launch_args,
    ]
    if cfg.launch.model.name not in (None, ""):
        cmd.extend(["--agent.model.name", str(cfg.launch.model.name)])
    if cfg.launch.model.backend not in (None, ""):
        cmd.extend(["--agent.model.backend", str(cfg.launch.model.backend)])
    if effective_sampling.get("temperature") is not None:
        cmd.extend(["--agent.model.temperature", str(effective_sampling["temperature"])])
    if effective_sampling.get("top_p") is not None:
        cmd.extend(["--agent.model.top_p", str(effective_sampling["top_p"])])
    if cfg.launch.model.tokenizer_path not in (None, ""):
        cmd.extend(["--agent.model.tokenizer_path", str(cfg.launch.model.tokenizer_path)])
    if cfg.launch.model.tensor_parallel_size not in (None, ""):
        cmd.extend(["--agent.model.tensor_parallel_size", str(cfg.launch.model.tensor_parallel_size)])
    if cfg.launch.model.gpu_memory_utilization not in (None, ""):
        cmd.extend(
            ["--agent.model.gpu_memory_utilization", str(cfg.launch.model.gpu_memory_utilization)]
        )
    if effective_sampling.get("max_output_tokens") is not None:
        cmd.extend(["--agent.model.max_output_tokens", str(effective_sampling["max_output_tokens"])])
    if effective_sampling.get("top_k") is not None:
        if str(cfg.launch.model.backend or "") != "vllm":
            raise ValueError(
                "launch.sampling.top_k is supported only when launch.model.backend='vllm'"
            )
        cmd.extend(["--agent.model.top_k", str(effective_sampling["top_k"])])
    if cfg.launch.model.trust_remote_code not in (None, ""):
        cmd.extend(
            [
                "--agent.model.trust_remote_code",
                "True" if bool(cfg.launch.model.trust_remote_code) else "False",
            ]
        )
    if completion_kwargs:
        cmd.extend(
            [
                "--agent.model.completion_kwargs",
                _serialize_completion_kwargs_for_cli(completion_kwargs),
            ]
        )
    if "num_candidates" in effective_candidate_sampling:
        cmd.extend(
            [
                "--agent.candidate_sampling.num_candidates",
                str(effective_candidate_sampling["num_candidates"]),
            ]
        )
    if cfg.launch.candidate_sampling.max_retry_candidates not in (None, ""):
        cmd.extend(
            [
                "--agent.candidate_sampling.max_retry_candidates",
                str(effective_candidate_sampling["max_retry_candidates"]),
            ]
        )
    if agent_max_steps not in (None, ""):
        cmd.extend(
            [
                "--agent.max_steps",
                str(agent_max_steps),
            ]
        )
    cmd.extend(extra_args)
    return cmd, launch_output_dir


def launch_vendored_swe_agent(
    cfg: DictConfig,
    resolved_source: ResolvedLaunchSource,
    authoritative: ResolvedAuthoritativeSWEConfig,
    *,
    launch_output_dir_override: str | pathlib.Path | None = None,
    build_launch_command: Callable[..., tuple[list[str], pathlib.Path]] = _build_launch_command,
) -> pathlib.Path:
    cmd, launch_output_dir = build_launch_command(
        cfg,
        resolved_source,
        authoritative,
        launch_output_dir_override=launch_output_dir_override,
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_entries = [str(REPO_ROOT / "src"), str(REPO_ROOT / "vendor" / "SWE-agent")]
    if existing_pythonpath:
        path_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(path_entries)
    log.info("launching vendored SWE-agent:\n%s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    return launch_output_dir


__all__ = ["_build_launch_command", "launch_vendored_swe_agent"]
