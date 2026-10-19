from __future__ import annotations

import json
from typing import Any

from omegaconf import DictConfig, OmegaConf

from agent_uq.swe_workflow.common import validate_swe_raw_turn_mode

WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
)


def _launch_extra_args(cfg: DictConfig) -> list[str]:
    return [str(arg) for arg in cfg.launch.extra_args]


def _launch_model_chat_template_kwargs(cfg: DictConfig) -> dict[str, Any]:
    raw_value = cfg.launch.model.get("chat_template_kwargs")
    if raw_value in (None, ""):
        return {}
    resolved = OmegaConf.to_container(raw_value, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("launch.model.chat_template_kwargs must resolve to a mapping")
    return resolved


def _launch_model_completion_kwargs(cfg: DictConfig) -> dict[str, Any]:
    chat_template_kwargs = _launch_model_chat_template_kwargs(cfg)
    if not chat_template_kwargs:
        return {}
    return {"chat_template_kwargs": chat_template_kwargs}


def _launch_sampling_config(cfg: DictConfig) -> dict[str, Any]:
    raw_value = cfg.launch.get("sampling")
    if raw_value in (None, ""):
        return {}
    resolved = OmegaConf.to_container(raw_value, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("launch.sampling must resolve to a mapping")
    return resolved


def _effective_launch_sampling(cfg: DictConfig) -> dict[str, float | int]:
    sampling_cfg = _launch_sampling_config(cfg)
    effective: dict[str, float | int] = {}
    if sampling_cfg.get("temperature") not in (None, ""):
        effective["temperature"] = float(sampling_cfg["temperature"])
    if sampling_cfg.get("top_p") not in (None, ""):
        effective["top_p"] = float(sampling_cfg["top_p"])
    if sampling_cfg.get("top_k") not in (None, ""):
        effective["top_k"] = int(sampling_cfg["top_k"])
    if sampling_cfg.get("max_output_tokens") not in (None, ""):
        effective["max_output_tokens"] = int(sampling_cfg["max_output_tokens"])
    return effective


def _effective_launch_candidate_sampling(cfg: DictConfig) -> dict[str, int]:
    effective: dict[str, int] = {}
    num_candidates = cfg.launch.candidate_sampling.get("num_candidates")
    if num_candidates not in (None, ""):
        effective["num_candidates"] = int(num_candidates)
    max_retry_candidates = cfg.launch.candidate_sampling.get("max_retry_candidates")
    if max_retry_candidates not in (None, ""):
        effective["max_retry_candidates"] = int(max_retry_candidates)
    elif "num_candidates" in effective:
        effective["max_retry_candidates"] = 3 * effective["num_candidates"]
    if (
        "num_candidates" in effective
        and "max_retry_candidates" in effective
        and effective["max_retry_candidates"] < effective["num_candidates"]
    ):
        raise ValueError(
            "launch.candidate_sampling.max_retry_candidates must be >= "
            "launch.candidate_sampling.num_candidates"
        )
    return effective


def _launch_agent_max_steps(cfg: DictConfig) -> int | None:
    raw_value = cfg.launch.agent.get("max_steps")
    if raw_value in (None, ""):
        return None
    return int(raw_value)


def _serialize_completion_kwargs_for_cli(completion_kwargs: dict[str, Any]) -> str:
    return json.dumps(completion_kwargs, separators=(",", ":"), sort_keys=True)


def _validate_launch_extra_args(
    extra_args: list[str],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    agent_max_steps: int | None = None,
) -> None:
    for duplicate_flag, field_name in (
        ("--agent.candidate_sampling.num_candidates", "launch.candidate_sampling.num_candidates"),
        (
            "--agent.candidate_sampling.max_retry_candidates",
            "launch.candidate_sampling.max_retry_candidates",
        ),
    ):
        for arg in extra_args:
            if arg == duplicate_flag or arg.startswith(f"{duplicate_flag}="):
                raise ValueError(
                    "launch.extra_args must not duplicate structured "
                    f"{field_name}"
                )
    if chat_template_kwargs:
        duplicate_completion_kwargs_flag = "--agent.model.completion_kwargs"
        for arg in extra_args:
            if (
                arg == duplicate_completion_kwargs_flag
                or arg.startswith(f"{duplicate_completion_kwargs_flag}=")
                or arg.startswith(f"{duplicate_completion_kwargs_flag}.")
            ):
                raise ValueError(
                    "launch.extra_args must not duplicate structured "
                    "launch.model.chat_template_kwargs"
                )
    for duplicate_flag, field_name in (
        ("--agent.model.temperature", "launch.sampling.temperature"),
        ("--agent.model.top_p", "launch.sampling.top_p"),
        ("--agent.model.top_k", "launch.sampling.top_k"),
        ("--agent.model.max_output_tokens", "launch.sampling.max_output_tokens"),
    ):
        for arg in extra_args:
            if arg == duplicate_flag or arg.startswith(f"{duplicate_flag}="):
                raise ValueError(
                    "launch.extra_args must not duplicate structured "
                    f"{field_name}"
                )
    if agent_max_steps is not None:
        duplicate_max_steps_flag = "--agent.max_steps"
        for arg in extra_args:
            if arg == duplicate_max_steps_flag or arg.startswith(f"{duplicate_max_steps_flag}="):
                raise ValueError(
                    "launch.extra_args must not duplicate structured "
                    "launch.agent.max_steps"
                )


def validate_swe_workflow_config(cfg: DictConfig) -> None:
    mode = str(cfg.mode)
    source_kind = str(cfg.source.kind)
    bootstrap_strategy = str(cfg.bootstrap.strategy)
    evaluation_mode = str(cfg.evaluation.mode)
    import_target_type = str(cfg["import"].target_type)
    validate_swe_raw_turn_mode(cfg["import"].get("raw_turn_mode"))
    for legacy_field in WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS:
        if legacy_field in cfg.launch.model:
            raise ValueError(
                "workflow-owned decode-time settings must be configured under "
                f"launch.sampling; found legacy launch.model.{legacy_field}"
            )
    effective_sampling = _effective_launch_sampling(cfg)
    if effective_sampling.get("top_k") is not None and str(cfg.launch.model.backend or "") != "vllm":
        raise ValueError(
            "launch.sampling.top_k is supported only when launch.model.backend='vllm'"
        )
    _effective_launch_candidate_sampling(cfg)
    _validate_launch_extra_args(
        _launch_extra_args(cfg),
        chat_template_kwargs=_launch_model_chat_template_kwargs(cfg),
        agent_max_steps=_launch_agent_max_steps(cfg),
    )
    if import_target_type != "binary":
        raise ValueError(
            "SWE raw import requires import.target_type='binary'; "
            f"got {import_target_type!r}"
        )

    if mode not in {"import_only", "launch_and_import"}:
        raise ValueError(f"unsupported swe_workflow mode: {mode!r}")
    if mode == "import_only":
        if source_kind != "artifacts":
            raise ValueError("mode='import_only' requires source.kind='artifacts'")
        if bootstrap_strategy != "none":
            raise ValueError("mode='import_only' requires bootstrap.strategy='none'")
        if evaluation_mode != "existing_report":
            raise ValueError("mode='import_only' requires evaluation.mode='existing_report'")
        return

    if source_kind not in {"swe_bench", "file"}:
        raise ValueError(
            "mode='launch_and_import' requires source.kind to be 'swe_bench' or 'file'"
        )
    if source_kind == "swe_bench" and str(cfg.source.subset) == "multimodal":
        raise ValueError("multimodal SWE-Bench launch subsets are not supported by this workflow")
    if evaluation_mode not in {"launch_results", "sb_cli_submit", "swebench_harness"}:
        raise ValueError(
            "mode='launch_and_import' requires evaluation.mode to be "
            "'launch_results', 'sb_cli_submit', or 'swebench_harness'"
        )
    if evaluation_mode == "launch_results" and (
        source_kind != "swe_bench" or bootstrap_strategy != "none"
    ):
        raise ValueError(
            "evaluation.mode='launch_results' is only supported when "
            "source.kind='swe_bench' and bootstrap.strategy='none'"
        )


__all__ = [
    "WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS",
    "_launch_extra_args",
    "_launch_model_chat_template_kwargs",
    "_launch_model_completion_kwargs",
    "_launch_sampling_config",
    "_effective_launch_sampling",
    "_effective_launch_candidate_sampling",
    "_launch_agent_max_steps",
    "_serialize_completion_kwargs_for_cli",
    "_validate_launch_extra_args",
    "validate_swe_workflow_config",
]
