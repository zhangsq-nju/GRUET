from __future__ import annotations

import pathlib

import hydra
from omegaconf import DictConfig

from agent_uq.swe_workflow.candidate_import import (
    export_candidate_traj_to_raw,
    resolve_launch_source,
)
from agent_uq.swe_workflow.config_validation import (
    WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS as _WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS,
    _effective_launch_candidate_sampling,
    _effective_launch_sampling,
    _launch_agent_max_steps,
    _launch_extra_args,
    _launch_model_chat_template_kwargs,
    _launch_model_completion_kwargs,
    _launch_sampling_config,
    _serialize_completion_kwargs_for_cli,
    _validate_launch_extra_args,
    validate_swe_workflow_config,
)
from agent_uq.swe_workflow.launch_command import (
    _build_launch_command,
    launch_vendored_swe_agent as _owner_launch_vendored_swe_agent,
)
from agent_uq.swe_workflow.semantic import (
    SemanticWorkflowPaths,
    _resolve_semantic_workflow_paths,
    _selected_swe_run_name,
    _semantic_cache_enabled,
    _semantic_existing_raw_ids_by_mode,
    _semantic_launch_key,
    _semantic_launch_key_payload,
    _slugify_component,
    _write_active_import_for_semantic_leaf,
)
from agent_uq.swe_workflow.workflow import (
    SweWorkflowDeps,
    _prepare_import_context,
    run_swe_workflow as _owner_run_swe_workflow,
)


def launch_vendored_swe_agent(
    cfg: DictConfig,
    resolved_source,
    authoritative,
    *,
    launch_output_dir_override: str | pathlib.Path | None = None,
) -> pathlib.Path:
    return _owner_launch_vendored_swe_agent(
        cfg,
        resolved_source,
        authoritative,
        launch_output_dir_override=launch_output_dir_override,
        build_launch_command=_build_launch_command,
    )


def _workflow_deps_from_facade() -> SweWorkflowDeps:
    return SweWorkflowDeps(
        validate_config=validate_swe_workflow_config,
        resolve_launch_source=resolve_launch_source,
        launch_vendored_swe_agent=launch_vendored_swe_agent,
        export_candidate_traj_to_raw=export_candidate_traj_to_raw,
    )


def run_swe_workflow(cfg: DictConfig) -> pathlib.Path:
    return _owner_run_swe_workflow(cfg, deps=_workflow_deps_from_facade())


@hydra.main(version_base=None, config_path="../conf", config_name="swe_workflow")
def main(cfg: DictConfig) -> None:
    run_swe_workflow(cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "SemanticWorkflowPaths",
    "_WORKFLOW_OWNED_LAUNCH_SAMPLING_FIELDS",
    "_launch_extra_args",
    "_launch_model_chat_template_kwargs",
    "_launch_model_completion_kwargs",
    "_launch_sampling_config",
    "_effective_launch_sampling",
    "_effective_launch_candidate_sampling",
    "_launch_agent_max_steps",
    "_serialize_completion_kwargs_for_cli",
    "_slugify_component",
    "_semantic_launch_key_payload",
    "_semantic_launch_key",
    "_semantic_cache_enabled",
    "_selected_swe_run_name",
    "_write_active_import_for_semantic_leaf",
    "_semantic_existing_raw_ids_by_mode",
    "_resolve_semantic_workflow_paths",
    "_validate_launch_extra_args",
    "validate_swe_workflow_config",
    "_build_launch_command",
    "launch_vendored_swe_agent",
    "_prepare_import_context",
    "run_swe_workflow",
    "main",
    "resolve_launch_source",
    "export_candidate_traj_to_raw",
]
