from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import pathlib
import re
from typing import Any

from omegaconf import DictConfig

from agent_uq.records import list_record_paths, read_raw
from agent_uq.swe_workflow.config_validation import (
    _effective_launch_candidate_sampling,
    _effective_launch_sampling,
    _launch_agent_max_steps,
    _launch_extra_args,
    _launch_model_chat_template_kwargs,
)
from agent_uq.swe_workflow.import_plan import ResolvedAuthoritativeSWEConfig
from agent_uq.swe_workflow.import_resolver import write_active_import_metadata
from agent_uq.swe_workflow.common import utc_now_iso

log = logging.getLogger("agent_uq.swe_workflow.semantic")


@dataclass(frozen=True)
class SemanticWorkflowPaths:
    scenario_output_dir: pathlib.Path
    launch_root: pathlib.Path
    import_root: pathlib.Path
    launch_key: str
    key_payload: dict[str, Any]
    launch_dir: pathlib.Path
    import_dir: pathlib.Path
    sb_cli_reports_root: pathlib.Path


def _slugify_component(value: str, *, fallback: str = "x", max_len: int = 32) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    if not slug:
        slug = fallback
    return slug[:max_len]


def _semantic_launch_key_payload(
    cfg: DictConfig,
    authoritative: ResolvedAuthoritativeSWEConfig,
) -> dict[str, Any]:
    source_payload: dict[str, Any] = {
        "kind": str(cfg.source.kind),
    }
    if str(cfg.source.kind) == "swe_bench":
        source_payload["subset"] = str(cfg.source.subset)
        source_payload["split"] = str(cfg.source.split)
    return {
        "key_version": 1,
        "source": source_payload,
        "authoritative": {
            "config_path": authoritative.config_path_display,
        },
        "launch": {
            "model_name": str(cfg.launch.model.name or ""),
            "backend": str(cfg.launch.model.backend or ""),
            "tokenizer_path": str(cfg.launch.model.tokenizer_path or ""),
            "sampling": _effective_launch_sampling(cfg),
            "chat_template_kwargs": _launch_model_chat_template_kwargs(cfg),
            "candidate_sampling": _effective_launch_candidate_sampling(cfg),
            "agent_max_steps": _launch_agent_max_steps(cfg),
            "extra_args": _launch_extra_args(cfg),
        },
    }


def _semantic_launch_key(
    cfg: DictConfig,
    authoritative: ResolvedAuthoritativeSWEConfig,
) -> tuple[str, dict[str, Any]]:
    payload = _semantic_launch_key_payload(cfg, authoritative)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    source = payload["source"]
    source_prefix = "__".join(
        _slugify_component(str(source.get(name, "")), fallback=name)
        for name in ("kind", "subset", "split")
        if source.get(name, "")
    )
    config_prefix = _slugify_component(pathlib.Path(authoritative.config_path_display).stem, fallback="cfg")
    model_prefix = _slugify_component(str(payload["launch"]["model_name"] or "model"), fallback="model")
    prefix_parts = [part for part in (source_prefix, config_prefix, model_prefix) if part]
    return "__".join(prefix_parts + [digest]), payload


def _semantic_cache_enabled(cfg: DictConfig) -> bool:
    return str(cfg.mode) == "launch_and_import" and str(cfg.source.kind) == "swe_bench"


def _selected_swe_run_name(cfg: DictConfig) -> str:
    swe_run = cfg.get("swe_run")
    if isinstance(swe_run, DictConfig):
        name = swe_run.get("name")
        if name not in (None, ""):
            return str(name)
    return pathlib.Path(str(cfg.output_dir)).name


def _write_active_import_for_semantic_leaf(
    cfg: DictConfig,
    semantic_paths: SemanticWorkflowPaths,
    *,
    raw_turn_mode: str,
    target_record_ids: list[str],
    raw_count: int,
    completed_at: str,
) -> pathlib.Path:
    path = write_active_import_metadata(
        import_root=semantic_paths.import_root,
        import_leaf=semantic_paths.import_dir,
        swe_run_name=_selected_swe_run_name(cfg),
        semantic_launch_key=semantic_paths.launch_key,
        raw_turn_mode=raw_turn_mode,
        target_record_ids=target_record_ids,
        raw_count=raw_count,
        completed_at=completed_at,
    )
    log.info("wrote active SWE import metadata at %s", path)
    return path


def _semantic_existing_raw_ids_by_mode(
    raw_dir: pathlib.Path,
    *,
    raw_turn_mode: str,
) -> tuple[set[str], set[str]]:
    matching: set[str] = set()
    mismatched: set[str] = set()
    if not raw_dir.is_dir():
        return matching, mismatched
    for path in list_record_paths(raw_dir, reject_duplicates=True):
        raw = read_raw(path)
        if raw.env_provenance.get("raw_turn_mode") == raw_turn_mode:
            matching.add(str(raw.record_id))
        else:
            mismatched.add(str(raw.record_id))
    return matching, mismatched


def _resolve_semantic_workflow_paths(
    cfg: DictConfig,
    authoritative: ResolvedAuthoritativeSWEConfig,
) -> SemanticWorkflowPaths:
    scenario_output_dir = pathlib.Path(cfg.output_dir)
    launch_root = pathlib.Path(cfg.launch.output_dir)
    import_root = scenario_output_dir / "import"
    launch_key, key_payload = _semantic_launch_key(cfg, authoritative)
    return SemanticWorkflowPaths(
        scenario_output_dir=scenario_output_dir,
        launch_root=launch_root,
        import_root=import_root,
        launch_key=launch_key,
        key_payload=key_payload,
        launch_dir=launch_root / launch_key,
        import_dir=import_root / launch_key,
        sb_cli_reports_root=scenario_output_dir / "runtime" / "sb-cli-reports" / launch_key,
    )


__all__ = [
    "SemanticWorkflowPaths",
    "_slugify_component",
    "_semantic_launch_key_payload",
    "_semantic_launch_key",
    "_semantic_cache_enabled",
    "_selected_swe_run_name",
    "_write_active_import_for_semantic_leaf",
    "_semantic_existing_raw_ids_by_mode",
    "_resolve_semantic_workflow_paths",
    "utc_now_iso",
]
