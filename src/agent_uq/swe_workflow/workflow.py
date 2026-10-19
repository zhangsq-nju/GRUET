from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
import pathlib
from typing import Callable

from omegaconf import DictConfig, OmegaConf

from agent_uq.records import list_record_paths, read_raw, write_raw
from agent_uq.swe_workflow.cache import (
    archive_semantic_import_downstream_artifacts,
    completed_semantic_launch_ids,
    ensure_semantic_import_targets_compatible,
    ensure_semantic_launch_predictions,
    ensure_semantic_launch_targets_compatible,
    resolve_semantic_import_selectors,
    write_semantic_launch_manifest,
)
from agent_uq.swe_workflow.candidate_import import (
    export_candidate_traj_to_raw,
    resolve_launch_source,
)
from agent_uq.swe_workflow.common import repo_relative_path, require_path, utc_now_iso, validate_swe_raw_turn_mode
from agent_uq.swe_workflow.config_validation import validate_swe_workflow_config
from agent_uq.swe_workflow.evaluator import PreparedEvaluatorInputs
from agent_uq.swe_workflow.harness import prepare_evaluator_inputs
from agent_uq.swe_workflow.import_plan import (
    ResolvedAuthoritativeSWEConfig,
    assert_authoritative_swe_assets,
    build_import_plan,
    build_swe_manifest_runtime_block,
    inspect_candidate_traj_source_metadata,
    resolve_authoritative_swe_config,
    resolve_imported_trajectory_format,
    resolve_swe_import_selectors,
    should_reuse_cached_swe_leaf,
    validate_import_only_replay_contract,
)
from agent_uq.swe_workflow.launch import ResolvedLaunchSource
from agent_uq.swe_workflow.launch_command import launch_vendored_swe_agent
from agent_uq.swe_workflow.semantic import (
    SemanticWorkflowPaths,
    _resolve_semantic_workflow_paths,
    _semantic_cache_enabled,
    _semantic_existing_raw_ids_by_mode,
    _write_active_import_for_semantic_leaf,
)
from agent_uq.cache.leaf_manifest import aggregate_diagnostics_from_raws, write_leaf_manifest

log = logging.getLogger("agent_uq.swe_workflow.workflow")


@dataclass(frozen=True)
class SweWorkflowDeps:
    validate_config: Callable[[DictConfig], None] = validate_swe_workflow_config
    resolve_launch_source: Callable[..., ResolvedLaunchSource] = resolve_launch_source
    launch_vendored_swe_agent: Callable[..., pathlib.Path] = launch_vendored_swe_agent
    export_candidate_traj_to_raw: Callable[..., object] = export_candidate_traj_to_raw


def _prepare_import_context(
    cfg: DictConfig,
    authoritative: ResolvedAuthoritativeSWEConfig,
    *,
    semantic_paths: SemanticWorkflowPaths | None = None,
    deps: SweWorkflowDeps | None = None,
) -> tuple[pathlib.Path, PreparedEvaluatorInputs, list[str], dict[str, pathlib.Path]]:
    deps = deps or SweWorkflowDeps()
    mode = str(cfg.mode)
    output_dir = semantic_paths.import_dir if semantic_paths is not None else pathlib.Path(cfg.output_dir)
    runtime_byproducts: dict[str, pathlib.Path] = {}
    if mode == "launch_and_import":
        launch_runtime_root = (
            semantic_paths.launch_dir if semantic_paths is not None else pathlib.Path(cfg.output_dir)
        )
        preflight_target_record_ids: list[str] | None = None
        if semantic_paths is not None:
            preflight_source = deps.resolve_launch_source(
                source_cfg=cfg.source,
                bootstrap_cfg=cfg.bootstrap,
                evaluation_cfg=cfg.evaluation,
                workflow_output_dir=launch_runtime_root,
                launch_cfg=cfg.launch,
                materialize_runtime_byproducts=False,
            )
            preflight_target_record_ids = list(preflight_source.target_record_ids)
            ensure_semantic_launch_targets_compatible(
                semantic_paths.launch_dir,
                launch_key=semantic_paths.launch_key,
                requested_target_record_ids=preflight_target_record_ids,
            )
            ensure_semantic_import_targets_compatible(
                semantic_paths.import_dir,
                launch_key=semantic_paths.launch_key,
                requested_target_record_ids=preflight_target_record_ids,
            )
        resolved_source = deps.resolve_launch_source(
            source_cfg=cfg.source,
            bootstrap_cfg=cfg.bootstrap,
            evaluation_cfg=cfg.evaluation,
            workflow_output_dir=launch_runtime_root,
            launch_cfg=cfg.launch,
        )
        target_record_ids = list(resolved_source.target_record_ids)
        if preflight_target_record_ids is not None and target_record_ids != preflight_target_record_ids:
            raise ValueError(
                "semantic launch target preflight diverged from materialized launch source: "
                f"preview={preflight_target_record_ids!r}, resolved={target_record_ids!r}"
            )
        runtime_byproducts.update(resolved_source.runtime_byproducts)
        launch_output_dir = deps.launch_vendored_swe_agent(
            cfg,
            resolved_source,
            authoritative,
            launch_output_dir_override=(semantic_paths.launch_dir if semantic_paths is not None else None),
        )
        candidate_traj_root = launch_output_dir
        if semantic_paths is not None:
            ensure_semantic_launch_predictions(
                launch_output_dir,
                requested_target_record_ids=target_record_ids,
            )
            covered_target_record_ids = completed_semantic_launch_ids(
                launch_output_dir,
                requested_target_record_ids=target_record_ids,
            )
            launch_manifest_path = write_semantic_launch_manifest(
                launch_output_dir,
                launch_key=semantic_paths.launch_key,
                key_payload=semantic_paths.key_payload,
                requested_target_record_ids=target_record_ids,
                covered_target_record_ids=covered_target_record_ids,
                updated_at=utc_now_iso(),
            )
            runtime_byproducts["launch_manifest_path"] = launch_manifest_path
        evaluator_inputs = prepare_evaluator_inputs(
            evaluation_cfg=cfg.evaluation,
            source_cfg=cfg.source,
            workflow_output_dir=output_dir,
            target_record_ids=target_record_ids,
            launch_output_dir=launch_output_dir,
            normalized_report_path_override=(
                output_dir / "runtime" / "results.json" if semantic_paths is not None else None
            ),
            sb_cli_output_dir_override=(
                semantic_paths.sb_cli_reports_root
                if semantic_paths is not None and str(cfg.evaluation.mode) == "sb_cli_submit"
                else None
            ),
        )
    else:
        candidate_traj_root = require_path(
            cfg.source.candidate_traj_root,
            field_name="source.candidate_traj_root",
        )
        target_record_ids = [str(record_id) for record_id in (cfg.source.target_record_ids or [])]
        evaluator_inputs = prepare_evaluator_inputs(
            evaluation_cfg=cfg.evaluation,
            source_cfg=cfg.source,
            workflow_output_dir=output_dir,
            target_record_ids=(target_record_ids or None),
        )
    runtime_byproducts.update(evaluator_inputs.runtime_byproducts)
    return candidate_traj_root, evaluator_inputs, target_record_ids, runtime_byproducts


def run_swe_workflow(cfg: DictConfig, *, deps: SweWorkflowDeps | None = None) -> pathlib.Path:
    deps = deps or SweWorkflowDeps()
    deps.validate_config(cfg)
    authoritative = resolve_authoritative_swe_config(cfg.authoritative.config_path)
    assert_authoritative_swe_assets(authoritative)
    log.info("agent_uq swe_workflow config:\n%s", OmegaConf.to_yaml(cfg))

    semantic_paths = _resolve_semantic_workflow_paths(cfg, authoritative) if _semantic_cache_enabled(cfg) else None
    output_dir = semantic_paths.import_dir if semantic_paths is not None else pathlib.Path(cfg.output_dir)

    candidate_traj_root, evaluator_inputs, target_record_ids, runtime_byproducts = _prepare_import_context(
        cfg,
        authoritative,
        semantic_paths=semantic_paths,
        deps=deps,
    )
    plan = build_import_plan(
        candidate_traj_root=candidate_traj_root,
        evaluator_report_path=evaluator_inputs.evaluator_report_path,
        target_record_ids=(target_record_ids or None),
        candidate_traj_pattern=str(cfg.source.candidate_traj_pattern),
    )
    source_metadata = {
        record_id: inspect_candidate_traj_source_metadata(path)
        for record_id, path in plan.candidate_traj_by_record.items()
    }
    if str(cfg.mode) == "import_only":
        validate_import_only_replay_contract(
            target_record_ids=plan.target_record_ids,
            candidate_traj_by_record=plan.candidate_traj_by_record,
            metadata_by_record=source_metadata,
            authoritative=authoritative,
        )
    imported_trajectory_format = resolve_imported_trajectory_format(
        target_record_ids=plan.target_record_ids,
        metadata_by_record=source_metadata,
    )
    import_cfg = cfg["import"]
    raw_turn_mode = validate_swe_raw_turn_mode(import_cfg.get("raw_turn_mode"))
    if semantic_paths is not None:
        selectors = resolve_semantic_import_selectors(
            launch_key=semantic_paths.launch_key,
            raw_turn_mode=raw_turn_mode,
        )
        reuse_if_cached = True
    else:
        selectors = resolve_swe_import_selectors(
            authoritative=authoritative,
            target_record_ids=plan.target_record_ids,
            candidate_traj_by_record=plan.candidate_traj_by_record,
            normalized_outcomes=plan.normalized_outcomes,
            imported_trajectory_format=imported_trajectory_format,
            dataset_key=str(import_cfg.dataset_key),
            target_type=str(import_cfg.target_type),
            raw_turn_mode=raw_turn_mode,
        )
        reuse_if_cached = bool(import_cfg.get("reuse_if_cached", False))

    raw_dir = output_dir / "raw"
    matching_raw_ids: set[str] = set()
    mismatched_raw_ids: set[str] = set()
    if semantic_paths is not None:
        matching_raw_ids, mismatched_raw_ids = _semantic_existing_raw_ids_by_mode(
            raw_dir,
            raw_turn_mode=raw_turn_mode,
        )
    requested_mismatched_raw_ids = set(plan.target_record_ids) & mismatched_raw_ids
    if reuse_if_cached and should_reuse_cached_swe_leaf(
        output_dir=output_dir,
        sampling_selectors=selectors,
        target_record_ids=plan.target_record_ids,
    ) and not requested_mismatched_raw_ids:
        log.info("reusing cached SWE imported leaf at %s", output_dir)
        if semantic_paths is not None:
            live_raw_paths = list_record_paths(raw_dir, reject_duplicates=True)
            _write_active_import_for_semantic_leaf(
                cfg,
                semantic_paths,
                raw_turn_mode=raw_turn_mode,
                target_record_ids=plan.target_record_ids,
                raw_count=len(live_raw_paths),
                completed_at=utc_now_iso(),
            )
        return output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if requested_mismatched_raw_ids:
        archived = archive_semantic_import_downstream_artifacts(output_dir)
        if archived:
            log.info(
                "archived semantic import downstream artifacts after raw_turn_mode drift: %s",
                ", ".join(str(path) for path in archived),
            )

    imported_any = False
    for record_id in plan.target_record_ids:
        if semantic_paths is not None and record_id in matching_raw_ids:
            continue
        raw = deps.export_candidate_traj_to_raw(
            authoritative=authoritative,
            record_id=record_id,
            candidate_traj_path=plan.candidate_traj_by_record[record_id],
            evaluator_outcome=plan.normalized_outcomes[record_id],
            evaluator_report_path=evaluator_inputs.evaluator_report_path,
            dataset_key=str(import_cfg.dataset_key),
            target_type=str(import_cfg.target_type),
            env_name=str(import_cfg.env_name),
            model_name=(
                None if import_cfg.get("model_name") in (None, "") else str(import_cfg.model_name)
            ),
            tokenizer_name=(
                None if import_cfg.get("tokenizer_name") in (None, "") else str(import_cfg.tokenizer_name)
            ),
            prediction_source_path=evaluator_inputs.prediction_source_path,
            seed=int(import_cfg.raw_seed),
            raw_turn_mode=raw_turn_mode,
        )
        write_raw(raw, raw_dir, gzip_records=bool(import_cfg.gzip_records))
        imported_any = True
        log.info(
            "imported %s -> solved=%s reward=%.1f turns=%d done=%s",
            record_id,
            raw.solved,
            raw.final_reward,
            len(raw.turns),
            raw.done_reason,
        )

    live_raw_paths = list_record_paths(raw_dir, reject_duplicates=True)
    live_raws = [read_raw(path) for path in live_raw_paths]
    diagnostics = aggregate_diagnostics_from_raws(live_raws)
    runtime = build_swe_manifest_runtime_block(
        authoritative=authoritative,
        imported_trajectory_format=imported_trajectory_format,
        candidate_traj_by_record=plan.candidate_traj_by_record,
        evaluator_report_path=evaluator_inputs.evaluator_report_path,
        prediction_source_path=evaluator_inputs.prediction_source_path,
        raw_turn_mode=raw_turn_mode,
    )
    runtime["workflow_run"] = {
        "mode": str(cfg.mode),
        "source_kind": str(cfg.source.kind),
        "bootstrap_strategy": str(cfg.bootstrap.strategy),
        "evaluation_mode": str(cfg.evaluation.mode),
        "semantic_launch_key": (semantic_paths.launch_key if semantic_paths is not None else None),
        "runtime_byproducts": {
            key: repo_relative_path(path)
            for key, path in sorted(runtime_byproducts.items())
        },
    }
    model_name = live_raws[0].model if live_raws else str(import_cfg.get("model_name") or "unknown_model")
    completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_leaf_manifest(
        output_dir,
        leaf={"model": model_name, "env": str(import_cfg.dataset_key)},
        sampling=selectors,
        runtime=runtime,
        target_record_ids=plan.target_record_ids,
        raw_count=len(live_raws),
        diagnostics=diagnostics,
        completed_at=completed_at,
        finalization={},
    )
    if semantic_paths is not None:
        _write_active_import_for_semantic_leaf(
            cfg,
            semantic_paths,
            raw_turn_mode=raw_turn_mode,
            target_record_ids=plan.target_record_ids,
            raw_count=len(live_raws),
            completed_at=completed_at,
        )
    if imported_any:
        log.info("wrote %d imported SWE raw records under %s", len(live_raws), raw_dir)
    else:
        log.info("rewrote semantic SWE manifest from %d live raw records under %s", len(live_raws), raw_dir)
    return output_dir


__all__ = ["SweWorkflowDeps", "_prepare_import_context", "run_swe_workflow"]
