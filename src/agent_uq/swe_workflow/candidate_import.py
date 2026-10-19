from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Mapping, Sequence

from agent_uq.actions import build_action
from agent_uq.records import (
    CandidateRecord,
    RawTrajectory,
    SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT,
    TopKEntryRec,
    TrajectoryConfig,
    Turn,
)
from agent_uq.swe_workflow.bootstrap import materialize_bootstrap_rewritten_instances
from agent_uq.swe_workflow.common import (
    SWE_RAW_TURN_MODE_SELECTED_ONLY,
    SWE_SELECTION_POLICY,
    ensure_runtime_dir,
    repo_relative_path,
    validate_swe_raw_turn_mode,
)
from agent_uq.swe_workflow.gemma import (
    _is_gemma_series_name,
    _project_gemma_current_reasoning_candidate,
)
from agent_uq.swe_workflow.import_plan import (
    ResolvedAuthoritativeSWEConfig,
    _load_replay_config_mapping,
    load_candidate_traj,
)
from agent_uq.swe_workflow.launch import (
    ResolvedLaunchSource,
    build_file_source_launch_args,
    build_native_source_launch_args,
    resolve_selected_simple_batch_instances,
)


def resolve_launch_source(
    *,
    source_cfg: Any,
    bootstrap_cfg: Any,
    evaluation_cfg: Any,
    workflow_output_dir: str | pathlib.Path,
    launch_cfg: Any,
    materialize_runtime_byproducts: bool = True,
) -> ResolvedLaunchSource:
    source_kind = str(source_cfg.kind)
    if source_kind not in {"swe_bench", "file"}:
        raise ValueError(
            "launch mode requires source.kind to be 'swe_bench' or 'file'; "
            f"got {source_kind!r}"
        )

    selected_instances = resolve_selected_simple_batch_instances(source_cfg)
    target_record_ids = [str(instance["instance_id"]) for instance in selected_instances]
    launch_args = build_native_source_launch_args(
        source_cfg,
        evaluation_cfg=evaluation_cfg,
        launch_cfg=launch_cfg,
    )
    if str(bootstrap_cfg.strategy) == "none":
        return ResolvedLaunchSource(
            target_record_ids=target_record_ids,
            launch_args=launch_args,
            runtime_byproducts={},
        )
    if str(bootstrap_cfg.strategy) != "local_swe_rex_rewrite":
        raise ValueError(f"unsupported SWE bootstrap strategy: {bootstrap_cfg.strategy!r}")
    if not materialize_runtime_byproducts:
        return ResolvedLaunchSource(
            target_record_ids=target_record_ids,
            launch_args=launch_args,
            runtime_byproducts={},
        )

    runtime_dir = ensure_runtime_dir(workflow_output_dir)
    selected_instances_path = runtime_dir / "selected_instances.json"
    selected_instances_path.write_text(
        json.dumps(selected_instances, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    patched_instances_path, image_map_path = materialize_bootstrap_rewritten_instances(
        selected_instances=selected_instances,
        bootstrap_cfg=bootstrap_cfg,
        launch_python_executable=str(launch_cfg.python_executable),
        runtime_dir=runtime_dir,
    )
    patched_launch_args = build_file_source_launch_args(
        instances_path=patched_instances_path,
        launch_cfg=launch_cfg,
    )
    return ResolvedLaunchSource(
        target_record_ids=target_record_ids,
        launch_args=patched_launch_args,
        runtime_byproducts={
            "selected_instances_path": selected_instances_path,
            "patched_instances_path": patched_instances_path,
            "image_map_path": image_map_path,
        },
    )


def export_candidate_traj_to_raw(
    *,
    authoritative: ResolvedAuthoritativeSWEConfig,
    record_id: str,
    candidate_traj_path: str | pathlib.Path,
    evaluator_outcome: Mapping[str, bool],
    evaluator_report_path: str | pathlib.Path,
    dataset_key: str,
    target_type: str,
    env_name: str,
    model_name: str | None = None,
    tokenizer_name: str | None = None,
    prediction_source_path: str | pathlib.Path | None = None,
    seed: int = 0,
    raw_turn_mode: str = SWE_RAW_TURN_MODE_SELECTED_ONLY,
) -> RawTrajectory:
    raw_turn_mode = validate_swe_raw_turn_mode(raw_turn_mode)
    data = load_candidate_traj(candidate_traj_path)
    imported_trajectory_format = str(data["trajectory_format"])
    exit_status = str(data.get("info", {}).get("exit_status") or "imported_candidate_traj")
    source_turns = _import_source_trajectory_turns(data, raw_turn_mode=raw_turn_mode)
    inferred_model_name = _infer_model_name(data)
    effective_model_name = str(model_name or inferred_model_name or "unknown_model")
    effective_tokenizer_name = str(tokenizer_name or effective_model_name)
    gemma_reasoning_tokenizer = (
        _get_import_tokenizer(effective_tokenizer_name)
        if _is_gemma_series_name(effective_model_name, effective_tokenizer_name)
        else None
    )
    turns = [
        _export_turn(
            turn_idx,
            turn,
            parser_mode=authoritative.parser_mode,
            trajectory_format=imported_trajectory_format,
            gemma_reasoning_tokenizer=gemma_reasoning_tokenizer,
        )
        for turn_idx, turn in enumerate(source_turns)
    ]
    submission = str(data.get("info", {}).get("submission") or "")
    return RawTrajectory(
        record_id=record_id,
        dataset_key=dataset_key,
        target_type=target_type,
        env=env_name,
        model=effective_model_name,
        config=_infer_trajectory_config(source_turns, seed=seed),
        tokenizer_name=effective_tokenizer_name,
        turns=turns,
        final_reward=1.0 if bool(evaluator_outcome["resolved"]) else 0.0,
        solved=bool(evaluator_outcome["resolved"]),
        done_reason=exit_status,
        initial_state=None,
        env_backend="swe_candidate_traj_import",
        env_provenance={
            "workflow": "vendored_swe_agent_candidate_traj",
            "authoritative_config_path": authoritative.config_path_display,
            "authoritative_demonstrations": list(authoritative.demonstration_display_paths),
            "parser_mode": authoritative.parser_mode,
            "trajectory_format": imported_trajectory_format,
            "raw_turn_mode": raw_turn_mode,
            "candidate_traj_path": repo_relative_path(candidate_traj_path),
            "evaluator_report_path": repo_relative_path(evaluator_report_path),
            "prediction_source_path": (
                None if prediction_source_path is None else repo_relative_path(prediction_source_path)
            ),
            "submission": {
                "present": bool(submission),
                "sha256": None if not submission else hashlib.sha256(submission.encode("utf-8")).hexdigest(),
                "source": "candidate_traj.info.submission",
            },
        },
    )


def _infer_model_name(candidate_traj: Mapping[str, Any]) -> str | None:
    payload = _load_replay_config_mapping(candidate_traj.get("replay_config"))
    if payload is None:
        return None
    model_payload = payload.get("agent", {}).get("model", {})
    if isinstance(model_payload, Mapping):
        name = model_payload.get("name")
        if name:
            return str(name)
    return None


_IMPORT_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_import_tokenizer(name: str) -> Any:
    if name not in _IMPORT_TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        from agent_uq.uq.hf_fallback import load_with_hf_home_fallback

        _IMPORT_TOKENIZER_CACHE[name] = load_with_hf_home_fallback(
            AutoTokenizer.from_pretrained,
            name,
            trust_remote_code=True,
        )
    return _IMPORT_TOKENIZER_CACHE[name]


def _source_turn_sampling_counts(turn: Mapping[str, Any]) -> tuple[int, int, int]:
    candidates_payload = list(turn.get("candidates", []))
    sampling = dict(turn.get("sampling") or {})
    compliant_default = sum(
        1 for candidate in candidates_payload if bool(candidate.get("compliant", True))
    )
    attempts_used = int(sampling.get("attempts_used", len(candidates_payload)))
    compliant_count = int(sampling.get("compliant_count", compliant_default))
    reject_count = int(sampling.get("reject_count", attempts_used - compliant_count))
    return attempts_used, compliant_count, reject_count


def _is_synthetic_exit_max_steps_terminal_turn(
    turn: Mapping[str, Any],
    *,
    exit_status: str,
) -> bool:
    if "exit_max_steps" not in exit_status:
        return False
    if turn.get("selected_idx") is not None:
        return False
    if list(turn.get("candidates", [])):
        return False
    attempts_used, compliant_count, reject_count = _source_turn_sampling_counts(turn)
    return attempts_used == 0 and compliant_count == 0 and reject_count == 0


def _import_source_trajectory_turns(
    candidate_traj: Mapping[str, Any],
    *,
    raw_turn_mode: str = SWE_RAW_TURN_MODE_SELECTED_ONLY,
) -> list[Mapping[str, Any]]:
    raw_turn_mode = validate_swe_raw_turn_mode(raw_turn_mode)
    turns = list(candidate_traj.get("trajectory", []))
    exit_status = str(candidate_traj.get("info", {}).get("exit_status") or "")
    if turns and _is_synthetic_exit_max_steps_terminal_turn(
        turns[-1],
        exit_status=exit_status,
    ):
        turns = turns[:-1]
    if raw_turn_mode == SWE_RAW_TURN_MODE_SELECTED_ONLY:
        return [
            turn
            for turn in turns
            if isinstance(turn.get("selected_idx"), int)
            and not isinstance(turn.get("selected_idx"), bool)
        ]
    return turns


def _infer_trajectory_config(
    trajectory_turns: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> TrajectoryConfig:
    turns = list(trajectory_turns)
    top_k_width = 0
    max_attempts = 0
    for turn in turns:
        attempts_used, _compliant_count, _reject_count = _source_turn_sampling_counts(turn)
        max_attempts = max(max_attempts, attempts_used)
        for candidate in turn.get("candidates", []):
            for row in candidate.get("top_k", []):
                top_k_width = max(top_k_width, len(row or []))
    return TrajectoryConfig(
        K=top_k_width,
        M=max_attempts,
        N=len(turns),
        policy="swe_candidate_traj_import",
        seed=int(seed),
        estimator=None,
        rejection=None,
    )


def _export_turn(
    turn_idx: int,
    turn: Mapping[str, Any],
    *,
    parser_mode: str,
    trajectory_format: str,
    gemma_reasoning_tokenizer: Any | None = None,
) -> Turn:
    candidates_payload = list(turn.get("candidates", []))
    selected_idx_raw = turn.get("selected_idx")
    selected_idx = None if selected_idx_raw is None else int(selected_idx_raw)
    if selected_idx is not None and not (0 <= selected_idx < len(candidates_payload)):
        raise ValueError(
            f"candidate.traj turn {turn_idx} has selected_idx={selected_idx} outside persisted candidate range"
        )

    candidates = [
        _export_candidate(
            candidate,
            observation=(turn.get("observation") if selected_idx == idx else None),
            parser_mode=parser_mode,
            gemma_reasoning_tokenizer=gemma_reasoning_tokenizer,
        )
        for idx, candidate in enumerate(candidates_payload)
    ]

    attempts_used, compliant_count, reject_count = _source_turn_sampling_counts(turn)
    return Turn(
        turn_idx=turn_idx,
        prompt=str(turn.get("prompt") or ""),
        candidates=candidates,
        selected_idx=selected_idx,
        selection_policy=str((turn.get("extra_info") or {}).get("selection_policy") or SWE_SELECTION_POLICY),
        attempts_used=attempts_used,
        reject_count=reject_count,
        compliant_count=compliant_count,
        state=(None if turn.get("state") is None else dict(turn.get("state") or {})),
        tool_call_ids=None,
        step_info={
            "query_snapshot": list(turn.get("query") or []),
            "source_selected_idx": selected_idx,
            "selected_path": {
                "thought": turn.get("thought"),
                "action": turn.get("action"),
                "response": turn.get("response"),
                "observation": turn.get("observation"),
            },
            "source_trajectory_format": trajectory_format,
        },
    )


def _export_candidate(
    candidate: Mapping[str, Any],
    *,
    observation: Any,
    parser_mode: str,
    gemma_reasoning_tokenizer: Any | None = None,
) -> CandidateRecord:
    source_response_token_ids = [
        int(token_id) for token_id in candidate.get("token_ids") or []
    ]
    source_response_token_logprobs = [
        float(value) for value in candidate.get("token_logprobs") or []
    ]
    source_response_top_k = _to_top_k_records(candidate.get("top_k") or [])
    if not (
        len(source_response_token_ids)
        == len(source_response_token_logprobs)
        == len(source_response_top_k)
    ):
        raise ValueError("candidate.traj candidate token arrays are misaligned")

    accepted_text = str(candidate.get("accepted_text") or "")
    raw_response = str(candidate.get("raw_response") or "")
    parsed_thought = (
        None if candidate.get("thought") is None else str(candidate.get("thought"))
    )
    char_spans = dict(candidate.get("char_spans") or {})
    token_spans = dict(candidate.get("token_spans") or {})
    response_token_ids = source_response_token_ids
    response_token_logprobs = source_response_token_logprobs
    response_top_k = source_response_top_k
    current_reasoning_projection = None
    if gemma_reasoning_tokenizer is not None:
        projection = _project_gemma_current_reasoning_candidate(
            candidate,
            tokenizer=gemma_reasoning_tokenizer,
        )
        if projection is not None:
            accepted_text = projection.accepted_text
            raw_response = projection.raw_response
            parsed_thought = projection.parsed_thought
            char_spans = projection.char_spans
            token_spans = projection.token_spans
            response_token_ids = source_response_token_ids[
                projection.response_token_start : projection.response_token_end
            ]
            response_token_logprobs = source_response_token_logprobs[
                projection.response_token_start : projection.response_token_end
            ]
            response_top_k = source_response_top_k[
                projection.response_token_start : projection.response_token_end
            ]
            current_reasoning_projection = projection.metadata

    required_span_keys = {
        "accepted_start",
        "accepted_end",
        "action_start",
        "action_end",
    }
    missing = required_span_keys - set(token_spans)
    if missing:
        raise ValueError(
            f"candidate.traj candidate is missing token span keys: {sorted(missing)}"
        )
    accepted_start = int(token_spans["accepted_start"])
    accepted_end = int(token_spans["accepted_end"])
    if not (0 <= accepted_start <= accepted_end <= len(response_token_ids)):
        raise ValueError("candidate.traj candidate accepted token span is outside response token arrays")

    backend = {
        "swe_candidate_idx": int(candidate.get("candidate_idx", 0)),
        "source_format": SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT,
        "char_spans": char_spans,
        "token_spans": token_spans,
        "response_token_ids": response_token_ids,
        "response_token_logprobs": response_token_logprobs,
        "response_top_k": _top_k_rows_to_jsonable(response_top_k),
    }
    if current_reasoning_projection is not None:
        backend["current_reasoning_projection"] = current_reasoning_projection

    return CandidateRecord(
        text=accepted_text,
        token_ids=list(response_token_ids[accepted_start:accepted_end]),
        token_logprobs=list(response_token_logprobs[accepted_start:accepted_end]),
        top_k=list(response_top_k[accepted_start:accepted_end]),
        finish_reason=str(candidate.get("finish_reason") or "unknown"),
        parse_status=str(candidate.get("parse_status") or "ok"),
        raw_response=raw_response,
        parse_error=(None if candidate.get("parse_error") is None else str(candidate.get("parse_error"))),
        parsed_thought=parsed_thought,
        parsed_action=build_action(
            "bash",
            {"command": str(candidate.get("action") or "")},
            text=str(candidate.get("action") or ""),
            parser_type=parser_mode,
            backend_source="swe_candidate_traj_import",
        ),
        observation=(None if observation is None else str(observation)),
        backend=backend,
        compliant=bool(candidate.get("compliant", True)),
        non_compliant_reason=(
            None if candidate.get("compliance_reason") is None else str(candidate.get("compliance_reason"))
        ),
    )


def _to_top_k_records(rows: Sequence[Sequence[Mapping[str, Any]]]) -> list[list[TopKEntryRec]]:
    return [
        [
            TopKEntryRec(token_id=int(entry["token_id"]), logprob=float(entry["logprob"]))
            for entry in step
        ]
        for step in rows
    ]


def _top_k_rows_to_jsonable(rows: Sequence[Sequence[TopKEntryRec]]) -> list[list[dict[str, float | int]]]:
    return [
        [{"token_id": int(entry.token_id), "logprob": float(entry.logprob)} for entry in step]
        for step in rows
    ]
