from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml

from agent_uq.swe_workflow.common import (
    DEFAULT_SWE_AGENT_CONFIG_ROOT,
    REPO_ROOT,
    SUPPORTED_SWE_PARSER_MODE,
    SWE_CANDIDATE_TRAJECTORY_FORMAT,
    SWE_RAW_TURN_MODE_SELECTED_ONLY,
    repo_relative_path,
    sha256_file,
    validate_swe_raw_turn_mode,
)
from agent_uq.cache.leaf_manifest import (
    canonical_sampling_cache_view,
    read_leaf_manifest,
    target_ids_complete_on_disk,
)


@dataclass(frozen=True)
class SWEImportPlan:
    target_record_ids: list[str]
    candidate_traj_by_record: dict[str, pathlib.Path]
    normalized_outcomes: dict[str, dict[str, bool]]
    normalized_report: dict[str, list[str]]

@dataclass(frozen=True)
class ResolvedAuthoritativeSWEConfig:
    config_path: pathlib.Path
    config_path_display: str
    demonstration_paths: tuple[pathlib.Path, ...]
    demonstration_display_paths: tuple[str, ...]
    parser_mode: str

@dataclass(frozen=True)
class CandidateTrajSourceMetadata:
    trajectory_format: str
    replay_parser_mode: str | None
    replay_demonstration_paths: tuple[str, ...] | None

def swe_agent_config_root() -> pathlib.Path:
    return pathlib.Path(os.getenv("SWE_AGENT_CONFIG_ROOT", DEFAULT_SWE_AGENT_CONFIG_ROOT)).resolve()


def resolve_swe_agent_semantic_path(value: Any, *, field_name: str) -> pathlib.Path:
    if value in (None, ""):
        raise ValueError(f"{field_name} must be set")
    path = pathlib.Path(str(value))
    if not path.is_absolute():
        if path.parts[:2] == ("vendor", "SWE-agent"):
            path = REPO_ROOT / path
        else:
            path = swe_agent_config_root() / path
    return path.resolve()


def _load_yaml_mapping(path: pathlib.Path, *, field_name: str) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must parse to a mapping")
    return payload


def _contains_unsupported_multimodal_hook(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"image_parsing", "image_url", "image_urls"}:
                return True
            if _contains_unsupported_multimodal_hook(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_unsupported_multimodal_hook(item) for item in value)
    return False


def resolve_authoritative_swe_config(config_path_value: Any) -> ResolvedAuthoritativeSWEConfig:
    config_path = resolve_swe_agent_semantic_path(
        config_path_value,
        field_name="authoritative.config_path",
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"authoritative SWE config not found: {config_path}")

    payload = _load_yaml_mapping(config_path, field_name="authoritative.config_path")
    if _contains_unsupported_multimodal_hook(payload):
        raise ValueError(
            "authoritative.config_path uses unsupported multimodal SWE prompt hooks "
            "(image_parsing/image_url)"
        )
    agent_payload = payload.get("agent")
    if not isinstance(agent_payload, Mapping):
        raise ValueError("authoritative.config_path must define an `agent` mapping")
    templates_payload = agent_payload.get("templates")
    if not isinstance(templates_payload, Mapping):
        raise ValueError("authoritative.config_path must define `agent.templates`")
    tools_payload = agent_payload.get("tools")
    if not isinstance(tools_payload, Mapping):
        raise ValueError("authoritative.config_path must define `agent.tools`")
    parse_function_payload = tools_payload.get("parse_function")
    if not isinstance(parse_function_payload, Mapping):
        raise ValueError("authoritative.config_path must define `agent.tools.parse_function`")

    parser_mode = parse_function_payload.get("type")
    if parser_mode in (None, ""):
        raise ValueError("authoritative.config_path must define `agent.tools.parse_function.type`")

    demonstrations_value = templates_payload.get("demonstrations", [])
    if demonstrations_value is None:
        demonstrations_value = []
    if not isinstance(demonstrations_value, list):
        raise ValueError("authoritative.config_path `agent.templates.demonstrations` must be a list")

    demonstration_paths: list[pathlib.Path] = []
    for idx, demo_value in enumerate(demonstrations_value):
        if demo_value in (None, ""):
            raise ValueError(
                "authoritative.config_path `agent.templates.demonstrations` entries must be non-empty paths"
            )
        demonstration_paths.append(
            resolve_swe_agent_semantic_path(
                demo_value,
                field_name=f"authoritative.config_path agent.templates.demonstrations[{idx}]",
            )
        )

    return ResolvedAuthoritativeSWEConfig(
        config_path=config_path,
        config_path_display=repo_relative_path(config_path),
        demonstration_paths=tuple(demonstration_paths),
        demonstration_display_paths=tuple(repo_relative_path(path) for path in demonstration_paths),
        parser_mode=str(parser_mode),
    )

def assert_authoritative_swe_assets(authoritative: ResolvedAuthoritativeSWEConfig) -> None:
    if authoritative.parser_mode != SUPPORTED_SWE_PARSER_MODE:
        raise ValueError(
            "unsupported authoritative SWE parser mode: "
            f"{authoritative.parser_mode!r}; expected {SUPPORTED_SWE_PARSER_MODE!r}"
        )
    for demo_path in authoritative.demonstration_paths:
        if not demo_path.is_file():
            raise FileNotFoundError(f"authoritative SWE demo not found: {demo_path}")


def load_candidate_traj(path: str | pathlib.Path) -> dict[str, Any]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("trajectory_format") != SWE_CANDIDATE_TRAJECTORY_FORMAT:
        raise ValueError(
            f"{path} is not a candidate.traj import source: "
            f"expected trajectory_format={SWE_CANDIDATE_TRAJECTORY_FORMAT!r}"
        )
    for turn_idx, turn in enumerate(data.get("trajectory", [])):
        prompt = turn.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(
                f"{path} turn {turn_idx} uses unsupported non-string prompt payload"
            )
        query = turn.get("query")
        if query is None:
            continue
        if not isinstance(query, list):
            raise ValueError(f"{path} turn {turn_idx} uses unsupported non-list query payload")
        for message_idx, message in enumerate(query):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"{path} turn {turn_idx} query[{message_idx}] must be a mapping"
                )
            if "image_url" in message or "image_urls" in message:
                raise ValueError(
                    f"{path} turn {turn_idx} query[{message_idx}] uses unsupported multimodal payload"
                )
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError(
                    f"{path} turn {turn_idx} query[{message_idx}] uses unsupported non-string content"
                )
    return data


def discover_candidate_traj_sources(
    candidate_traj_root: str | pathlib.Path,
    *,
    target_record_ids: Sequence[str] | None = None,
    pattern: str = "**/*.traj",
) -> dict[str, pathlib.Path]:
    root = pathlib.Path(candidate_traj_root)
    if not root.exists():
        raise FileNotFoundError(f"candidate.traj root does not exist: {root}")

    target_set = None if target_record_ids is None else {str(record_id) for record_id in target_record_ids}
    matches: dict[str, list[pathlib.Path]] = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        record_id = path.stem
        if target_set is not None and record_id not in target_set:
            continue
        matches.setdefault(record_id, []).append(path.resolve())

    if target_record_ids is None:
        duplicates = {record_id: paths for record_id, paths in matches.items() if len(paths) != 1}
        if duplicates:
            detail = ", ".join(
                f"{record_id}: {[repo_relative_path(path) for path in paths]}"
                for record_id, paths in sorted(duplicates.items())
            )
            raise ValueError(f"candidate.traj discovery found ambiguous coverage: {detail}")
        return {record_id: paths[0] for record_id, paths in sorted(matches.items())}

    resolved: dict[str, pathlib.Path] = {}
    for record_id in target_record_ids:
        record_id = str(record_id)
        paths = matches.get(record_id, [])
        if not paths:
            raise ValueError(f"missing candidate.traj coverage for target_record_id={record_id!r}")
        if len(paths) > 1:
            detail = ", ".join(repo_relative_path(path) for path in paths)
            raise ValueError(f"duplicate candidate.traj coverage for target_record_id={record_id!r}: {detail}")
        resolved[record_id] = paths[0]
    return resolved


def normalize_evaluator_report(
    evaluator_report_path: str | pathlib.Path,
    *,
    target_record_ids: Sequence[str],
) -> tuple[dict[str, dict[str, bool]], dict[str, list[str]]]:
    payload = json.loads(pathlib.Path(evaluator_report_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("supported SWE evaluator report must be a JSON object")

    submitted_ids = payload.get("submitted_ids")
    resolved_ids = payload.get("resolved_ids", payload.get("resolved"))
    if not isinstance(submitted_ids, list) or not isinstance(resolved_ids, list):
        raise ValueError(
            "supported SWE evaluator report must expose list-like `submitted_ids` and `resolved_ids` "
            "(or legacy `resolved`)"
        )

    submitted = {str(item) for item in submitted_ids}
    resolved = {str(item) for item in resolved_ids}
    errors = {str(item) for item in payload.get("error_ids", [])}
    failed = errors.intersection(str(item) for item in target_record_ids)
    if failed:
        raise ValueError(f"SWE harness error for requested targets: {sorted(failed)}")
    if any(key in payload for key in ("completed_ids", "error_ids", "empty_patch_ids", "incomplete_ids", "unresolved_ids")):
        terminal = {str(item) for item in payload.get("completed_ids", [])}
        terminal.update(str(item) for item in payload.get("empty_patch_ids", []))
        missing = set(map(str, target_record_ids)) - terminal
        if missing:
            raise ValueError(f"SWE harness missing terminal outcomes for requested targets: {sorted(missing)}")
    normalized = {
        "submitted_ids": sorted(submitted),
        "resolved_ids": sorted(resolved),
    }

    outcomes: dict[str, dict[str, bool]] = {}
    for record_id in target_record_ids:
        record_id = str(record_id)
        if record_id not in submitted:
            raise ValueError(f"evaluator report does not cover target_record_id={record_id!r}")
        outcomes[record_id] = {
            "submitted": True,
            "resolved": record_id in resolved,
        }
    return outcomes, normalized


def build_import_plan(
    *,
    candidate_traj_root: str | pathlib.Path,
    evaluator_report_path: str | pathlib.Path,
    target_record_ids: Sequence[str] | None = None,
    candidate_traj_pattern: str = "**/*.traj",
) -> SWEImportPlan:
    resolved_sources = discover_candidate_traj_sources(
        candidate_traj_root,
        target_record_ids=target_record_ids,
        pattern=candidate_traj_pattern,
    )
    planned_ids = list(target_record_ids or sorted(resolved_sources))
    outcomes, normalized_report = normalize_evaluator_report(
        evaluator_report_path,
        target_record_ids=planned_ids,
    )
    return SWEImportPlan(
        target_record_ids=planned_ids,
        candidate_traj_by_record=resolved_sources,
        normalized_outcomes=outcomes,
        normalized_report=normalized_report,
    )


def _load_replay_config_mapping(value: Any) -> Mapping[str, Any] | None:
    try:
        if isinstance(value, str):
            payload = json.loads(value)
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            return None
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _extract_replay_parser_mode(payload: Mapping[str, Any]) -> str | None:
    agent_payload = payload.get("agent")
    if not isinstance(agent_payload, Mapping):
        return None
    tools_payload = agent_payload.get("tools")
    if not isinstance(tools_payload, Mapping):
        return None
    parse_function_payload = tools_payload.get("parse_function")
    if not isinstance(parse_function_payload, Mapping):
        return None
    parser_mode = parse_function_payload.get("type")
    return str(parser_mode) if parser_mode not in (None, "") else None


def _extract_replay_demonstration_projection(payload: Mapping[str, Any]) -> tuple[str, ...] | None:
    agent_payload = payload.get("agent")
    if not isinstance(agent_payload, Mapping):
        return None
    templates_payload = agent_payload.get("templates")
    if not isinstance(templates_payload, Mapping):
        return None
    demonstrations_value = templates_payload.get("demonstrations")
    if not isinstance(demonstrations_value, list):
        return None

    normalized: list[str] = []
    for idx, demo_value in enumerate(demonstrations_value):
        if demo_value in (None, ""):
            return None
        normalized.append(
            repo_relative_path(
                resolve_swe_agent_semantic_path(
                    demo_value,
                    field_name=f"candidate.traj.replay_config.agent.templates.demonstrations[{idx}]",
                )
            )
        )
    return tuple(normalized)


def inspect_candidate_traj_source_metadata(path: str | pathlib.Path) -> CandidateTrajSourceMetadata:
    data = load_candidate_traj(path)
    replay_payload = _load_replay_config_mapping(data.get("replay_config"))
    replay_parser_mode = None
    replay_demonstration_paths = None
    if replay_payload is not None:
        replay_parser_mode = _extract_replay_parser_mode(replay_payload)
        replay_demonstration_paths = _extract_replay_demonstration_projection(replay_payload)
    return CandidateTrajSourceMetadata(
        trajectory_format=str(data["trajectory_format"]),
        replay_parser_mode=replay_parser_mode,
        replay_demonstration_paths=replay_demonstration_paths,
    )


def resolve_imported_trajectory_format(
    *,
    target_record_ids: Sequence[str],
    metadata_by_record: Mapping[str, CandidateTrajSourceMetadata],
) -> str:
    formats = {metadata_by_record[record_id].trajectory_format for record_id in target_record_ids}
    if not formats:
        raise ValueError("no candidate.traj sources were resolved for trajectory-format validation")
    if len(formats) != 1:
        raise ValueError(f"mixed candidate.traj trajectory formats are not supported: {sorted(formats)}")
    return next(iter(formats))


def validate_import_only_replay_contract(
    *,
    target_record_ids: Sequence[str],
    candidate_traj_by_record: Mapping[str, pathlib.Path],
    metadata_by_record: Mapping[str, CandidateTrajSourceMetadata],
    authoritative: ResolvedAuthoritativeSWEConfig,
) -> None:
    expected_demos = authoritative.demonstration_display_paths
    for record_id in target_record_ids:
        path = candidate_traj_by_record[record_id]
        metadata = metadata_by_record[record_id]
        if (
            metadata.replay_parser_mode is not None
            and metadata.replay_parser_mode != authoritative.parser_mode
        ):
            raise ValueError(
                "candidate.traj replay_config parser mode mismatch for "
                f"{repo_relative_path(path)}: imported {metadata.replay_parser_mode!r}, "
                f"expected {authoritative.parser_mode!r} from {authoritative.config_path_display}"
            )
        if (
            metadata.replay_demonstration_paths is not None
            and metadata.replay_demonstration_paths != expected_demos
        ):
            raise ValueError(
                "candidate.traj replay_config demonstrations mismatch for "
                f"{repo_relative_path(path)}: imported {list(metadata.replay_demonstration_paths)!r}, "
                f"expected {list(expected_demos)!r} from {authoritative.config_path_display}"
            )


def resolve_swe_import_selectors(
    *,
    authoritative: ResolvedAuthoritativeSWEConfig,
    target_record_ids: Sequence[str],
    candidate_traj_by_record: Mapping[str, pathlib.Path],
    normalized_outcomes: Mapping[str, Mapping[str, bool]],
    imported_trajectory_format: str,
    dataset_key: str,
    target_type: str,
    raw_turn_mode: str = SWE_RAW_TURN_MODE_SELECTED_ONLY,
) -> dict[str, Any]:
    raw_turn_mode = validate_swe_raw_turn_mode(raw_turn_mode)
    source_projection = [
        {
            "record_id": record_id,
            "path": repo_relative_path(candidate_traj_by_record[record_id]),
            "sha256": sha256_file(candidate_traj_by_record[record_id]),
        }
        for record_id in sorted(target_record_ids)
    ]
    outcome_projection = [
        {
            "record_id": record_id,
            "submitted": bool(normalized_outcomes[record_id]["submitted"]),
            "resolved": bool(normalized_outcomes[record_id]["resolved"]),
        }
        for record_id in sorted(target_record_ids)
    ]
    return {
        "workflow": "vendored_swe_agent_candidate_traj",
        "authoritative_config_path": authoritative.config_path_display,
        "authoritative_demonstrations": list(authoritative.demonstration_display_paths),
        "candidate_traj_sources": source_projection,
        "evaluator_outcomes": outcome_projection,
        "trajectory_format": imported_trajectory_format,
        "parser_mode": authoritative.parser_mode,
        "dataset_key": dataset_key,
        "target_type": target_type,
        "raw_turn_mode": raw_turn_mode,
    }


def build_swe_manifest_runtime_block(
    *,
    authoritative: ResolvedAuthoritativeSWEConfig,
    imported_trajectory_format: str,
    candidate_traj_by_record: Mapping[str, pathlib.Path],
    evaluator_report_path: str | pathlib.Path,
    prediction_source_path: str | pathlib.Path | None = None,
    raw_turn_mode: str = SWE_RAW_TURN_MODE_SELECTED_ONLY,
) -> dict[str, Any]:
    raw_turn_mode = validate_swe_raw_turn_mode(raw_turn_mode)
    runtime: dict[str, Any] = {
        "workflow": {
            "type": "vendored_swe_agent_candidate_traj",
            "authoritative_config_path": authoritative.config_path_display,
            "authoritative_demonstrations": list(authoritative.demonstration_display_paths),
            "parser_mode": authoritative.parser_mode,
            "trajectory_format": imported_trajectory_format,
            "raw_turn_mode": raw_turn_mode,
        },
        "candidate_traj_sources": {
            record_id: repo_relative_path(path)
            for record_id, path in sorted(candidate_traj_by_record.items())
        },
        "evaluator_report_path": repo_relative_path(evaluator_report_path),
    }
    if prediction_source_path is not None:
        runtime["prediction_source_path"] = repo_relative_path(prediction_source_path)
    return runtime


def should_reuse_cached_swe_leaf(
    *,
    output_dir: str | pathlib.Path,
    sampling_selectors: Mapping[str, Any],
    target_record_ids: Sequence[str],
) -> bool:
    output_dir = pathlib.Path(output_dir)
    manifest = read_leaf_manifest(output_dir)
    if manifest is None:
        return False
    if canonical_sampling_cache_view(manifest.sampling) != canonical_sampling_cache_view(sampling_selectors):
        return False
    if list(manifest.target_record_ids) != [str(record_id) for record_id in target_record_ids]:
        return False
    complete, _missing = target_ids_complete_on_disk(output_dir / "raw", list(target_record_ids))
    return complete and int(manifest.raw_count) == len(target_record_ids)
