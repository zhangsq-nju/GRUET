from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_uq.swe_workflow.common import (
    DEFAULT_SWE_AGENT_CONFIG_ROOT,
    SWE_RAW_TURN_MODE_SELECTED_ONLY,
    utc_now_iso,
    utc_path_timestamp,
    validate_swe_raw_turn_mode,
)
from agent_uq.cache.leaf_manifest import read_leaf_manifest


@dataclass(frozen=True)
class SemanticLaunchManifest:
    launch_key: str
    key_payload: dict[str, Any]
    requested_target_record_ids: list[str]
    covered_target_record_ids: list[str]
    updated_at: str

def semantic_launch_manifest_path(output_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(output_dir) / "launch_manifest.json"


def read_semantic_launch_manifest(output_dir: str | pathlib.Path) -> SemanticLaunchManifest | None:
    path = semantic_launch_manifest_path(output_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"semantic launch manifest at {path} must be a JSON object")
    return SemanticLaunchManifest(
        launch_key=str(payload["launch_key"]),
        key_payload=dict(payload.get("key_payload") or {}),
        requested_target_record_ids=[str(record_id) for record_id in payload.get("requested_target_record_ids") or []],
        covered_target_record_ids=[str(record_id) for record_id in payload.get("covered_target_record_ids") or []],
        updated_at=str(payload.get("updated_at") or ""),
    )


def write_semantic_launch_manifest(
    output_dir: str | pathlib.Path,
    *,
    launch_key: str,
    key_payload: Mapping[str, Any],
    requested_target_record_ids: Sequence[str],
    covered_target_record_ids: Sequence[str],
    updated_at: str | None = None,
) -> pathlib.Path:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = semantic_launch_manifest_path(output_dir)
    payload = {
        "launch_key": str(launch_key),
        "key_payload": json.loads(json.dumps(key_payload, sort_keys=True)),
        "requested_target_record_ids": [str(record_id) for record_id in requested_target_record_ids],
        "covered_target_record_ids": [str(record_id) for record_id in covered_target_record_ids],
        "updated_at": str(updated_at or utc_now_iso()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def semantic_launch_traj_path(output_dir: str | pathlib.Path, record_id: str) -> pathlib.Path:
    output_dir = pathlib.Path(output_dir)
    return output_dir / record_id / f"{record_id}.traj"


def semantic_launch_pred_path(output_dir: str | pathlib.Path, record_id: str) -> pathlib.Path:
    output_dir = pathlib.Path(output_dir)
    return output_dir / record_id / f"{record_id}.pred"


def semantic_launch_preds_json_path(output_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(output_dir) / "preds.json"


def _load_vendored_merge_predictions():
    try:
        from sweagent.run.merge_predictions import merge_predictions
    except ModuleNotFoundError as exc:
        if exc.name != "sweagent":
            raise
        swe_agent_root = str(DEFAULT_SWE_AGENT_CONFIG_ROOT)
        if swe_agent_root not in sys.path:
            sys.path.insert(0, swe_agent_root)
        from sweagent.run.merge_predictions import merge_predictions
    return merge_predictions


def load_semantic_launch_traj(path: str | pathlib.Path) -> dict[str, Any] | None:
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def semantic_launch_traj_is_complete(path: str | pathlib.Path) -> bool:
    payload = load_semantic_launch_traj(path)
    if payload is None:
        return False
    info = payload.get("info")
    if not isinstance(info, Mapping):
        return False
    exit_status = info.get("exit_status")
    return exit_status not in (None, "", "early_exit")


def completed_semantic_launch_ids(
    output_dir: str | pathlib.Path,
    *,
    requested_target_record_ids: Sequence[str],
) -> list[str]:
    output_dir = pathlib.Path(output_dir)
    return [
        str(record_id)
        for record_id in requested_target_record_ids
        if semantic_launch_traj_is_complete(semantic_launch_traj_path(output_dir, str(record_id)))
    ]


def ensure_semantic_launch_predictions(
    output_dir: str | pathlib.Path,
    *,
    requested_target_record_ids: Sequence[str],
) -> None:
    merge_predictions = _load_vendored_merge_predictions()

    output_dir = pathlib.Path(output_dir)
    for record_id in requested_target_record_ids:
        record_id = str(record_id)
        traj_path = semantic_launch_traj_path(output_dir, record_id)
        pred_path = semantic_launch_pred_path(output_dir, record_id)
        if pred_path.is_file() or not semantic_launch_traj_is_complete(traj_path):
            continue
        payload = load_semantic_launch_traj(traj_path)
        if payload is None:
            continue
        submission = None
        info = payload.get("info")
        if isinstance(info, Mapping):
            submission = info.get("submission")
        pred_payload = {
            "model_name_or_path": output_dir.name,
            "instance_id": record_id,
            "model_patch": submission,
        }
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(pred_payload), encoding="utf-8")
    merge_predictions([output_dir], semantic_launch_preds_json_path(output_dir))


def ensure_semantic_launch_targets_compatible(
    output_dir: str | pathlib.Path,
    *,
    launch_key: str,
    requested_target_record_ids: Sequence[str],
) -> None:
    manifest = read_semantic_launch_manifest(output_dir)
    if manifest is None:
        return
    if manifest.launch_key != launch_key:
        raise ValueError(
            f"semantic launch manifest under {output_dir} belongs to {manifest.launch_key!r}, "
            f"expected {launch_key!r}"
        )
    previous = list(manifest.requested_target_record_ids)
    current = [str(record_id) for record_id in requested_target_record_ids]
    if not previous:
        return
    if previous == current:
        return
    if previous == current[: len(previous)]:
        return
    raise ValueError(
        "semantic launch cache only supports deterministic prefix-compatible target-set reuse; "
        f"existing request has {len(previous)} ids, current request has {len(current)} ids"
    )


def ensure_semantic_import_targets_compatible(
    output_dir: str | pathlib.Path,
    *,
    launch_key: str,
    requested_target_record_ids: Sequence[str],
) -> None:
    manifest = read_leaf_manifest(output_dir)
    if manifest is None:
        return
    manifest_launch_key = str(manifest.sampling.get("launch_key") or "")
    if manifest_launch_key and manifest_launch_key != launch_key:
        raise ValueError(
            f"semantic import manifest under {output_dir} belongs to launch_key "
            f"{manifest_launch_key!r}, expected {launch_key!r}"
        )
    previous = list(manifest.target_record_ids)
    current = [str(record_id) for record_id in requested_target_record_ids]
    if not previous:
        return
    if previous == current:
        return
    if previous == current[: len(previous)]:
        return
    raise ValueError(
        "semantic import cache only supports deterministic prefix-compatible target-set reuse; "
        f"existing request has {len(previous)} ids, current request has {len(current)} ids"
    )


def resolve_semantic_import_selectors(
    *,
    launch_key: str,
    raw_turn_mode: str = SWE_RAW_TURN_MODE_SELECTED_ONLY,
) -> dict[str, Any]:
    raw_turn_mode = validate_swe_raw_turn_mode(raw_turn_mode)
    return {
        "workflow": "vendored_swe_agent_candidate_traj_semantic_cache",
        "launch_key": str(launch_key),
        "raw_turn_mode": raw_turn_mode,
    }


def archive_semantic_import_leaf(output_dir: str | pathlib.Path, *, timestamp: str | None = None) -> list[pathlib.Path]:
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        return []
    stamp = timestamp or utc_path_timestamp()
    archived_paths: list[pathlib.Path] = []
    def _unique_archive_path(candidate: pathlib.Path) -> pathlib.Path:
        if not candidate.exists():
            return candidate
        suffix = 1
        while True:
            amended = candidate.with_name(f"{candidate.name}-{suffix}")
            if not amended.exists():
                return amended
            suffix += 1
    for name in ("raw", "scored", "nli_pairs", "graphs"):
        path = output_dir / name
        if not path.exists():
            continue
        archived = _unique_archive_path(output_dir / f"{name}.stale.{stamp}")
        path.rename(archived)
        archived_paths.append(archived)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        archived_manifest = _unique_archive_path(output_dir / f"manifest.stale.{stamp}.json")
        manifest_path.rename(archived_manifest)
        archived_paths.append(archived_manifest)
    return archived_paths


def archive_semantic_import_downstream_artifacts(
    output_dir: str | pathlib.Path,
    *,
    timestamp: str | None = None,
) -> list[pathlib.Path]:
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        return []
    stamp = timestamp or utc_path_timestamp()
    archived_paths: list[pathlib.Path] = []

    def _unique_archive_path(candidate: pathlib.Path) -> pathlib.Path:
        if not candidate.exists():
            return candidate
        suffix = 1
        while True:
            amended = candidate.with_name(f"{candidate.name}-{suffix}")
            if not amended.exists():
                return amended
            suffix += 1

    for name in ("scored", "nli_pairs", "graphs"):
        path = output_dir / name
        if not path.exists():
            continue
        archived = _unique_archive_path(output_dir / f"{name}.stale.{stamp}")
        path.rename(archived)
        archived_paths.append(archived)
    return archived_paths
