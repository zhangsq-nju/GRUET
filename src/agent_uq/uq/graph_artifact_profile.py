from __future__ import annotations

import datetime as _datetime
import json
import os
import pathlib
import tempfile
from typing import Any, Mapping

from agent_uq.uq.graph_artifact_io import _read_record_file
from agent_uq.uq.graph_artifact_paths import (
    _is_graph_record_path,
    _list_graph_cache_entries,
    _list_graph_record_paths,
    graph_cache_profile_metadata_path,
    normalize_graph_artifact_cache_adoption,
    normalize_graph_artifact_cache_mode,
)
from agent_uq.uq.graph_artifact_types import (
    GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT,
    GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT,
    GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY,
    GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    GRAPH_CACHE_PROFILE_KEYS,
    GRAPH_CACHE_PROFILE_METADATA_FILENAME,
    GraphArtifactProfileError,
    GraphArtifactWriteError,
)


def _read_graph_cache_profile_payload(graph_artifact_dir: pathlib.Path) -> dict[str, Any] | None:
    path = graph_cache_profile_metadata_path(graph_artifact_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GraphArtifactProfileError(
            f"graph cache profile metadata is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GraphArtifactProfileError(
            f"graph cache profile metadata must be an object: {path}"
        )
    return dict(payload)


def read_graph_cache_profile_metadata(
    graph_artifact_dir: pathlib.Path,
) -> dict[str, Any] | None:


    payload = _read_graph_cache_profile_payload(graph_artifact_dir)
    return None if payload is None else dict(payload)


def _normalize_expected_profile(expected_profile: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(expected_profile)
    missing = [key for key in GRAPH_CACHE_PROFILE_KEYS if key not in expected]
    if missing:
        raise GraphArtifactProfileError(
            f"expected graph cache profile is missing required key(s): {missing}"
        )
    normalized = {key: expected[key] for key in GRAPH_CACHE_PROFILE_KEYS}
    normalized["artifact_schema_version"] = int(normalized["artifact_schema_version"])
    normalized["action_similarity_threshold"] = float(
        normalized["action_similarity_threshold"]
    )
    for key in (
        "raw_profile_id",
        "nli_execution_mode",
        "merge_way",
        "sim_way",
        "action_similarity_model_name",
    ):
        normalized[key] = str(normalized[key])
    return normalized


def _profile_subset(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in GRAPH_CACHE_PROFILE_KEYS if key not in payload]
    if missing:
        raise GraphArtifactProfileError(
            f"graph cache profile metadata is missing required key(s): {missing}"
        )
    return _normalize_expected_profile(payload)


def validate_graph_cache_profile_metadata(
    graph_artifact_dir: pathlib.Path,
    *,
    expected_profile: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_graph_cache_profile_payload(graph_artifact_dir)
    if payload is None:
        raise GraphArtifactProfileError(
            f"graph cache profile metadata is required under {graph_artifact_dir}"
        )
    expected = _normalize_expected_profile(expected_profile)
    actual = _profile_subset(payload)
    mismatches = [
        f"{key}: expected {expected[key]!r}, found {actual[key]!r}"
        for key in GRAPH_CACHE_PROFILE_KEYS
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise GraphArtifactProfileError(
            "graph cache profile metadata does not match effective profile: "
            + "; ".join(mismatches)
        )
    return dict(payload)


def _atomic_write_profile_metadata(
    graph_artifact_dir: pathlib.Path,
    *,
    expected_profile: Mapping[str, Any],
    adopted_unprofiled: bool,
) -> pathlib.Path:
    root = pathlib.Path(graph_artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = graph_cache_profile_metadata_path(root)
    payload = _normalize_expected_profile(expected_profile)
    payload.update(
        {
            "created_by": "agent_uq.graph_artifact_cache",
            "created_at": _datetime.datetime.now(_datetime.UTC).isoformat(),
            "adopted_unprofiled": bool(adopted_unprofiled),
        }
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(root),
        text=True,
    )
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise GraphArtifactWriteError(
                f"failed to write graph cache profile metadata {path}: {exc}"
            ) from exc
    return path


def _validate_unprofiled_graph_records_for_adoption(
    graph_artifact_dir: pathlib.Path,
    *,
    expected_profile: Mapping[str, Any],
) -> None:
    expected = _normalize_expected_profile(expected_profile)
    for path in _list_graph_record_paths(graph_artifact_dir):
        record = _read_record_file(path)
        if int(record.artifact_schema_version) != expected["artifact_schema_version"]:
            raise GraphArtifactProfileError(
                f"graph artifact {path} has artifact_schema_version "
                f"{record.artifact_schema_version!r}, expected "
                f"{expected['artifact_schema_version']!r}"
            )
        for turn in record.turns.values():
            for entry in turn.artifacts.values():
                params = dict(entry.build_params)
                for key in (
                    "merge_way",
                    "sim_way",
                    "action_similarity_model_name",
                    "action_similarity_threshold",
                ):
                    if key not in params:
                        raise GraphArtifactProfileError(
                            f"graph artifact {path} build params are missing {key!r}"
                        )
                    if key == "action_similarity_threshold":
                        actual = float(params[key])
                    else:
                        actual = str(params[key])
                    if actual != expected[key]:
                        raise GraphArtifactProfileError(
                            f"graph artifact {path} build param {key!r} is "
                            f"{actual!r}, expected {expected[key]!r}"
                        )


def prepare_graph_artifact_cache_root(
    graph_artifact_dir: pathlib.Path,
    *,
    expected_profile: Mapping[str, Any] | None,
    access_mode: str = GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    adoption: str = GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT,
) -> None:


    if expected_profile is None:
        return
    mode = normalize_graph_artifact_cache_mode(access_mode)
    adoption_mode = normalize_graph_artifact_cache_adoption(adoption)
    if mode != GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE and (
        adoption_mode == GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT
    ):
        raise GraphArtifactProfileError(
            "graph_artifact_cache_adoption=adopt_unprofiled is only valid in "
            "read_write mode"
        )

    root = pathlib.Path(graph_artifact_dir)
    existing_profile = _read_graph_cache_profile_payload(root)
    if existing_profile is not None:
        validate_graph_cache_profile_metadata(root, expected_profile=expected_profile)
        return

    if mode == GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY:
        raise GraphArtifactProfileError(
            f"read_only graph artifact cache requires profile metadata under {root}"
        )

    entries = _list_graph_cache_entries(root)
    if not entries:
        _atomic_write_profile_metadata(
            root,
            expected_profile=expected_profile,
            adopted_unprofiled=False,
        )
        return

    unknown_entries = [
        p for p in entries
        if p.name != GRAPH_CACHE_PROFILE_METADATA_FILENAME and not _is_graph_record_path(p)
    ]
    if adoption_mode != GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT:
        raise GraphArtifactProfileError(
            f"configured graph artifact root is non-empty but lacks profile metadata: {root}"
        )
    if unknown_entries:
        raise GraphArtifactProfileError(
            "cannot adopt unprofiled graph artifact root with unknown entries: "
            + ", ".join(str(p) for p in unknown_entries)
        )
    _validate_unprofiled_graph_records_for_adoption(
        root,
        expected_profile=expected_profile,
    )
    _atomic_write_profile_metadata(
        root,
        expected_profile=expected_profile,
        adopted_unprofiled=True,
    )
