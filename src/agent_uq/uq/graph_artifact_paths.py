from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Iterator

from agent_uq.uq.graph_artifact_types import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
    DEFAULT_ACTION_SIMILARITY_THRESHOLD,
    GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT,
    GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT,
    GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY,
    GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    GRAPH_CACHE_PROFILE_METADATA_FILENAME,
    GraphArtifactPathAmbiguityError,
    GraphArtifactProfileError,
    GraphArtifactRecord,
)


def derive_build_key(
    *,
    responses: list[dict[str, Any]],
    merge_way: str,
    sim_way: str,
    action_similarity_model_name: str = DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
    action_similarity_threshold: float = DEFAULT_ACTION_SIMILARITY_THRESHOLD,
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION,
) -> str:


    payload = {
        "artifact_schema_version": int(artifact_schema_version),
        "merge_way": str(merge_way),
        "sim_way": str(sim_way),
        "action_similarity_model_name": str(action_similarity_model_name),
        "action_similarity_threshold": float(action_similarity_threshold),
        "responses": responses,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_record_path(
    graph_artifact_dir: pathlib.Path,
    record_id: str,
    *,
    gzip_records: bool,
) -> pathlib.Path:


    json_path = graph_artifact_dir / f"{record_id}.json"
    gz_path = graph_artifact_dir / f"{record_id}.json.gz"
    json_exists = json_path.is_file()
    gz_exists = gz_path.is_file()

    if json_exists and gz_exists:
        raise GraphArtifactPathAmbiguityError(
            f"graph-artifact path ambiguity: both {json_path} and {gz_path} exist; "
            f"leaf is in mixed-format limbo and must be cleaned up by an operator"
        )

    if gzip_records:
        if json_exists and not gz_exists:
            raise GraphArtifactPathAmbiguityError(
                f"graph-artifact path ambiguity: scorer policy requests gzip but "
                f"existing artifact at {json_path} is uncompressed"
            )
        return gz_path

    if gz_exists and not json_exists:
        raise GraphArtifactPathAmbiguityError(
            f"graph-artifact path ambiguity: scorer policy requests uncompressed "
            f"but existing artifact at {gz_path} is gzipped"
        )
    return json_path


def normalize_graph_artifact_cache_mode(value: Any) -> str:
    mode = GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE if value is None else str(value)
    if mode not in {
        GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
        GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY,
    }:
        raise GraphArtifactProfileError(
            "graph_artifact_cache_mode must be 'read_write' or 'read_only'; "
            f"got {mode!r}"
        )
    return mode


def normalize_graph_artifact_cache_adoption(value: Any) -> str:
    adoption = GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT if value is None else str(value)
    if adoption not in {
        GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT,
        GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT,
    }:
        raise GraphArtifactProfileError(
            "graph_artifact_cache_adoption must be 'reject_unprofiled' or "
            f"'adopt_unprofiled'; got {adoption!r}"
        )
    return adoption


def graph_cache_profile_metadata_path(graph_artifact_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(graph_artifact_dir) / GRAPH_CACHE_PROFILE_METADATA_FILENAME


def _is_graph_record_path(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    if path.name == GRAPH_CACHE_PROFILE_METADATA_FILENAME:
        return False
    return path.name.endswith(".json") or path.name.endswith(".json.gz")


def _list_graph_cache_entries(graph_artifact_dir: pathlib.Path) -> list[pathlib.Path]:
    root = pathlib.Path(graph_artifact_dir)
    if not root.exists():
        return []
    if not root.is_dir():
        raise GraphArtifactProfileError(f"graph artifact root is not a directory: {root}")
    return sorted(root.iterdir(), key=lambda p: p.name)


def _list_graph_record_paths(graph_artifact_dir: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in _list_graph_cache_entries(graph_artifact_dir) if _is_graph_record_path(p)]


def iter_graph_artifact_records(
    graph_artifact_dir: str | pathlib.Path,
) -> Iterator[tuple[pathlib.Path, GraphArtifactRecord]]:

    from agent_uq.uq.graph_artifact_io import _read_record_file

    for path in _list_graph_record_paths(pathlib.Path(graph_artifact_dir)):
        yield path, _read_record_file(path)
