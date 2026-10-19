from __future__ import annotations

import pathlib

from agent_uq.uq.graph_artifact_io import (
    _atomic_write_record,
    _merge_records,
    _read_record_file,
)
from agent_uq.uq.graph_artifact_paths import (
    normalize_graph_artifact_cache_mode,
    resolve_record_path,
)
from agent_uq.uq.graph_artifact_types import (
    GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY,
    GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    GraphArtifactCacheMissError,
    GraphArtifactEntry,
    GraphArtifactRecord,
)


def read_only_session_from_record(
    *,
    graph_artifact_dir: str | pathlib.Path,
    record: GraphArtifactRecord,
    gzip_records: bool,
) -> "GraphArtifactRecordSession":

    session = GraphArtifactRecordSession(
        graph_artifact_dir=pathlib.Path(graph_artifact_dir),
        record_id=record.record_id,
        dataset_key=record.dataset_key,
        tokenizer_name=record.tokenizer_name,
        gzip_records=gzip_records,
        access_mode=GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY,
    )
    session._loaded_record = record
    session._loaded = True
    return session


class GraphArtifactRecordSession:


    def __init__(
        self,
        *,
        graph_artifact_dir: pathlib.Path,
        record_id: str,
        dataset_key: str,
        tokenizer_name: str,
        gzip_records: bool,
        access_mode: str = GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    ) -> None:
        self.graph_artifact_dir = pathlib.Path(graph_artifact_dir)
        self.record_id = record_id
        self.dataset_key = dataset_key
        self.tokenizer_name = tokenizer_name
        self.gzip_records = bool(gzip_records)
        self.access_mode = normalize_graph_artifact_cache_mode(access_mode)
        self._loaded_record: GraphArtifactRecord | None = None
        self._loaded = False
        self._staged_turns: dict[int, dict[str, GraphArtifactEntry]] = {}

    def _resolve_path(self) -> pathlib.Path:
        return resolve_record_path(
            self.graph_artifact_dir,
            self.record_id,
            gzip_records=self.gzip_records,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        path = self._resolve_path()
        if path.is_file():
            self._loaded_record = _read_record_file(path)
        else:
            self._loaded_record = None
        self._loaded = True

    def lookup(self, turn_idx: int, build_key: str) -> GraphArtifactEntry | None:

        staged_turn = self._staged_turns.get(turn_idx)
        if staged_turn is not None and build_key in staged_turn:
            return staged_turn[build_key]
        self._ensure_loaded()
        if self._loaded_record is None:
            return self._cache_miss(turn_idx, build_key)
        loaded_turn = self._loaded_record.turns.get(turn_idx)
        if loaded_turn is None:
            return self._cache_miss(turn_idx, build_key)
        hit = loaded_turn.artifacts.get(build_key)
        if hit is None:
            return self._cache_miss(turn_idx, build_key)
        return hit

    def _cache_miss(self, turn_idx: int, build_key: str) -> GraphArtifactEntry | None:
        if self.access_mode == GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY:
            raise GraphArtifactCacheMissError(
                "read_only graph artifact cache miss for "
                f"record_id={self.record_id!r}, turn_idx={turn_idx}, build_key={build_key}"
            )
        return None

    def stage(self, turn_idx: int, entry: GraphArtifactEntry) -> None:

        if self.access_mode == GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY:
            raise GraphArtifactCacheMissError(
                "read_only graph artifact cache cannot stage graph artifact for "
                f"record_id={self.record_id!r}, turn_idx={turn_idx}, "
                f"build_key={entry.build_key}"
            )
        bucket = self._staged_turns.setdefault(turn_idx, {})
        bucket[entry.build_key] = entry

    @property
    def has_staged_changes(self) -> bool:
        return bool(self._staged_turns)

    def flush(self) -> pathlib.Path | None:

        if not self._staged_turns:
            return None

        path = self._resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            base = _read_record_file(path)
        else:
            base = None

        merged = _merge_records(
            base=base,
            record_id=self.record_id,
            dataset_key=self.dataset_key,
            tokenizer_name=self.tokenizer_name,
            staged_turns=self._staged_turns,
        )
        _atomic_write_record(merged, path)

        self._loaded_record = merged
        self._loaded = True
        self._staged_turns = {}
        return path
