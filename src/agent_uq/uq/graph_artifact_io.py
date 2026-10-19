from __future__ import annotations

import gzip
import json
import os
import pathlib
import tempfile

from agent_uq.uq.graph_artifact_codec import record_from_jsonable, record_to_jsonable
from agent_uq.uq.graph_artifact_types import (
    ARTIFACT_SCHEMA_VERSION,
    GraphArtifactDecodeError,
    GraphArtifactEntry,
    GraphArtifactRecord,
    GraphArtifactSchemaError,
    GraphArtifactTurn,
    GraphArtifactWriteError,
)


def _open_for_read(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _read_record_file(path: pathlib.Path) -> GraphArtifactRecord:
    try:
        with _open_for_read(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphArtifactDecodeError(
            f"graph-artifact decode failed for {path}: {exc}"
        ) from exc
    return record_from_jsonable(data, source=str(path))


def _atomic_write_record(record: GraphArtifactRecord, path: pathlib.Path) -> None:

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = record_to_jsonable(record)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = pathlib.Path(tmp_name)
    try:
        os.close(fd)
        if path.suffix == ".gz":
            with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        else:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise GraphArtifactWriteError(
            f"graph-artifact write failed for {path}: {exc}"
        ) from exc


def _merge_records(
    *,
    base: GraphArtifactRecord | None,
    record_id: str,
    dataset_key: str,
    tokenizer_name: str,
    staged_turns: dict[int, dict[str, GraphArtifactEntry]],
) -> GraphArtifactRecord:

    if base is None:
        merged = GraphArtifactRecord(
            record_id=record_id,
            dataset_key=dataset_key,
            tokenizer_name=tokenizer_name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        )
    else:
        if base.record_id != record_id:
            raise GraphArtifactSchemaError(
                f"graph-artifact merge: base.record_id={base.record_id!r} "
                f"does not match expected {record_id!r}"
            )
        merged = GraphArtifactRecord(
            record_id=base.record_id,
            dataset_key=base.dataset_key,
            tokenizer_name=base.tokenizer_name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            turns={
                turn_idx: GraphArtifactTurn(
                    turn_idx=turn_idx,
                    artifacts=dict(turn.artifacts),
                )
                for turn_idx, turn in base.turns.items()
            },
        )

    for turn_idx, entries in staged_turns.items():
        existing_turn = merged.turns.get(turn_idx)
        if existing_turn is None:
            existing_turn = GraphArtifactTurn(turn_idx=turn_idx, artifacts={})
            merged.turns[turn_idx] = existing_turn
        for build_key, entry in entries.items():
            existing_turn.artifacts[build_key] = entry
    return merged
