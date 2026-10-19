from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping


ARTIFACT_SCHEMA_VERSION = 1

_REQUIRED_TOP_LEVEL_FIELDS = (
    "record_id",
    "dataset_key",
    "artifact_schema_version",
    "turns",
)

_REQUIRED_ARTIFACT_FIELDS = (
    "build_key",
    "build_params",
    "valid_candidate_indices",
    "sample_texts",
    "entailment_matrix",
    "neutral_matrix",
    "contradiction_matrix",
    "n_valid_candidates",
)

_REQUIRED_BUILD_PARAM_FIELDS = ("nli_model_name", "nli_execution_mode")


class PairwiseNLIArtifactError(Exception):
    pass


class PairwiseNLIArtifactDecodeError(PairwiseNLIArtifactError):
    pass


class PairwiseNLIArtifactSchemaError(PairwiseNLIArtifactError):
    pass


class PairwiseNLIArtifactPathAmbiguityError(PairwiseNLIArtifactError):
    pass


class PairwiseNLIArtifactWriteError(PairwiseNLIArtifactError):
    pass


@dataclass
class PairwiseNLIArtifactEntry:
    build_key: str
    build_params: dict[str, Any]
    valid_candidate_indices: list[int]
    sample_texts: list[str]
    entailment_matrix: list[list[float]]
    neutral_matrix: list[list[float]]
    contradiction_matrix: list[list[float]]
    n_valid_candidates: int


@dataclass
class PairwiseNLIArtifactTurn:
    turn_idx: int
    artifacts: dict[str, PairwiseNLIArtifactEntry] = field(default_factory=dict)


@dataclass
class PairwiseNLIArtifactRecord:
    record_id: str
    dataset_key: str
    artifact_schema_version: int
    turns: dict[int, PairwiseNLIArtifactTurn] = field(default_factory=dict)


def _ensure_dict(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: expected dict at {where}, got {type(value).__name__}"
        )
    return value


def _ensure_list(value: Any, *, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: expected list at {where}, got {type(value).__name__}"
        )
    return value


def _ensure_string_list(
    value: Any,
    *,
    where: str,
    expected_len: int,
) -> list[str]:
    items = _ensure_list(value, where=where)
    if len(items) != expected_len:
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {where} must have length {expected_len}, "
            f"got {len(items)}"
        )
    if not all(isinstance(item, str) for item in items):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {where} must contain strings"
        )
    return [str(item) for item in items]


def _ensure_int_list(
    value: Any,
    *,
    where: str,
    expected_len: int,
) -> list[int]:
    items = _ensure_list(value, where=where)
    if len(items) != expected_len:
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {where} must have length {expected_len}, "
            f"got {len(items)}"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {where} must contain ints"
        )
    return [int(item) for item in items]


def _ensure_square_numeric_matrix(
    value: Any,
    *,
    where: str,
    expected_size: int,
) -> list[list[float]]:
    rows = _ensure_list(value, where=where)
    if len(rows) != expected_size:
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {where} must have {expected_size} rows, "
            f"got {len(rows)}"
        )
    out: list[list[float]] = []
    for row_idx, raw_row in enumerate(rows):
        row = _ensure_list(raw_row, where=f"{where}[{row_idx}]")
        if len(row) != expected_size:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {where}[{row_idx}] must have "
                f"{expected_size} columns, got {len(row)}"
            )
        converted_row: list[float] = []
        for col_idx, value_at_cell in enumerate(row):
            if isinstance(value_at_cell, bool) or not isinstance(value_at_cell, (int, float)):
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {where}[{row_idx}][{col_idx}] "
                    f"must be numeric"
                )
            converted_row.append(float(value_at_cell))
        out.append(converted_row)
    return out


def _entry_to_jsonable(entry: PairwiseNLIArtifactEntry) -> dict[str, Any]:
    return {
        "build_key": entry.build_key,
        "build_params": dict(entry.build_params),
        "valid_candidate_indices": list(entry.valid_candidate_indices),
        "sample_texts": list(entry.sample_texts),
        "entailment_matrix": [list(row) for row in entry.entailment_matrix],
        "neutral_matrix": [list(row) for row in entry.neutral_matrix],
        "contradiction_matrix": [list(row) for row in entry.contradiction_matrix],
        "n_valid_candidates": int(entry.n_valid_candidates),
    }


def record_to_jsonable(record: PairwiseNLIArtifactRecord) -> dict[str, Any]:
    turns_out: list[dict[str, Any]] = []
    for turn_idx in sorted(record.turns):
        turn = record.turns[turn_idx]
        artifacts_out: dict[str, dict[str, Any]] = {}
        for build_key in sorted(turn.artifacts):
            artifacts_out[build_key] = _entry_to_jsonable(turn.artifacts[build_key])
        turns_out.append({"turn_idx": int(turn_idx), "artifacts": artifacts_out})
    return {
        "record_id": record.record_id,
        "dataset_key": record.dataset_key,
        "artifact_schema_version": int(record.artifact_schema_version),
        "turns": turns_out,
    }


def record_from_jsonable(
    payload: Any,
    *,
    source: str = "<memory>",
) -> PairwiseNLIArtifactRecord:
    data = _ensure_dict(payload, where=f"{source} (top level)")
    for key in _REQUIRED_TOP_LEVEL_FIELDS:
        if key not in data:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source} is missing required "
                f"top-level field {key!r}"
            )

    record_id = data["record_id"]
    dataset_key = data["dataset_key"]
    artifact_schema_version = data["artifact_schema_version"]
    if not isinstance(record_id, str) or not record_id:
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {source}.record_id must be a non-empty string"
        )
    if not isinstance(dataset_key, str):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {source}.dataset_key must be a string"
        )
    if not isinstance(artifact_schema_version, int) or isinstance(artifact_schema_version, bool):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {source}.artifact_schema_version must be an int"
        )

    turns_payload = data["turns"]
    turns_out: dict[int, PairwiseNLIArtifactTurn] = {}
    if isinstance(turns_payload, list):
        for i, raw_turn in enumerate(turns_payload):
            turn_dict = _ensure_dict(raw_turn, where=f"{source}.turns[{i}]")
            if "turn_idx" not in turn_dict:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.turns[{i}] missing 'turn_idx'"
                )
            if "artifacts" not in turn_dict:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.turns[{i}] missing 'artifacts'"
                )
            turn = _read_turn(turn_dict, source=f"{source}.turns[{i}]")
            turns_out[turn.turn_idx] = turn
    elif isinstance(turns_payload, dict):
        for key, raw_turn in turns_payload.items():
            try:
                turn_idx = int(key)
            except (TypeError, ValueError) as exc:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.turns has non-int key {key!r}"
                ) from exc
            turn_dict = _ensure_dict(raw_turn, where=f"{source}.turns[{key}]")
            if "artifacts" not in turn_dict:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.turns[{key}] missing 'artifacts'"
                )
            turn_dict.setdefault("turn_idx", turn_idx)
            turn = _read_turn(turn_dict, source=f"{source}.turns[{key}]")
            turns_out[turn.turn_idx] = turn
    else:
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {source}.turns must be a list or dict, "
            f"got {type(turns_payload).__name__}"
        )

    return PairwiseNLIArtifactRecord(
        record_id=record_id,
        dataset_key=dataset_key,
        artifact_schema_version=int(artifact_schema_version),
        turns=turns_out,
    )


def _read_turn(
    turn_dict: Mapping[str, Any],
    *,
    source: str,
) -> PairwiseNLIArtifactTurn:
    raw_turn_idx = turn_dict["turn_idx"]
    if not isinstance(raw_turn_idx, int) or isinstance(raw_turn_idx, bool):
        raise PairwiseNLIArtifactSchemaError(
            f"pairwise-nli-artifact schema: {source}.turn_idx must be int, "
            f"got {type(raw_turn_idx).__name__}"
        )
    artifacts_payload = _ensure_dict(turn_dict["artifacts"], where=f"{source}.artifacts")
    artifacts_out: dict[str, PairwiseNLIArtifactEntry] = {}
    for build_key, raw_entry in artifacts_payload.items():
        if not isinstance(build_key, str) or not build_key:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts has non-string "
                f"build_key {build_key!r}"
            )
        entry_dict = _ensure_dict(raw_entry, where=f"{source}.artifacts[{build_key}]")
        for required in _REQUIRED_ARTIFACT_FIELDS:
            if required not in entry_dict:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}] "
                    f"missing required field {required!r}"
                )
        if entry_dict["build_key"] != build_key:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}].build_key "
                f"({entry_dict['build_key']!r}) does not match its own key"
            )
        build_params = _ensure_dict(
            entry_dict["build_params"],
            where=f"{source}.artifacts[{build_key}].build_params",
        )
        for required in _REQUIRED_BUILD_PARAM_FIELDS:
            if required not in build_params:
                raise PairwiseNLIArtifactSchemaError(
                    f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}]."
                    f"build_params missing required field {required!r}"
                )
        if not isinstance(build_params["nli_model_name"], str):
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}]."
                "build_params.nli_model_name must be a string"
            )
        if not isinstance(build_params["nli_execution_mode"], str):
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}]."
                "build_params.nli_execution_mode must be a string"
            )
        n_valid_candidates = entry_dict["n_valid_candidates"]
        if not isinstance(n_valid_candidates, int) or isinstance(n_valid_candidates, bool):
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}]."
                "n_valid_candidates must be an int"
            )
        if n_valid_candidates < 0:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact schema: {source}.artifacts[{build_key}]."
                "n_valid_candidates must be >= 0"
            )
        valid_candidate_indices = _ensure_int_list(
            entry_dict["valid_candidate_indices"],
            where=f"{source}.artifacts[{build_key}].valid_candidate_indices",
            expected_len=n_valid_candidates,
        )
        sample_texts = _ensure_string_list(
            entry_dict["sample_texts"],
            where=f"{source}.artifacts[{build_key}].sample_texts",
            expected_len=n_valid_candidates,
        )
        entailment_matrix = _ensure_square_numeric_matrix(
            entry_dict["entailment_matrix"],
            where=f"{source}.artifacts[{build_key}].entailment_matrix",
            expected_size=n_valid_candidates,
        )
        neutral_matrix = _ensure_square_numeric_matrix(
            entry_dict["neutral_matrix"],
            where=f"{source}.artifacts[{build_key}].neutral_matrix",
            expected_size=n_valid_candidates,
        )
        contradiction_matrix = _ensure_square_numeric_matrix(
            entry_dict["contradiction_matrix"],
            where=f"{source}.artifacts[{build_key}].contradiction_matrix",
            expected_size=n_valid_candidates,
        )
        artifacts_out[build_key] = PairwiseNLIArtifactEntry(
            build_key=build_key,
            build_params=dict(build_params),
            valid_candidate_indices=valid_candidate_indices,
            sample_texts=sample_texts,
            entailment_matrix=entailment_matrix,
            neutral_matrix=neutral_matrix,
            contradiction_matrix=contradiction_matrix,
            n_valid_candidates=int(n_valid_candidates),
        )
    return PairwiseNLIArtifactTurn(turn_idx=int(raw_turn_idx), artifacts=artifacts_out)


def derive_nli_execution_mode(resolved_device: str | None) -> str:


    device = str(resolved_device or "cpu").strip().lower()
    if device.startswith("cuda"):
        return "cuda-fp16"
    if not device or device.startswith("cpu"):
        return "cpu-fp32"
    return f"{device}-fp32"


def derive_build_key(
    *,
    sample_texts: list[str],
    nli_model_name: str,
    nli_execution_mode: str,
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION,
) -> str:
    payload = {
        "artifact_schema_version": int(artifact_schema_version),
        "sample_texts": list(sample_texts),
        "nli_model_name": str(nli_model_name),
        "nli_execution_mode": str(nli_execution_mode),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_record_path(
    pairwise_nli_artifact_dir: pathlib.Path,
    record_id: str,
    *,
    gzip_records: bool,
) -> pathlib.Path:
    json_path = pairwise_nli_artifact_dir / f"{record_id}.json"
    gz_path = pairwise_nli_artifact_dir / f"{record_id}.json.gz"
    json_exists = json_path.is_file()
    gz_exists = gz_path.is_file()

    if json_exists and gz_exists:
        raise PairwiseNLIArtifactPathAmbiguityError(
            f"pairwise-nli-artifact path ambiguity: both {json_path} and {gz_path} "
            "exist; leaf is in mixed-format limbo and must be cleaned up by an operator"
        )
    if gzip_records:
        if json_exists and not gz_exists:
            raise PairwiseNLIArtifactPathAmbiguityError(
                f"pairwise-nli-artifact path ambiguity: scorer policy requests gzip but "
                f"existing artifact at {json_path} is uncompressed"
            )
        return gz_path
    if gz_exists and not json_exists:
        raise PairwiseNLIArtifactPathAmbiguityError(
            "pairwise-nli-artifact path ambiguity: scorer policy requests uncompressed "
            f"but existing artifact at {gz_path} is gzipped"
        )
    return json_path


def _open_for_read(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _read_record_file(path: pathlib.Path) -> PairwiseNLIArtifactRecord:
    try:
        with _open_for_read(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PairwiseNLIArtifactDecodeError(
            f"pairwise-nli-artifact decode failed for {path}: {exc}"
        ) from exc
    return record_from_jsonable(data, source=str(path))


def _atomic_write_record(record: PairwiseNLIArtifactRecord, path: pathlib.Path) -> None:
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
        raise PairwiseNLIArtifactWriteError(
            f"pairwise-nli-artifact write failed for {path}: {exc}"
        ) from exc


def _merge_records(
    *,
    base: PairwiseNLIArtifactRecord | None,
    record_id: str,
    dataset_key: str,
    staged_turns: dict[int, dict[str, PairwiseNLIArtifactEntry]],
) -> PairwiseNLIArtifactRecord:
    if base is None:
        merged = PairwiseNLIArtifactRecord(
            record_id=record_id,
            dataset_key=dataset_key,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        )
    else:
        if base.record_id != record_id:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact merge: base.record_id={base.record_id!r} "
                f"does not match expected {record_id!r}"
            )
        if base.dataset_key != dataset_key:
            raise PairwiseNLIArtifactSchemaError(
                f"pairwise-nli-artifact merge: base.dataset_key={base.dataset_key!r} "
                f"does not match expected {dataset_key!r}"
            )
        merged = PairwiseNLIArtifactRecord(
            record_id=base.record_id,
            dataset_key=base.dataset_key,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            turns={
                turn_idx: PairwiseNLIArtifactTurn(
                    turn_idx=turn_idx,
                    artifacts=dict(turn.artifacts),
                )
                for turn_idx, turn in base.turns.items()
            },
        )

    for turn_idx, entries in staged_turns.items():
        existing_turn = merged.turns.get(turn_idx)
        if existing_turn is None:
            existing_turn = PairwiseNLIArtifactTurn(turn_idx=turn_idx, artifacts={})
            merged.turns[turn_idx] = existing_turn
        for build_key, entry in entries.items():
            existing_turn.artifacts[build_key] = entry
    return merged


class PairwiseNLIArtifactRecordSession:


    def __init__(
        self,
        *,
        pairwise_nli_artifact_dir: pathlib.Path,
        record_id: str,
        dataset_key: str,
        gzip_records: bool,
    ) -> None:
        self.pairwise_nli_artifact_dir = pathlib.Path(pairwise_nli_artifact_dir)
        self.record_id = record_id
        self.dataset_key = dataset_key
        self.gzip_records = bool(gzip_records)
        self._loaded_record: PairwiseNLIArtifactRecord | None = None
        self._loaded = False
        self._staged_turns: dict[int, dict[str, PairwiseNLIArtifactEntry]] = {}

    def _resolve_path(self) -> pathlib.Path:
        return resolve_record_path(
            self.pairwise_nli_artifact_dir,
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

    def lookup(self, turn_idx: int, build_key: str) -> PairwiseNLIArtifactEntry | None:
        staged_turn = self._staged_turns.get(turn_idx)
        if staged_turn is not None and build_key in staged_turn:
            return staged_turn[build_key]
        self._ensure_loaded()
        if self._loaded_record is None:
            return None
        loaded_turn = self._loaded_record.turns.get(turn_idx)
        if loaded_turn is None:
            return None
        return loaded_turn.artifacts.get(build_key)

    def stage(self, turn_idx: int, entry: PairwiseNLIArtifactEntry) -> None:
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

        base: PairwiseNLIArtifactRecord | None
        if path.is_file():
            base = _read_record_file(path)
        else:
            base = None

        merged = _merge_records(
            base=base,
            record_id=self.record_id,
            dataset_key=self.dataset_key,
            staged_turns=self._staged_turns,
        )
        _atomic_write_record(merged, path)
        self._loaded_record = merged
        self._loaded = True
        self._staged_turns = {}
        return path
