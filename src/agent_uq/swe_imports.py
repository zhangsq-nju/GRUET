from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Mapping, Sequence

ACTIVE_IMPORT_FILENAME = "active_import.json"


@dataclass(frozen=True)
class ActiveImportMetadata:
    swe_run_name: str
    semantic_launch_key: str
    leaf_path: str
    raw_turn_mode: str
    target_record_ids: tuple[str, ...]
    raw_count: int
    completed_at: str


def active_import_metadata_path(import_root: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(import_root) / ACTIVE_IMPORT_FILENAME


def _valid_import_leaf(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / "raw").is_dir()


def _candidate_import_leaves(import_root: pathlib.Path) -> list[pathlib.Path]:
    if not import_root.is_dir():
        return []
    return sorted(
        child
        for child in import_root.iterdir()
        if child.is_dir() and _valid_import_leaf(child)
    )


def _read_active_import_leaf(import_root: pathlib.Path) -> pathlib.Path | None:
    metadata_path = active_import_metadata_path(import_root)
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    leaf_path = payload.get("leaf_path")
    if not isinstance(leaf_path, str) or not leaf_path:
        raise ValueError(f"{metadata_path} must define non-empty leaf_path")
    candidate = pathlib.Path(leaf_path)
    if candidate.is_absolute():
        raise ValueError(f"{metadata_path} leaf_path must be relative to the import root")
    resolved = import_root / candidate
    if not _valid_import_leaf(resolved):
        raise FileNotFoundError(
            f"{metadata_path} points to {resolved}, but that directory does not contain raw/"
        )
    return resolved


def resolve_swe_import_leaf(
    path_value: str | pathlib.Path,
    *,
    base_dir: str | pathlib.Path,
    field_name: str = "run_dir",
) -> pathlib.Path:


    if path_value in (None, ""):
        raise ValueError(f"{field_name} must be set")
    path = pathlib.Path(str(path_value))
    if not path.is_absolute():
        path = pathlib.Path(base_dir) / path
    if not path.is_dir():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if _valid_import_leaf(path):
        return path

    active_leaf = _read_active_import_leaf(path)
    if active_leaf is not None:
        return active_leaf

    candidates = _candidate_import_leaves(path)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"{field_name} must contain raw/ or be an import root "
            f"with one imported leaf: {path}"
        )
    details = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise ValueError(
        f"{field_name} import root is ambiguous: {path}\n"
        f"Candidate imported leaves:\n{details}"
    )


def write_active_import_metadata(
    *,
    import_root: str | pathlib.Path,
    import_leaf: str | pathlib.Path,
    swe_run_name: str,
    semantic_launch_key: str,
    raw_turn_mode: str,
    target_record_ids: Sequence[str],
    raw_count: int,
    completed_at: str,
) -> pathlib.Path:
    import_root = pathlib.Path(import_root)
    import_leaf = pathlib.Path(import_leaf)
    leaf_path = import_leaf.relative_to(import_root).as_posix()
    payload = {
        "swe_run_name": str(swe_run_name),
        "semantic_launch_key": str(semantic_launch_key),
        "leaf_path": leaf_path,
        "raw_turn_mode": str(raw_turn_mode),
        "target_record_ids": [str(record_id) for record_id in target_record_ids],
        "raw_count": int(raw_count),
        "completed_at": str(completed_at),
    }
    import_root.mkdir(parents=True, exist_ok=True)
    path = active_import_metadata_path(import_root)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "ACTIVE_IMPORT_FILENAME",
    "ActiveImportMetadata",
    "active_import_metadata_path",
    "resolve_swe_import_leaf",
    "write_active_import_metadata",
]
