from __future__ import annotations

import datetime
import hashlib
import pathlib
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_SWE_AGENT_CONFIG_ROOT = REPO_ROOT / "vendor" / "SWE-agent"
SWE_CANDIDATE_TRAJECTORY_FORMAT = "candidate_v1"
SUPPORTED_SWE_PARSER_MODE = "thought_action"
SWE_SELECTION_POLICY = "first_compliant"
SWE_RAW_TURN_MODE_SELECTED_ONLY = "selected_only"
SWE_RAW_TURN_MODE_ALL_TRAJECTORY = "all_trajectory"
SUPPORTED_SWE_RAW_TURN_MODES = frozenset(
    {SWE_RAW_TURN_MODE_SELECTED_ONLY, SWE_RAW_TURN_MODE_ALL_TRAJECTORY}
)


def repo_relative_path(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: str | pathlib.Path) -> str:
    payload = pathlib.Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def require_path(value: Any, *, field_name: str) -> pathlib.Path:
    if value in (None, ""):
        raise ValueError(f"{field_name} must be set")
    return pathlib.Path(value)


def validate_swe_raw_turn_mode(value: Any) -> str:
    raw_turn_mode = str(value or "")
    if raw_turn_mode not in SUPPORTED_SWE_RAW_TURN_MODES:
        supported = ", ".join(sorted(SUPPORTED_SWE_RAW_TURN_MODES))
        raise ValueError(
            f"unsupported import.raw_turn_mode: {raw_turn_mode!r}; "
            f"expected one of: {supported}"
        )
    return raw_turn_mode


def optional_path(value: Any) -> pathlib.Path | None:
    if value in (None, ""):
        return None
    return pathlib.Path(value)


def ensure_runtime_dir(output_dir: str | pathlib.Path) -> pathlib.Path:
    runtime_dir = pathlib.Path(output_dir) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def utc_path_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
