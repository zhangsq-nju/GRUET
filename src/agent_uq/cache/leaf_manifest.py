from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class LeafDiagnostics:


    mean_reject_rate: float
    mean_attempts_used: float
    episodes_exhausted_retry_budget: int


@dataclass
class LeafManifest:


    leaf: dict[str, str]
    sampling: dict[str, Any]
    target_record_ids: list[str]
    completed_record_ids: list[str]
    raw_count: int
    diagnostics: LeafDiagnostics
    completed_at: str
    status: str = "complete"
    runtime: dict[str, Any] = field(default_factory=dict)
    finalization: dict[str, Any] = field(default_factory=dict)


_FLOAT_PRECISION = 12
_RUNTIME_ONLY_SAMPLING_KEYS = frozenset({"tensor_parallel_size"})
_AGENT_UQ_MODEL_IDENTITY_KEYS = frozenset(
    {
        "model_path",
        "num_samples",
        "max_turns",
        "rejection",
    }
)
_LIGHTWEIGHT_SAMPLING_CACHE_FIELDS = (
    "model_path",
    "top_k_logprobs",
    "top_k",
    "num_samples",
    "max_turns",
    "num_shots",
    "temperature",
    "top_p",
    "max_new_tokens",
    "stop_sequences",
    "seed",
)
_LIGHTWEIGHT_REJECTION_CACHE_FIELDS = (
    "enabled",
    "max_retry_candidates_per_turn",
    "max_retry_candidates_per_episode",
)
_MISSING = object()


def _normalise(value: Any) -> Any:

    if isinstance(value, float):
        if math.isnan(value):


            return "__nan__"
        if math.isinf(value):
            return "__inf__" if value > 0 else "__neg_inf__"
        return round(value, _FLOAT_PRECISION)
    if isinstance(value, Mapping):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    return value


def _coerce_sampling_and_runtime(
    sampling: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:

    canonical_sampling = _normalise(dict(sampling))
    runtime_payload = _normalise(dict(runtime or {}))

    runtime_sampling_raw = runtime_payload.get("sampling") or {}
    if not isinstance(runtime_sampling_raw, Mapping):
        raise TypeError(
            "leaf manifest runtime.sampling must be a mapping; "
            f"got {type(runtime_sampling_raw).__name__}"
        )
    runtime_sampling = dict(runtime_sampling_raw)

    for key in _RUNTIME_ONLY_SAMPLING_KEYS:
        legacy_value = canonical_sampling.pop(key, None)
        runtime_value = runtime_sampling.get(key)
        if legacy_value is None:
            continue
        if runtime_value is not None and runtime_value != legacy_value:
            raise ValueError(
                "leaf manifest runtime metadata conflict for "
                f"{key!r}: sampling has {legacy_value!r}, runtime has {runtime_value!r}"
            )
        runtime_sampling[key] = legacy_value

    if runtime_sampling:
        runtime_payload["sampling"] = _normalise(runtime_sampling)
    else:
        runtime_payload.pop("sampling", None)
    return canonical_sampling, runtime_payload


def _looks_like_runtime_model_locator(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return "/" in value or "\\" in value


def _is_agent_uq_sampling_manifest(sampling: Mapping[str, Any]) -> bool:
    return _AGENT_UQ_MODEL_IDENTITY_KEYS.issubset(sampling.keys())


def canonical_sampling_cache_view(
    sampling: Mapping[str, Any],
    *,
    leaf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:

    canonical_sampling, _ = _coerce_sampling_and_runtime(sampling)
    leaf_model = None if leaf is None else leaf.get("model")
    if (
        isinstance(leaf_model, str)
        and leaf_model
        and _is_agent_uq_sampling_manifest(canonical_sampling)
        and _looks_like_runtime_model_locator(canonical_sampling.get("model_path"))
    ):
        canonical_sampling = dict(canonical_sampling)
        canonical_sampling["model_path"] = leaf_model
    return canonical_sampling


def sampling_subobject_deep_equal(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    left_leaf: Mapping[str, Any] | None = None,
    right_leaf: Mapping[str, Any] | None = None,
) -> bool:

    return canonical_sampling_cache_view(
        a,
        leaf=left_leaf,
    ) == canonical_sampling_cache_view(
        b,
        leaf=right_leaf,
    )


def _format_cache_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    return repr(value)


def _lightweight_sampling_cache_values(
    sampling: Mapping[str, Any],
    *,
    leaf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = canonical_sampling_cache_view(sampling, leaf=leaf)
    out: dict[str, Any] = {
        key: canonical.get(key, _MISSING)
        for key in _LIGHTWEIGHT_SAMPLING_CACHE_FIELDS
    }
    rejection = canonical.get("rejection")
    if isinstance(rejection, Mapping):
        out["rejection"] = {
            key: rejection.get(key, _MISSING)
            for key in _LIGHTWEIGHT_REJECTION_CACHE_FIELDS
        }
    else:
        out["rejection"] = {
            key: _MISSING
            for key in _LIGHTWEIGHT_REJECTION_CACHE_FIELDS
        }
    return out


def lightweight_sampling_cache_validation_mismatches(
    existing_sampling: Mapping[str, Any],
    effective_sampling: Mapping[str, Any],
    *,
    existing_leaf: Mapping[str, Any] | None = None,
    effective_leaf: Mapping[str, Any] | None = None,
) -> list[str]:

    existing = _lightweight_sampling_cache_values(
        existing_sampling,
        leaf=existing_leaf,
    )
    effective = _lightweight_sampling_cache_values(
        effective_sampling,
        leaf=effective_leaf,
    )
    mismatches: list[str] = []
    for key in _LIGHTWEIGHT_SAMPLING_CACHE_FIELDS:
        if existing[key] != effective[key]:
            mismatches.append(
                f"{key}: cached={_format_cache_value(existing[key])}, "
                f"requested={_format_cache_value(effective[key])}"
            )
    existing_rejection = existing["rejection"]
    effective_rejection = effective["rejection"]
    for key in _LIGHTWEIGHT_REJECTION_CACHE_FIELDS:
        if existing_rejection[key] != effective_rejection[key]:
            mismatches.append(
                f"rejection.{key}: cached={_format_cache_value(existing_rejection[key])}, "
                f"requested={_format_cache_value(effective_rejection[key])}"
            )
    return mismatches


_MANIFEST_FILENAME = "manifest.json"


def manifest_path(output_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(output_dir) / _MANIFEST_FILENAME


def write_leaf_manifest(
    output_dir: str | pathlib.Path,
    *,
    leaf: dict[str, str],
    sampling: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    target_record_ids: list[str],
    completed_record_ids: list[str] | None = None,
    raw_count: int,
    diagnostics: LeafDiagnostics,
    completed_at: str,
    status: str = "complete",
    finalization: Mapping[str, Any] | None = None,
) -> pathlib.Path:

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / _MANIFEST_FILENAME
    tmp_path = out_dir / f"{_MANIFEST_FILENAME}.tmp"
    canonical_sampling, runtime_payload = _coerce_sampling_and_runtime(
        sampling,
        runtime,
    )

    completed_ids = (
        list(target_record_ids[: int(raw_count)])
        if completed_record_ids is None
        else list(completed_record_ids)
    )
    payload = {
        "leaf": dict(leaf),
        "status": str(status),
        "sampling": canonical_sampling,
        "target_record_ids": list(target_record_ids),
        "completed_record_ids": completed_ids,
        "raw_count": int(raw_count),
        "diagnostics": {
            "mean_reject_rate": float(diagnostics.mean_reject_rate),
            "mean_attempts_used": float(diagnostics.mean_attempts_used),
            "episodes_exhausted_retry_budget": int(
                diagnostics.episodes_exhausted_retry_budget
            ),
        },
        "completed_at": str(completed_at),
        "finalization": _normalise(dict(finalization or {})),
    }
    if runtime_payload:
        payload["runtime"] = runtime_payload

    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    os.replace(tmp_path, final_path)
    return final_path


def read_leaf_manifest(output_dir: str | pathlib.Path) -> LeafManifest | None:

    p = manifest_path(output_dir)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    sampling, runtime = _coerce_sampling_and_runtime(
        dict(data.get("sampling") or {}),
        data.get("runtime"),
    )
    target_record_ids = list(data.get("target_record_ids") or [])
    raw_count = int(data.get("raw_count", 0))
    completed_record_ids = data.get("completed_record_ids")
    if completed_record_ids is None:
        completed_record_ids = target_record_ids[:raw_count]
    diag = data.get("diagnostics") or {}
    return LeafManifest(
        leaf=dict(data.get("leaf") or {}),
        sampling=sampling,
        target_record_ids=target_record_ids,
        completed_record_ids=list(completed_record_ids or []),
        raw_count=raw_count,
        diagnostics=LeafDiagnostics(
            mean_reject_rate=float(diag.get("mean_reject_rate", 0.0)),
            mean_attempts_used=float(diag.get("mean_attempts_used", 0.0)),
            episodes_exhausted_retry_budget=int(
                diag.get("episodes_exhausted_retry_budget", 0)
            ),
        ),
        completed_at=str(data.get("completed_at", "")),
        status=str(data.get("status", "complete")),
        runtime=runtime,
        finalization=dict(data.get("finalization") or {}),
    )


def target_ids_complete_on_disk(
    raw_dir: str | pathlib.Path, target_record_ids: list[str]
) -> tuple[bool, list[str]]:

    raw_dir = pathlib.Path(raw_dir)
    if not raw_dir.is_dir():

        return False, list(target_record_ids)
    missing: list[str] = []
    for rid in target_record_ids:
        plain = raw_dir / f"{rid}.json"
        gz = raw_dir / f"{rid}.json.gz"
        if not plain.is_file() and not gz.is_file():
            missing.append(rid)
    return len(missing) == 0, missing


def aggregate_diagnostics_from_raws(raws: list) -> LeafDiagnostics:

    sum_reject = 0
    sum_attempts = 0
    n_turns = 0
    exhausted = 0
    for raw in raws:
        for t in getattr(raw, "turns", []) or []:
            sum_reject += int(getattr(t, "reject_count", 0) or 0)
            sum_attempts += int(getattr(t, "attempts_used", 0) or 0)
            n_turns += 1
        if bool(getattr(raw, "episode_exhausted_retry_budget", False)):
            exhausted += 1
    mean_reject_rate = (sum_reject / sum_attempts) if sum_attempts > 0 else 0.0
    mean_attempts_used = (sum_attempts / n_turns) if n_turns > 0 else 0.0
    return LeafDiagnostics(
        mean_reject_rate=float(mean_reject_rate),
        mean_attempts_used=float(mean_attempts_used),
        episodes_exhausted_retry_budget=int(exhausted),
    )
