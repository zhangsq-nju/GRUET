from __future__ import annotations

import dataclasses
import gzip
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from agent_uq.actions import normalize_action


SCORED_RECORD_FORMAT = "scored_compact_v1"
SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT = "swe_thought_action_codeblock_v1"
ALLOWED_TARGET_TYPES = frozenset({"binary", "continuous"})


@dataclass
class TopKEntryRec:
    token_id: int
    logprob: float


@dataclass
class CandidateRecord:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    top_k: list[list[TopKEntryRec]]
    finish_reason: str

    parse_status: str
    raw_response: str | None = None
    parse_error: str | None = None
    parsed_thought: str | None = None
    parsed_action: dict[str, Any] | None = None

    observation: str | None = None
    backend: dict[str, Any] = field(default_factory=dict)


    compliant: bool = True
    non_compliant_reason: str | None = None


@dataclass
class Turn:
    turn_idx: int
    prompt: str
    candidates: list[CandidateRecord]
    selected_idx: int | None
    selection_policy: str


    attempts_used: int = 0
    reject_count: int = 0
    compliant_count: int = 0


    uncertainty_estimates: dict[str, dict[str, Any]] | None = None
    state: dict[str, Any] | None = None
    tool_call_ids: list[str] | None = None
    step_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectionConfigRec:


    enabled: bool
    max_retry_candidates_per_turn: int
    max_retry_candidates_per_episode: int


@dataclass
class TrajectoryConfig:


    K: int
    M: int
    N: int
    policy: str
    seed: int
    estimator: str | None = None
    rejection: RejectionConfigRec | None = None


@dataclass
class RawTrajectory:
    record_id: str
    dataset_key: str
    target_type: str
    env: str
    model: str
    config: TrajectoryConfig
    tokenizer_name: str
    turns: list[Turn]
    final_reward: float
    solved: bool
    done_reason: str
    initial_state: dict[str, Any] | None = None


    episode_exhausted_retry_budget: bool = False
    env_backend: str = "docker"
    env_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredTurn:
    turn_idx: int
    uncertainty_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ScoredTrajectory:
    record_format: str
    record_id: str
    raw_path: str
    dataset_key: str
    target_type: str
    env: str
    model: str
    final_reward: float
    solved: bool
    done_reason: str
    env_backend: str = "docker"
    env_provenance: dict[str, Any] = field(default_factory=dict)
    estimators: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: list[ScoredTurn] = field(default_factory=list)


def _to_jsonable(obj: Any) -> Any:

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            out[f.name] = _to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def _open_for_write(path: pathlib.Path, gzip_records: bool):
    if gzip_records:
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _open_for_read(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _atomic_dump_json(path: pathlib.Path, payload: Any, *, gzip_records: bool) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with _open_for_write(tmp_path, gzip_records) as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _record_path(out_dir: pathlib.Path, record_id: str, gzip_records: bool) -> pathlib.Path:
    suffix = ".json.gz" if gzip_records else ".json"
    return out_dir / f"{record_id}{suffix}"


def _record_id_from_path(path: pathlib.Path) -> str:
    if path.name.endswith(".json.gz"):
        return path.name[:-8]
    if path.name.endswith(".json"):
        return path.name[:-5]
    raise ValueError(f"unsupported record filename: {path}")


def _validate_target_type(target_type: str, *, context: str) -> str:
    value = str(target_type)
    if value in ALLOWED_TARGET_TYPES:
        return value
    allowed = ", ".join(sorted(ALLOWED_TARGET_TYPES))
    if value == "patch":
        raise ValueError(
            f"{context} has unsupported target_type='patch'; target_type must be one of "
            f"{allowed}. Legacy SWE records with target_type='patch' must be regenerated."
        )
    raise ValueError(
        f"{context} has unsupported target_type={value!r}; "
        f"target_type must be one of {allowed}."
    )


def _format_duplicate_record_error(
    record_dir: pathlib.Path,
    duplicates: Mapping[str, list[pathlib.Path]],
) -> str:
    pieces = []
    for record_id, paths in sorted(duplicates.items()):
        names = ", ".join(sorted(p.name for p in paths))
        pieces.append(f"{record_id}: {names}")
    return (
        f"duplicate record files found under {record_dir}: " + "; ".join(pieces)
    )


def list_record_paths(
    record_dir: str | pathlib.Path,
    *,
    reject_duplicates: bool = False,
) -> list[pathlib.Path]:

    root = pathlib.Path(record_dir)
    paths = sorted(root.glob("*.json")) + sorted(root.glob("*.json.gz"))
    if not reject_duplicates:
        return paths

    grouped: dict[str, list[pathlib.Path]] = {}
    for path in paths:
        grouped.setdefault(_record_id_from_path(path), []).append(path)
    duplicates = {record_id: files for record_id, files in grouped.items() if len(files) > 1}
    if duplicates:
        raise ValueError(_format_duplicate_record_error(root, duplicates))
    return paths


def find_record_path(
    record_dir: str | pathlib.Path,
    record_id: str,
    *,
    prefer_gzip: bool | None = None,
    reject_duplicates: bool = False,
) -> pathlib.Path | None:

    root = pathlib.Path(record_dir)
    json_path = root / f"{record_id}.json"
    gzip_path = root / f"{record_id}.json.gz"

    existing = [path for path in (json_path, gzip_path) if path.is_file()]
    if not existing:
        return None
    if reject_duplicates and len(existing) > 1:
        raise ValueError(_format_duplicate_record_error(root, {record_id: existing}))
    if prefer_gzip is True and gzip_path in existing:
        return gzip_path
    if prefer_gzip is False and json_path in existing:
        return json_path
    return existing[0]


def _infer_raw_path(
    record_id: str,
    *,
    leaf_root: pathlib.Path | None,
    preferred_suffix: str = ".json",
) -> str:
    if leaf_root is not None:
        raw_dir = leaf_root / "raw"
        existing = list_record_paths(raw_dir, reject_duplicates=False) if raw_dir.is_dir() else []
        matches = [path for path in existing if _record_id_from_path(path) == record_id]
        if len(matches) == 1:
            return matches[0].relative_to(leaf_root).as_posix()
    return f"raw/{record_id}{preferred_suffix}"


def _normalize_raw_path(
    raw_path: str | pathlib.Path | None,
    *,
    record_id: str,
    leaf_root: pathlib.Path | None,
    preferred_suffix: str = ".json",
) -> str:
    if raw_path is None:
        return _infer_raw_path(
            record_id,
            leaf_root=leaf_root,
            preferred_suffix=preferred_suffix,
        )
    p = pathlib.Path(raw_path)
    if p.is_absolute() and leaf_root is not None:
        try:
            return p.relative_to(leaf_root).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def build_scored_shell(
    raw: RawTrajectory,
    *,
    raw_path: str | pathlib.Path | None = None,
    leaf_root: pathlib.Path | None = None,
) -> ScoredTrajectory:
    target_type = _validate_target_type(
        raw.target_type,
        context=f"raw record {raw.record_id!r}",
    )
    return ScoredTrajectory(
        record_format=SCORED_RECORD_FORMAT,
        record_id=raw.record_id,
        raw_path=_normalize_raw_path(
            raw_path,
            record_id=raw.record_id,
            leaf_root=leaf_root,
        ),
        dataset_key=raw.dataset_key,
        target_type=target_type,
        env=raw.env,
        model=raw.model,
        final_reward=float(raw.final_reward),
        solved=bool(raw.solved),
        done_reason=raw.done_reason,
        env_backend=raw.env_backend,
        env_provenance=dict(raw.env_provenance),
        estimators={},
        turns=[ScoredTurn(turn_idx=turn.turn_idx) for turn in raw.turns],
    )


def _normalize_compact_turn_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    out = {"scalar": float(entry["scalar"])}
    if "extras" in entry:
        extras = entry.get("extras")
        out["extras"] = {} if extras is None else _to_jsonable(extras)
    return out


def _normalize_legacy_turn_entry(entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    normalized = {"scalar": float(entry["scalar"])}
    if "extras" in entry:
        extras = entry.get("extras")
        normalized["extras"] = {} if extras is None else _to_jsonable(extras)
    params = entry.get("params")
    if params is None:
        return normalized, None
    return normalized, dict(params)


def _compact_from_legacy_raw(
    raw_like: RawTrajectory,
    *,
    raw_path: str | pathlib.Path | None = None,
    leaf_root: pathlib.Path | None = None,
    preferred_suffix: str = ".json",
) -> ScoredTrajectory:
    scored = build_scored_shell(
        raw_like,
        raw_path=_normalize_raw_path(
            raw_path,
            record_id=raw_like.record_id,
            leaf_root=leaf_root,
            preferred_suffix=preferred_suffix,
        ),
        leaf_root=leaf_root,
    )

    params_by_estimator: dict[str, dict[str, Any]] = {}
    estimator_params_missing: set[str] = set()
    estimator_params_inconsistent: set[str] = set()

    for raw_turn, compact_turn in zip(raw_like.turns, scored.turns):
        estimates = raw_turn.uncertainty_estimates or {}
        normalized_turn: dict[str, dict[str, Any]] = {}
        for estimator_name, entry in estimates.items():
            normalized_entry, params = _normalize_legacy_turn_entry(entry)
            normalized_turn[estimator_name] = normalized_entry

            if params is None:
                estimator_params_missing.add(estimator_name)
                continue
            if estimator_name not in params_by_estimator:
                params_by_estimator[estimator_name] = params
            elif params_by_estimator[estimator_name] != params:
                estimator_params_inconsistent.add(estimator_name)
        compact_turn.uncertainty_estimates = normalized_turn

    all_estimators = {
        estimator_name
        for turn in scored.turns
        for estimator_name in turn.uncertainty_estimates
    }
    for estimator_name in sorted(all_estimators):
        if estimator_name in estimator_params_missing:
            continue
        if estimator_name in estimator_params_inconsistent:
            continue
        params = params_by_estimator.get(estimator_name)
        if params is None:
            continue
        scored.estimators[estimator_name] = {"params": dict(params)}

    return scored


def _clone_scored(scored: ScoredTrajectory) -> ScoredTrajectory:
    target_type = _validate_target_type(
        scored.target_type,
        context=f"scored record {scored.record_id!r}",
    )
    return ScoredTrajectory(
        record_format=scored.record_format,
        record_id=scored.record_id,
        raw_path=scored.raw_path,
        dataset_key=scored.dataset_key,
        target_type=target_type,
        env=scored.env,
        model=scored.model,
        final_reward=float(scored.final_reward),
        solved=bool(scored.solved),
        done_reason=scored.done_reason,
        env_backend=scored.env_backend,
        env_provenance=dict(scored.env_provenance),
        estimators={
            name: dict(_to_jsonable(meta))
            for name, meta in scored.estimators.items()
        },
        turns=[
            ScoredTurn(
                turn_idx=turn.turn_idx,
                uncertainty_estimates={
                    name: dict(entry)
                    for name, entry in turn.uncertainty_estimates.items()
                },
            )
            for turn in scored.turns
        ],
    )


def write_raw(
    raw: RawTrajectory,
    out_dir: str | pathlib.Path,
    *,
    gzip_records: bool = False,
) -> pathlib.Path:

    _validate_target_type(raw.target_type, context=f"raw record {raw.record_id!r}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = _record_path(out, raw.record_id, gzip_records)
    payload = _to_jsonable(raw)
    _atomic_dump_json(path, payload, gzip_records=gzip_records)
    stale_path = _record_path(out, raw.record_id, not gzip_records)
    if stale_path != path and stale_path.exists():
        stale_path.unlink()
    return path


def write_scored(
    scored: ScoredTrajectory | RawTrajectory,
    out_dir: str | pathlib.Path,
    *,
    gzip_records: bool = False,
    raw_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaf_root = out.parent

    if isinstance(scored, RawTrajectory):
        compact = _compact_from_legacy_raw(
            scored,
            raw_path=raw_path,
            leaf_root=leaf_root,
            preferred_suffix=".json.gz" if gzip_records else ".json",
        )
    else:
        compact = _clone_scored(scored)
        if raw_path is not None:
            compact.raw_path = _normalize_raw_path(
                raw_path,
                record_id=compact.record_id,
                leaf_root=leaf_root,
            )
        elif not compact.raw_path:
            compact.raw_path = _infer_raw_path(
                compact.record_id,
                leaf_root=leaf_root,
                preferred_suffix=".json.gz" if gzip_records else ".json",
            )
        compact.record_format = SCORED_RECORD_FORMAT

    path = _record_path(out, compact.record_id, gzip_records)
    payload = _to_jsonable(compact)
    _atomic_dump_json(path, payload, gzip_records=gzip_records)

    stale_path = _record_path(out, compact.record_id, not gzip_records)
    if stale_path != path and stale_path.exists():
        stale_path.unlink()
    return path


def read_raw(path: str | pathlib.Path) -> RawTrajectory:
    p = pathlib.Path(path)
    with _open_for_read(p) as fh:
        data = json.load(fh)
    return _from_jsonable_raw(data)


def read_scored(path: str | pathlib.Path) -> ScoredTrajectory:
    p = pathlib.Path(path)
    with _open_for_read(p) as fh:
        data = json.load(fh)
    return _from_jsonable_scored(data, source_path=p)


def _from_jsonable_top_k(rows: list[list[dict[str, Any]]]) -> list[list[TopKEntryRec]]:
    return [
        [TopKEntryRec(token_id=int(e["token_id"]), logprob=float(e["logprob"])) for e in step]
        for step in rows
    ]


def _from_jsonable_candidate(d: dict[str, Any]) -> CandidateRecord:
    parse_status = d["parse_status"]
    legacy_compliant = parse_status == "ok"
    compliant = d.get("compliant", legacy_compliant)
    non_compliant_reason = d.get("non_compliant_reason")
    return CandidateRecord(
        text=d["text"],
        token_ids=list(d["token_ids"]),
        token_logprobs=[float(x) for x in d["token_logprobs"]],
        top_k=_from_jsonable_top_k(d["top_k"]),
        finish_reason=d["finish_reason"],
        raw_response=d.get("raw_response"),
        parse_status=parse_status,
        parse_error=d.get("parse_error"),
        parsed_thought=d.get("parsed_thought"),
        parsed_action=normalize_action(d.get("parsed_action")),
        observation=d.get("observation"),
        backend=dict(d.get("backend") or {}),
        compliant=bool(compliant),
        non_compliant_reason=non_compliant_reason,
    )


def _from_jsonable_turn(d: dict[str, Any]) -> Turn:
    candidates = [_from_jsonable_candidate(c) for c in d["candidates"]]
    legacy_compliant_count = sum(1 for c in candidates if c.compliant)
    attempts_used = int(d.get("attempts_used", len(candidates)))
    compliant_count = int(d.get("compliant_count", legacy_compliant_count))
    reject_count = int(d.get("reject_count", attempts_used - compliant_count))
    return Turn(
        turn_idx=int(d["turn_idx"]),
        prompt=d["prompt"],
        candidates=candidates,
        selected_idx=(None if d.get("selected_idx") is None else int(d["selected_idx"])),
        selection_policy=d["selection_policy"],
        attempts_used=attempts_used,
        reject_count=reject_count,
        compliant_count=compliant_count,
        uncertainty_estimates=d.get("uncertainty_estimates"),
        state=(None if d.get("state") is None else dict(d.get("state") or {})),
        tool_call_ids=(
            None if d.get("tool_call_ids") is None else [str(x) for x in d.get("tool_call_ids") or []]
        ),
        step_info=dict(d.get("step_info") or d.get("info") or {}),
    )


def _from_jsonable_rejection(d: dict[str, Any] | None) -> RejectionConfigRec | None:
    if d is None:
        return None
    return RejectionConfigRec(
        enabled=bool(d["enabled"]),
        max_retry_candidates_per_turn=int(d["max_retry_candidates_per_turn"]),
        max_retry_candidates_per_episode=int(d["max_retry_candidates_per_episode"]),
    )


def _from_jsonable_config(d: dict[str, Any]) -> TrajectoryConfig:
    return TrajectoryConfig(
        K=int(d["K"]),
        M=int(d["M"]),
        N=int(d["N"]),
        policy=d["policy"],
        seed=int(d["seed"]),
        estimator=d.get("estimator"),
        rejection=_from_jsonable_rejection(d.get("rejection")),
    )


def _from_jsonable_raw(d: dict[str, Any]) -> RawTrajectory:
    record_id = str(d["record_id"])
    return RawTrajectory(
        record_id=record_id,
        dataset_key=d["dataset_key"],
        target_type=_validate_target_type(
            d["target_type"],
            context=f"persisted raw record {record_id!r}",
        ),
        env=d["env"],
        model=d["model"],
        config=_from_jsonable_config(d["config"]),
        tokenizer_name=d["tokenizer_name"],
        initial_state=(None if d.get("initial_state") is None else dict(d.get("initial_state") or {})),
        turns=[_from_jsonable_turn(t) for t in d["turns"]],
        final_reward=float(d["final_reward"]),
        solved=bool(d["solved"]),
        done_reason=d["done_reason"],
        episode_exhausted_retry_budget=bool(d.get("episode_exhausted_retry_budget", False)),
        env_backend=str(d.get("env_backend", "docker")),
        env_provenance=dict(d.get("env_provenance") or {}),
    )


def _from_jsonable_compact_scored(d: dict[str, Any]) -> ScoredTrajectory:
    record_id = str(d["record_id"])
    turns = []
    for turn in d.get("turns", []):
        estimates = {
            estimator_name: _normalize_compact_turn_entry(entry)
            for estimator_name, entry in (turn.get("uncertainty_estimates") or {}).items()
        }
        turns.append(
            ScoredTurn(
                turn_idx=int(turn["turn_idx"]),
                uncertainty_estimates=estimates,
            )
        )

    estimators = {}
    for estimator_name, meta in (d.get("estimators") or {}).items():
        normalized_meta: dict[str, Any] = {}
        for key, value in (meta or {}).items():
            if isinstance(value, Mapping):
                normalized_meta[key] = dict(_to_jsonable(value))
            else:
                normalized_meta[key] = value
        estimators[estimator_name] = normalized_meta

    return ScoredTrajectory(
        record_format=str(d.get("record_format", SCORED_RECORD_FORMAT)),
        record_id=record_id,
        raw_path=str(d.get("raw_path") or ""),
        dataset_key=d["dataset_key"],
        target_type=_validate_target_type(
            d["target_type"],
            context=f"persisted scored record {record_id!r}",
        ),
        env=d["env"],
        model=d["model"],
        final_reward=float(d["final_reward"]),
        solved=bool(d["solved"]),
        done_reason=d["done_reason"],
        env_backend=str(d.get("env_backend", "docker")),
        env_provenance=dict(d.get("env_provenance") or {}),
        estimators=estimators,
        turns=turns,
    )


def _from_jsonable_scored(
    d: dict[str, Any],
    *,
    source_path: pathlib.Path | None = None,
) -> ScoredTrajectory:
    if d.get("record_format") == SCORED_RECORD_FORMAT:
        scored = _from_jsonable_compact_scored(d)
        if not scored.raw_path:
            scored.raw_path = _infer_raw_path(
                scored.record_id,
                leaf_root=(source_path.parent.parent if source_path and source_path.parent.name == "scored" else None),
                preferred_suffix=".json.gz" if source_path and source_path.name.endswith(".json.gz") else ".json",
            )
        return scored

    raw_like = _from_jsonable_raw(d)
    return _compact_from_legacy_raw(
        raw_like,
        raw_path=None,
        leaf_root=(source_path.parent.parent if source_path and source_path.parent.name == "scored" else None),
        preferred_suffix=".json.gz" if source_path and source_path.name.endswith(".json.gz") else ".json",
    )


def merge_uncertainty_estimates(
    scored_path: pathlib.Path,
    estimator_name: str,
    per_turn_results: Iterable[dict[str, Any]],
    *,
    gzip_records: bool = False,
    estimator_params: Mapping[str, Any] | None = None,
) -> pathlib.Path:

    record = read_scored(scored_path)
    per_turn_list = list(per_turn_results)
    if len(per_turn_list) != len(record.turns):
        raise ValueError(
            f"merge_uncertainty_estimates: estimator returned {len(per_turn_list)} "
            f"per-turn results but record has {len(record.turns)} turns"
        )

    record.estimators[estimator_name] = {
        "params": dict(estimator_params or {}),
    }
    for turn, result in zip(record.turns, per_turn_list):
        normalized = {"scalar": float(result["scalar"])}
        if "extras" in result:
            extras = result.get("extras")
            normalized["extras"] = {} if extras is None else extras
        turn.uncertainty_estimates[estimator_name] = normalized

    return write_scored(
        record,
        scored_path.parent,
        gzip_records=gzip_records,
    )
