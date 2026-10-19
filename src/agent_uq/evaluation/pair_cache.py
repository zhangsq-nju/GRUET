from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PAIR_CACHE_SCHEMA_VERSION = "posthoc_pair_cache_v1"
PAIR_CACHE_COLUMNS = (
    "source_kind",
    "record_id",
    "dataset_key",
    "model",
    "env_backend",
    "estimator",
    "aggregator",
    "metric_name",
    "candidate_id",
    "uncertainty",
    "target",
    "target_type",
)


@dataclass(frozen=True)
class PairCacheRef:


    path: pathlib.Path
    schema_version: str = PAIR_CACHE_SCHEMA_VERSION
    source_kind: str | None = None
    n_rows: int | None = None

    def to_jsonable(self, *, base_dir: pathlib.Path | None = None) -> dict[str, Any]:
        path = self.path
        if base_dir is not None:
            try:
                rendered_path = str(path.relative_to(base_dir))
            except ValueError:
                rendered_path = str(path)
        else:
            rendered_path = str(path)
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "path": rendered_path,
        }
        if self.source_kind is not None:
            out["source_kind"] = self.source_kind
        if self.n_rows is not None:
            out["n_rows"] = int(self.n_rows)
        return out


def make_pair_cache_row(
    *,
    source_kind: str,
    record_id: str,
    dataset_key: str,
    model: str,
    env_backend: str,
    estimator: str,
    aggregator: str,
    uncertainty: float,
    target: float,
    target_type: str,
    metric_name: str = "",
    candidate_id: str = "",
) -> dict[str, Any]:


    if target_type not in {"binary", "continuous"}:
        raise ValueError(f"unsupported target_type {target_type!r}")
    uncertainty_value = float(uncertainty)
    target_value = float(target)
    if not math.isfinite(uncertainty_value):
        raise ValueError("uncertainty must be finite")
    if not math.isfinite(target_value):
        raise ValueError("target must be finite")
    return {
        "source_kind": str(source_kind),
        "record_id": str(record_id),
        "dataset_key": str(dataset_key),
        "model": str(model),
        "env_backend": str(env_backend),
        "estimator": str(estimator),
        "aggregator": str(aggregator),
        "metric_name": str(metric_name),
        "candidate_id": str(candidate_id),
        "uncertainty": uncertainty_value,
        "target": target_value,
        "target_type": str(target_type),
    }


def validate_pair_cache_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_kind: str | None = None,
) -> None:


    required = set(PAIR_CACHE_COLUMNS)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"pair-cache row {index} missing columns: {sorted(missing)}")
        if source_kind is not None and str(row["source_kind"]) != source_kind:
            raise ValueError(
                f"pair-cache row {index} has source_kind={row['source_kind']!r}; "
                f"expected {source_kind!r}"
            )
        target_type = str(row["target_type"])
        if target_type not in {"binary", "continuous"}:
            raise ValueError(
                f"pair-cache row {index} has unsupported target_type {target_type!r}"
            )
        for field in ("uncertainty", "target"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"pair-cache row {index} has non-finite {field}")


def write_pair_cache_csv(
    path: str | pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_kind: str | None = None,
) -> PairCacheRef:


    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_pair_cache_rows(rows, source_kind=source_kind)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIR_CACHE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column))
                    for column in PAIR_CACHE_COLUMNS
                }
            )
    return PairCacheRef(
        path=path,
        source_kind=source_kind,
        n_rows=len(rows),
    )


def read_pair_cache_csv(path: str | pathlib.Path) -> list[dict[str, Any]]:


    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PAIR_CACHE_COLUMNS):
            raise ValueError(
                f"pair-cache {path} has columns {reader.fieldnames!r}; "
                f"expected {list(PAIR_CACHE_COLUMNS)!r}"
            )
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["uncertainty"] = float(parsed["uncertainty"])
            parsed["target"] = float(parsed["target"])
            rows.append(parsed)
    validate_pair_cache_rows(rows)
    return rows


def filter_pair_cache_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_kind: str | None = None,
    dataset_key: str | None = None,
    model: str | None = None,
    env_backend: str | None = None,
    estimator: str | None = None,
    aggregator: str | None = None,
    metric_name: str | None = None,
    candidate_id: str | None = None,
) -> list[dict[str, Any]]:


    out: list[dict[str, Any]] = []
    for row in rows:
        if source_kind is not None and str(row["source_kind"]) != source_kind:
            continue
        if dataset_key is not None and str(row["dataset_key"]) != dataset_key:
            continue
        if model is not None and str(row["model"]) != model:
            continue
        if env_backend is not None and str(row["env_backend"]) != env_backend:
            continue
        if estimator is not None and str(row["estimator"]) != estimator:
            continue
        if aggregator is not None and str(row["aggregator"]) != aggregator:
            continue
        if metric_name is not None and str(row["metric_name"]) not in {"", metric_name}:
            continue
        if candidate_id is not None and str(row["candidate_id"]) != candidate_id:
            continue
        out.append(dict(row))
    return out


def pair_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float], str]:


    validate_pair_cache_rows(rows)
    target_types = {str(row["target_type"]) for row in rows}
    if not rows:
        return [], [], "binary"
    if len(target_types) != 1:
        raise ValueError(f"pair-cache rows mix target types: {sorted(target_types)}")
    return (
        [float(row["uncertainty"]) for row in rows],
        [float(row["target"]) for row in rows],
        next(iter(target_types)),
    )


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


__all__ = [
    "PAIR_CACHE_SCHEMA_VERSION",
    "PAIR_CACHE_COLUMNS",
    "PairCacheRef",
    "make_pair_cache_row",
    "validate_pair_cache_rows",
    "write_pair_cache_csv",
    "read_pair_cache_csv",
    "filter_pair_cache_rows",
    "pair_arrays",
]
