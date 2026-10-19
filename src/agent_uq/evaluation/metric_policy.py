from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_uq.evaluation.pair_cache import make_pair_cache_row
from agent_uq.metrics import safe_metric_call as _safe_metric_call
from agent_uq.uq.aggregators.base import get_aggregator
from agent_uq.uq.base import UNSCORABLE_KIND_NO_VALID_CANDIDATES


@dataclass
class MetricRowInputs:
    valid_pairs: list[dict[str, Any]]
    target_type: str
    n_excluded_plain: int
    n_salvaged_no_candidate_nan: int
    n_excluded_augmented: int
    metrics: dict[str, float | None] = field(default_factory=dict)


def _record_target(record: Any) -> float:
    if record.target_type == "binary":
        return 1.0 if record.solved else 0.0
    return float(record.final_reward)


def _per_turn_uncertainties(record: Any, estimator_name: str) -> list[float] | None:

    series: list[float] = []
    for turn in record.turns:
        ue = turn.uncertainty_estimates or {}
        if estimator_name not in ue:
            return None
        scalar = ue[estimator_name].get("scalar")
        if scalar is None or not isinstance(scalar, (int, float)):
            return None
        if isinstance(scalar, float) and math.isnan(scalar):
            return None
        series.append(float(scalar))
    if not series:
        return None
    return series


def _salvageable_no_candidate_nan_series(
    record: Any,
    estimator_name: str,
) -> list[float] | None:

    series: list[float] = []
    dropped_any = False
    for turn in record.turns:
        ue = turn.uncertainty_estimates or {}
        if estimator_name not in ue:
            return None
        entry = ue[estimator_name]
        scalar = entry.get("scalar")
        if scalar is None or not isinstance(scalar, (int, float)):
            return None
        scalar_value = float(scalar)
        if math.isnan(scalar_value):
            extras = entry.get("extras")
            if (
                not isinstance(extras, dict)
                or extras.get("unscorable_kind")
                != UNSCORABLE_KIND_NO_VALID_CANDIDATES
            ):
                return None
            dropped_any = True
            continue
        series.append(scalar_value)
    if not dropped_any or not series:
        return None
    return series


def _build_arrays_for_row(
    records: Sequence[Any],
    estimator_name: str,
    aggregator_name: str,
) -> tuple[list[float], list[float], int, str]:

    pairs, excluded, target_type = _build_valid_pairs_for_row(
        records,
        estimator_name,
        aggregator_name,
    )
    return (
        [float(pair["uncertainty"]) for pair in pairs],
        [float(pair["target"]) for pair in pairs],
        excluded,
        target_type,
    )


def _build_valid_pairs_for_row(
    records: Sequence[Any],
    estimator_name: str,
    aggregator_name: str,
) -> tuple[list[dict[str, Any]], int, str]:

    aggregator = get_aggregator(aggregator_name)
    pairs: list[dict[str, Any]] = []
    excluded = 0
    target_type = records[0].target_type if records else "binary"

    for record in records:
        per_turn = _per_turn_uncertainties(record, estimator_name)
        if per_turn is None:
            excluded += 1
            continue
        try:
            uncertainty = float(aggregator.aggregate(per_turn))
        except Exception:
            excluded += 1
            continue
        pairs.append(
            make_pair_cache_row(
                source_kind="contender",
                record_id=record.record_id,
                dataset_key=record.dataset_key,
                model=record.model,
                env_backend=record.env_backend,
                estimator=estimator_name,
                aggregator=aggregator_name,
                uncertainty=uncertainty,
                target=_record_target(record),
                target_type=target_type,
            )
        )
    return pairs, excluded, target_type


def _build_augmented_arrays(
    U_valid: list[float],
    targets_valid: list[float],
    n_excluded_augmented: int,
) -> tuple[list[float] | None, list[float] | None]:

    if not U_valid:
        return None, None
    u_max = max(U_valid) + 1.0
    u_aug = list(U_valid) + [u_max] * n_excluded_augmented
    targets_aug = list(targets_valid) + [0.0] * n_excluded_augmented
    return u_aug, targets_aug


def _build_augmented_arrays_for_row(
    records: Sequence[Any],
    estimator_name: str,
    aggregator_name: str,
    U_values: list[float],
    targets: list[float],
) -> tuple[list[float] | None, list[float] | None, int, int]:

    aggregator = get_aggregator(aggregator_name)
    augmented_valid = list(U_values)
    augmented_targets = list(targets)
    salvaged = 0
    excluded_augmented = 0

    for record in records:
        if _per_turn_uncertainties(record, estimator_name) is not None:
            continue
        salvage_series = _salvageable_no_candidate_nan_series(record, estimator_name)
        if salvage_series is None:
            excluded_augmented += 1
            continue
        try:
            augmented_valid.append(float(aggregator.aggregate(salvage_series)))
        except Exception:
            excluded_augmented += 1
            continue
        augmented_targets.append(_record_target(record))
        salvaged += 1

    u_aug, targets_aug = _build_augmented_arrays(
        augmented_valid,
        augmented_targets,
        excluded_augmented,
    )
    return u_aug, targets_aug, salvaged, excluded_augmented


def build_interleaved_metric_names(metrics: Sequence[str]) -> list[str]:
    out: list[str] = []
    for metric in metrics:
        out.append(metric)
        out.append(f"{metric}_incl_excluded")
    return out


def build_excluded_imputation_metadata(metrics: Sequence[str]) -> dict[str, Any]:
    return {
        "target": 0.0,
        "uncertainty_rule": "max(U_valid_augmented) + 1.0",
        "salvage_rule": (
            "drop per-turn NaN values whose extras.unscorable_kind == "
            f"{UNSCORABLE_KIND_NO_VALID_CANDIDATES!r} before aggregation"
        ),
        "applies_to": [f"{m}_incl_excluded" for m in metrics],
        "empty_u_valid_behavior": "NA",
        "accounting_fields": [
            "n_excluded_plain",
            "n_salvaged_no_candidate_nan",
            "n_excluded_augmented",
            "n_tasks_total",
        ],
    }


def build_metric_row_inputs(
    records: Sequence[Any],
    estimator_name: str,
    aggregator_name: str,
    metrics: Sequence[str],
) -> MetricRowInputs:
    valid_pairs, n_excluded_plain, target_type = _build_valid_pairs_for_row(
        records,
        estimator_name,
        aggregator_name,
    )
    U_values = [float(pair["uncertainty"]) for pair in valid_pairs]
    targets = [float(pair["target"]) for pair in valid_pairs]
    (
        u_aug,
        targets_aug,
        n_salvaged_no_candidate_nan,
        n_excluded_augmented,
    ) = _build_augmented_arrays_for_row(
        records,
        estimator_name,
        aggregator_name,
        U_values,
        targets,
    )
    row_metrics: dict[str, float | None] = {}
    for metric in metrics:
        row_metrics[metric] = _safe_metric_call(metric, U_values, targets, target_type)
        aug_key = f"{metric}_incl_excluded"
        row_metrics[aug_key] = (
            None
            if u_aug is None or targets_aug is None
            else _safe_metric_call(metric, u_aug, targets_aug, target_type)
        )
    return MetricRowInputs(
        valid_pairs=valid_pairs,
        target_type=target_type,
        n_excluded_plain=n_excluded_plain,
        n_salvaged_no_candidate_nan=n_salvaged_no_candidate_nan,
        n_excluded_augmented=n_excluded_augmented,
        metrics=row_metrics,
    )
