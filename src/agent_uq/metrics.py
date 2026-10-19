from __future__ import annotations

from agent_uq.vendor import graphuq as graphuq_bootstrap

_BINARY_ONLY_METRICS = {"ROCAUC"}


def _instantiate_metric(metric_name: str):

    graphuq_bootstrap.ensure()
    from src.ue_metrics import ROCAUC

    table = {
        "ROCAUC": ROCAUC,
    }
    if metric_name not in table:
        raise KeyError(
            f"unknown metric {metric_name!r}; supported: {sorted(table)}"
        )
    return table[metric_name]()


def safe_metric_call(
    metric_name: str,
    U_values: list[float],
    targets: list[float],
    target_type: str,
) -> float | None:

    if metric_name not in _BINARY_ONLY_METRICS:
        return None
    if metric_name in _BINARY_ONLY_METRICS:
        if target_type != "binary":
            return None
        unique_targets = {int(t) for t in targets}
        if len(unique_targets) < 2:
            return None
    if not U_values:
        return None
    try:
        metric = _instantiate_metric(metric_name)
        return float(metric(U_values, targets))
    except Exception:
        return None


__all__ = ["safe_metric_call"]
