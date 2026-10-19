from __future__ import annotations

from typing import Sequence

from agent_uq.uq.aggregators.base import register_aggregator


def _require_nonempty(values: Sequence[float]) -> None:
    if not values:
        raise ValueError("aggregator: per-turn series must be non-empty")


@register_aggregator("first_turn")
class FirstTurnAggregator:
    name = "first_turn"

    def aggregate(self, per_turn_values: Sequence[float]) -> float:
        _require_nonempty(per_turn_values)
        return float(per_turn_values[0])


@register_aggregator("last_turn")
class LastTurnAggregator:
    name = "last_turn"

    def aggregate(self, per_turn_values: Sequence[float]) -> float:
        _require_nonempty(per_turn_values)
        return float(per_turn_values[-1])


@register_aggregator("mean")
class MeanAggregator:
    name = "mean"

    def aggregate(self, per_turn_values: Sequence[float]) -> float:
        _require_nonempty(per_turn_values)
        return float(sum(per_turn_values) / len(per_turn_values))
