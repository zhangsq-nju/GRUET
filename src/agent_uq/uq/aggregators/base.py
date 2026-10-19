from __future__ import annotations

from typing import Callable, Protocol, Sequence


class TaskUQAggregator(Protocol):
    name: str

    def aggregate(self, per_turn_values: Sequence[float]) -> float: ...


AGGREGATOR_REGISTRY: dict[str, Callable[..., TaskUQAggregator]] = {}


def register_aggregator(name: str) -> Callable[[Callable[..., TaskUQAggregator]], Callable[..., TaskUQAggregator]]:
    def deco(factory: Callable[..., TaskUQAggregator]) -> Callable[..., TaskUQAggregator]:
        if name in AGGREGATOR_REGISTRY:
            raise ValueError(f"register_aggregator: name {name!r} already registered")
        AGGREGATOR_REGISTRY[name] = factory
        return factory

    return deco


def get_aggregator(name: str, **kwargs) -> TaskUQAggregator:
    if name not in AGGREGATOR_REGISTRY:
        registered = sorted(AGGREGATOR_REGISTRY.keys())
        raise KeyError(
            f"unknown aggregator {name!r}. Registered: {registered}"
        )
    return AGGREGATOR_REGISTRY[name](**kwargs)
