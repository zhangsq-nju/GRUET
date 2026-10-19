from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from agent_uq.records import CandidateRecord


@dataclass
class SelectableCandidate:


    raw_index: int
    candidate: CandidateRecord


class SelectionPolicy(Protocol):
    name: str

    def choose(self, selectable: Sequence[SelectableCandidate]) -> int: ...


POLICY_REGISTRY: dict[str, Callable[..., SelectionPolicy]] = {}


def register_policy(name: str) -> Callable[[Callable[..., SelectionPolicy]], Callable[..., SelectionPolicy]]:
    def deco(factory: Callable[..., SelectionPolicy]) -> Callable[..., SelectionPolicy]:
        if name in POLICY_REGISTRY:
            raise ValueError(f"register_policy: name {name!r} already registered")
        POLICY_REGISTRY[name] = factory
        return factory

    return deco


def get_policy(name: str, **kwargs) -> SelectionPolicy:
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"unknown policy {name!r}. Registered: {sorted(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](**kwargs)
