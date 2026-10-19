from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

from agent_uq.records import CandidateRecord

if TYPE_CHECKING:
    from agent_uq.uq.graph_artifact import GraphArtifactRecordSession
    from agent_uq.uq.pairwise_nli_artifact import PairwiseNLIArtifactRecordSession


UNSCORABLE_KIND_NO_VALID_CANDIDATES = "no_valid_candidates"


def with_unscorable_kind(extras: dict[str, Any], *, kind: str) -> dict[str, Any]:

    return {**extras, "unscorable_kind": str(kind)}


@dataclass
class UQResult:


    turn_uncertainty: float
    extras: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class EstimatorInfrastructureError(RuntimeError):
    pass


@dataclass
class TurnContext:


    record_id: str
    dataset_key: str
    turn_idx: int
    tokenizer_name: str
    model_name: str = ""
    prompt: str = ""
    initial_prompt: str = ""
    task_context_text: str = ""
    selection_policy: str = ""
    selected_idx: int | None = None
    excluded_candidate_indices: list[int] = field(default_factory=list)
    pairwise_nli_artifact_dir: pathlib.Path | None = None
    pairwise_nli_artifact_gzip: bool = False
    pairwise_nli_artifact_session: "PairwiseNLIArtifactRecordSession | None" = None
    graph_artifact_dir: pathlib.Path | None = None
    graph_artifact_gzip: bool = False
    graph_artifact_session: "GraphArtifactRecordSession | None" = None


class UQEstimator(Protocol):


    name: str

    def score(
        self,
        candidates: Sequence[CandidateRecord],
        context: TurnContext,
    ) -> UQResult: ...


ESTIMATOR_REGISTRY: dict[str, Callable[..., UQEstimator]] = {}


def register_estimator(name: str) -> Callable[[Callable[..., UQEstimator]], Callable[..., UQEstimator]]:


    def deco(factory: Callable[..., UQEstimator]) -> Callable[..., UQEstimator]:
        if name in ESTIMATOR_REGISTRY:
            raise ValueError(f"register_estimator: name {name!r} already registered")
        ESTIMATOR_REGISTRY[name] = factory
        return factory

    return deco


def get_estimator(name: str, **kwargs: Any) -> UQEstimator:

    if name not in ESTIMATOR_REGISTRY:
        registered = sorted(ESTIMATOR_REGISTRY.keys())
        raise KeyError(
            f"unknown estimator {name!r}. Registered: {registered}"
        )
    return ESTIMATOR_REGISTRY[name](**kwargs)
