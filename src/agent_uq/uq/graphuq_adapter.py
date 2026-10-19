from __future__ import annotations

from typing import Any

from agent_uq.uq.base import register_estimator
from agent_uq.uq.graphuq_estimator import GraphUQEstimator
from agent_uq.uq.graphuq_graph_builder import (
    _build_graph,
    _predict_action_similarity,
    _predict_text_equivalence,
)
from agent_uq.uq.graphuq_response_adapter import (
    PROBINFO_KEYS,
    _accepted_span_candidate_to_graphuq_response,
    _entropy,
    _make_step_probinfo,
    _normalize_to_probs,
    _split_thought_into_step_text_segments,
    to_graphuq_probinfo,
)
from agent_uq.uq.graphuq_runtime import (
    DEFAULT_ACTION_SIMILARITY_BATCH_SIZE,
    _ActionSimilarityScorer,
    _ActionSimilarityScorerSingleton,
    _disable_transformers_auto_device_map,
    _enable_hf_home_fallback_for_transformers_loaders,
    _get_tokenizer,
    _GraphConstructorSingleton,
    _release_graphuq_gpu_state,
    _resolve_action_similarity_device,
    _resolve_nli_device,
    resolve_graphuq_action_similarity_device,
    resolve_graphuq_nli_device,
)


@register_estimator("gut_up")
def _make_gut_up(**kwargs: Any) -> GraphUQEstimator:
    kwargs.setdefault("graph_type", "gut_up")
    return GraphUQEstimator(**kwargs)


__all__ = [
    "DEFAULT_ACTION_SIMILARITY_BATCH_SIZE",
    "GraphUQEstimator",
    "PROBINFO_KEYS",
    "_ActionSimilarityScorer",
    "_ActionSimilarityScorerSingleton",
    "_GraphConstructorSingleton",
    "_accepted_span_candidate_to_graphuq_response",
    "_build_graph",
    "_disable_transformers_auto_device_map",
    "_enable_hf_home_fallback_for_transformers_loaders",
    "_entropy",
    "_get_tokenizer",
    "_make_step_probinfo",
    "_normalize_to_probs",
    "_predict_action_similarity",
    "_predict_text_equivalence",
    "_release_graphuq_gpu_state",
    "_resolve_action_similarity_device",
    "_resolve_nli_device",
    "_split_thought_into_step_text_segments",
    "resolve_graphuq_action_similarity_device",
    "resolve_graphuq_nli_device",
    "to_graphuq_probinfo",
]
