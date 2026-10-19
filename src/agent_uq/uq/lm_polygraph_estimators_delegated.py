from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np

from agent_uq.records import CandidateRecord
from agent_uq.uq.base import TurnContext, UQResult
from agent_uq.uq.lm_polygraph_adapter import (
    adapt_turn_samples,
    resolve_anchor,
    _no_valid_candidates_result,
    _sample_diagnostics,
    _unscorable_anchor_result,
)
from agent_uq.uq.lm_polygraph_pairwise import _with_adapter_contract
from agent_uq.vendor import lm_polygraph as lm_polygraph_bootstrap

lm_polygraph_bootstrap.ensure()
from lm_polygraph.estimators import Perplexity, PredictiveEntropy


def _facade_override(name: str, original: Any) -> Any | None:
    facade = sys.modules.get("agent_uq.uq.lm_polygraph")
    override = getattr(facade, name, None)
    if callable(override) and override is not original:
        return override
    return None


class PPLEstimator:
    name = "ppl"

    def __init__(self) -> None:
        estimator_cls = _facade_override("Perplexity", Perplexity) or Perplexity
        self._estimator = estimator_cls()

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract({})

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if context.selected_idx is None:
            return _unscorable_anchor_result(
                params=params,
                samples=samples,
                diagnostics={
                    "anchor_source": "selected",
                    "selection_policy": context.selection_policy,
                    "anchor_raw_index": context.selected_idx,
                    "reason": "selected_idx_missing",
                },
            )
        if context.selected_idx < 0 or context.selected_idx >= len(candidates):
            return _unscorable_anchor_result(
                params=params,
                samples=samples,
                diagnostics={
                    "anchor_source": "selected",
                    "selection_policy": context.selection_policy,
                    "anchor_raw_index": context.selected_idx,
                    "reason": "selected_idx_out_of_range",
                },
            )

        anchor = resolve_anchor(samples, context, anchor_mode="selected")
        if anchor is None:
            diagnostics = {
                "anchor_source": "selected",
                "selection_policy": context.selection_policy,
                "anchor_raw_index": context.selected_idx,
            }
            if context.selected_idx in samples.adapter_invalid_reasons:
                diagnostics["reason"] = "selected_candidate_not_adapter_valid"
                diagnostics["adapter_invalid_reason"] = samples.adapter_invalid_reasons[
                    context.selected_idx
                ]
            else:
                diagnostics["reason"] = "selected_candidate_not_compliant"
            return _unscorable_anchor_result(
                params=params,
                samples=samples,
                diagnostics=diagnostics,
            )

        stats = {
            "greedy_log_likelihoods": [list(anchor.candidate.token_logprobs)],
        }
        scores = np.asarray(self._estimator(stats), dtype=float)
        extras = {
            **_sample_diagnostics(samples),
            "anchor_source": anchor.diagnostics["anchor_source"],
            "anchor_raw_index": anchor.raw_index,
            "selection_policy": anchor.diagnostics["selection_policy"],
            "n_selected_candidates": 1,
            "selected_text": anchor.candidate.text,
        }
        return UQResult(
            turn_uncertainty=float(scores[0]),
            extras=extras,
            params=params,
        )


class PredictiveEntropyEstimator:
    name = "predictive_entropy"

    def __init__(self) -> None:
        estimator_cls = (
            _facade_override("PredictiveEntropy", PredictiveEntropy)
            or PredictiveEntropy
        )
        self._estimator = estimator_cls()

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract({})

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        stats = {
            "sample_texts": [list(samples.sample_texts)],
            "sample_log_probs": [np.asarray(samples.sample_log_probs, dtype=float)],
            "sample_tokens": [list(samples.sample_tokens)],
        }
        scores = np.asarray(self._estimator(stats), dtype=float)
        return UQResult(
            turn_uncertainty=float(scores[0]),
            extras={
                **_sample_diagnostics(samples),
                "valid_candidate_indices": [item.raw_index for item in samples.valid_items],
                "sample_texts": list(samples.sample_texts),
            },
            params=params,
        )
