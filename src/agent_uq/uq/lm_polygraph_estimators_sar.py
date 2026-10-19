from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from agent_uq.records import CandidateRecord
from agent_uq.uq.base import (
    EstimatorInfrastructureError,
    TurnContext,
    UQResult,
)
from agent_uq.uq.graphuq_adapter import _get_tokenizer
from agent_uq.uq.lm_polygraph_adapter import (
    adapt_turn_samples,
    resolve_anchor,
    _concat_input,
    _no_valid_candidates_result,
    _safe_exp,
    _sample_diagnostics,
    _unscorable_anchor_result,
)
from agent_uq.uq.lm_polygraph_constants import (
    _DEFAULT_CROSS_ENCODER_NAME,
    _DEFAULT_NLI_MODEL_NAME,
)
from agent_uq.uq.lm_polygraph_pairwise import (
    _anchor_nli_forward_scores,
    _cross_encoder_similarity_matrix,
    _effective_cross_encoder_params_from_self,
    _effective_nli_params_from_self,
    _with_adapter_contract,
)
from agent_uq.uq.lm_polygraph_runtime import get_cross_encoder_backend


class SentenceSAREstimator:
    name = "sentence_sar"

    def __init__(
        self,
        *,
        cross_encoder_name: str = _DEFAULT_CROSS_ENCODER_NAME,
        cross_encoder_batch_size: int = 10,
        cross_encoder_device: str | None = None,
    ) -> None:
        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder_batch_size = int(cross_encoder_batch_size)
        self.cross_encoder_device = cross_encoder_device
        self.t = 0.001

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_cross_encoder_params_from_self(
                self,
                (
                    "cross_encoder_name",
                    "cross_encoder_device",
                ),
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        similarity = _cross_encoder_similarity_matrix(
            samples.sample_texts,
            model_name=self.cross_encoder_name,
            batch_size=self.cross_encoder_batch_size,
            device=self.cross_encoder_device,
        )
        sample_probs = np.exp(np.clip(samples.sample_log_probs, -700.0, 700.0))
        relevance = (
            sample_probs
            * similarity
            * (1.0 - np.eye(similarity.shape[0]))
        )
        sent_relevance = relevance.sum(-1) / self.t
        e_s = -np.log(sent_relevance + sample_probs)
        return UQResult(
            turn_uncertainty=float(np.mean(e_s)),
            extras=_sample_diagnostics(samples),
            params=params,
        )


def _anchor_text_without_one_token(
    *,
    tokenizer_name: str,
    token_ids: Sequence[int],
) -> list[str]:
    tokenizer = _get_tokenizer(tokenizer_name)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if len(token_ids) <= 1:
        return ["" for _ in token_ids]
    outputs: list[str] = []
    for idx, token_id in enumerate(token_ids):
        if token_id in special_ids:
            outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True))
            continue
        cropped = list(token_ids[:idx]) + list(token_ids[idx + 1 :])
        outputs.append(tokenizer.decode(cropped, skip_special_tokens=True))
    return outputs


def _token_similarity_scores(
    *,
    context: TurnContext,
    token_ids: Sequence[int],
    full_text: str,
    cross_encoder_name: str,
    cross_encoder_batch_size: int,
    cross_encoder_device: str | None,
) -> np.ndarray:
    if len(token_ids) <= 1:
        return np.full(len(token_ids), 0.5, dtype=float)
    backend = get_cross_encoder_backend(
        model_name=cross_encoder_name,
        batch_size=cross_encoder_batch_size,
        device=cross_encoder_device,
    )
    full_with_input = _concat_input(
        context.task_context_text,
        full_text,
        concat_input=True,
    )
    cropped_texts = _anchor_text_without_one_token(
        tokenizer_name=context.tokenizer_name,
        token_ids=token_ids,
    )
    pairs = [
        (
            full_with_input,
            _concat_input(context.task_context_text, cropped, concat_input=True),
        )
        for cropped in cropped_texts
    ]
    try:
        return np.asarray(backend.predict(pairs), dtype=float)
    except EstimatorInfrastructureError:
        raise
    except Exception as exc:
        raise EstimatorInfrastructureError(
            f"cross-encoder backend runtime failed for {cross_encoder_name!r}: {exc}"
        ) from exc


class TokenSAREstimator:
    name = "token_sar"

    def __init__(
        self,
        *,
        anchor_mode: str = "selected",
        cross_encoder_name: str = _DEFAULT_CROSS_ENCODER_NAME,
        cross_encoder_batch_size: int = 10,
        cross_encoder_device: str | None = None,
    ) -> None:
        if anchor_mode != "selected":
            raise ValueError(
                f"unsupported anchor_mode {anchor_mode!r}; v1 supports only 'selected'"
            )
        self.anchor_mode = anchor_mode
        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder_batch_size = int(cross_encoder_batch_size)
        self.cross_encoder_device = cross_encoder_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_cross_encoder_params_from_self(
                self,
                (
                    "anchor_mode",
                    "cross_encoder_name",
                    "cross_encoder_device",
                ),
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        anchor = resolve_anchor(samples, context, anchor_mode=self.anchor_mode)
        if anchor is None:
            return _unscorable_anchor_result(
                params=params,
                samples=samples,
                diagnostics={
                    "anchor_source": "selected",
                    "selection_policy": context.selection_policy,
                    "anchor_raw_index": context.selected_idx,
                    **({
                        "reason": "selected_idx_missing",
                    } if context.selected_idx is None else {}),
                    **({
                        "reason": "selected_candidate_not_adapter_valid",
                        "adapter_invalid_reason": samples.adapter_invalid_reasons[context.selected_idx],
                    } if context.selected_idx in samples.adapter_invalid_reasons else {}),
                    **({
                        "reason": "selected_candidate_not_compliant",
                    } if (
                        context.selected_idx is not None
                        and context.selected_idx not in samples.adapter_invalid_reasons
                    ) else {}),
                },
            )
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        similarity = _token_similarity_scores(
            context=context,
            token_ids=anchor.candidate.token_ids,
            full_text=anchor.candidate.text,
            cross_encoder_name=self.cross_encoder_name,
            cross_encoder_batch_size=self.cross_encoder_batch_size,
            cross_encoder_device=self.cross_encoder_device,
        )
        log_likelihoods = np.asarray(anchor.candidate.token_logprobs, dtype=float)
        relevance = 1.0 - similarity
        denom = float(np.sum(relevance))
        if denom <= 0:
            weights = np.full_like(relevance, 1.0 / len(relevance))
        else:
            weights = relevance / denom
        per_token_scores = (-log_likelihoods) * weights
        extras = {
            **_sample_diagnostics(samples),
            "anchor_source": anchor.diagnostics["anchor_source"],
            "anchor_raw_index": anchor.raw_index,
            "selection_policy": anchor.diagnostics["selection_policy"],
            "token_similarity": similarity.tolist(),
        }
        return UQResult(
            turn_uncertainty=float(np.sum(per_token_scores)),
            extras=extras,
            params=params,
        )


class SemanticDensityEstimator:
    name = "semantic_density"

    def __init__(
        self,
        *,
        anchor_mode: str = "selected",
        concat_input: bool = True,
        nli_model_name: str = _DEFAULT_NLI_MODEL_NAME,
        nli_batch_size: int = 32,
        nli_device: str | None = None,
    ) -> None:
        if anchor_mode != "selected":
            raise ValueError(
                f"unsupported anchor_mode {anchor_mode!r}; v1 supports only 'selected'"
            )
        self.anchor_mode = anchor_mode
        self.concat_input = bool(concat_input)
        self.nli_model_name = nli_model_name
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_nli_params_from_self(
                self,
                (
                    "anchor_mode",
                    "concat_input",
                    "nli_model_name",
                    "nli_device",
                ),
                include_execution_mode=True,
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        anchor = resolve_anchor(samples, context, anchor_mode=self.anchor_mode)
        if anchor is None:
            return _unscorable_anchor_result(
                params=params,
                samples=samples,
                diagnostics={
                    "anchor_source": "selected",
                    "selection_policy": context.selection_policy,
                    "anchor_raw_index": context.selected_idx,
                    **({
                        "reason": "selected_idx_missing",
                    } if context.selected_idx is None else {}),
                    **({
                        "reason": "selected_candidate_not_adapter_valid",
                        "adapter_invalid_reason": samples.adapter_invalid_reasons[context.selected_idx],
                    } if context.selected_idx in samples.adapter_invalid_reasons else {}),
                    **({
                        "reason": "selected_candidate_not_compliant",
                    } if (
                        context.selected_idx is not None
                        and context.selected_idx not in samples.adapter_invalid_reasons
                    ) else {}),
                },
            )
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        anchor_text = _concat_input(
            context.task_context_text,
            anchor.candidate.text,
            concat_input=self.concat_input,
        )
        sample_texts = [
            _concat_input(
                context.task_context_text,
                text,
                concat_input=self.concat_input,
            )
            for text in samples.sample_texts
        ]
        _, neutral_forward, contra_forward = _anchor_nli_forward_scores(
            anchor_text,
            sample_texts,
            model_name=self.nli_model_name,
            batch_size=self.nli_batch_size,
            device=self.nli_device,
        )
        unique_indices = np.unique(
            np.asarray(samples.sample_texts, dtype=object),
            return_index=True,
        )[1]
        numerator: list[float] = []
        denominator: list[float] = []
        for idx in unique_indices:
            token_count = max(1, len(samples.sample_tokens[int(idx)]))
            normed_prob = _safe_exp(float(samples.sample_log_probs[int(idx)]) / token_count)
            distance = float(contra_forward[int(idx)] + (neutral_forward[int(idx)] / 2.0))
            kernel_value = max(0.0, 1.0 - distance)
            numerator.append(normed_prob * kernel_value)
            denominator.append(normed_prob)
        anchor_token_count = max(1, len(anchor.candidate.token_logprobs))
        greedy_normed_prob = _safe_exp(
            float(sum(anchor.candidate.token_logprobs)) / anchor_token_count
        )
        numerator.append(greedy_normed_prob)
        denominator.append(greedy_normed_prob)
        density = float(np.sum(numerator) / np.sum(denominator))
        extras = {
            **_sample_diagnostics(samples),
            "anchor_source": anchor.diagnostics["anchor_source"],
            "anchor_raw_index": anchor.raw_index,
            "selection_policy": anchor.diagnostics["selection_policy"],
        }
        return UQResult(
            turn_uncertainty=-density,
            extras=extras,
            params=params,
        )
