from __future__ import annotations

import re
from typing import Any, Sequence

from agent_uq.records import CandidateRecord
from agent_uq.uq.base import (
    EstimatorInfrastructureError,
    TurnContext,
    UQResult,
    with_unscorable_kind,
)
from agent_uq.uq.lm_polygraph_adapter import (
    adapt_turn_samples,
    resolve_anchor,
    _no_valid_candidates_result,
    _sample_diagnostics,
    _unscorable_anchor_result,
)
from agent_uq.uq.lm_polygraph_constants import (
    DEFAULT_P_TRUE_PROMPT,
    DEFAULT_VERBALIZED_CONFIDENCE_REGEX,
    DEFAULT_VERBALIZED_PROMPT,
    _P_TRUE_SCORING_MODE,
)
from agent_uq.uq.lm_polygraph_pairwise import (
    _effective_auxiliary_params_from_self,
    _with_adapter_contract,
)
from agent_uq.uq.lm_polygraph_runtime import get_auxiliary_model


class PTrueEstimator:
    name = "p_true"

    def __init__(
        self,
        *,
        anchor_mode: str = "selected",
        auxiliary_model_ref: str = "sampled_model",
        auxiliary_worker_device: str | None = None,
        prompt_template: str = DEFAULT_P_TRUE_PROMPT,
        expected_text: str = "True",
    ) -> None:
        if anchor_mode != "selected":
            raise ValueError(
                f"unsupported anchor_mode {anchor_mode!r}; v1 supports only 'selected'"
            )
        self.anchor_mode = anchor_mode
        self.auxiliary_model_ref = auxiliary_model_ref
        self.auxiliary_worker_device = auxiliary_worker_device
        self.prompt_template = prompt_template
        self.expected_text = expected_text

    def _effective_params(self) -> dict[str, Any]:
        params = _with_adapter_contract(
            _effective_auxiliary_params_from_self(
                self,
                (
                    "anchor_mode",
                    "auxiliary_model_ref",
                    "prompt_template",
                ),
            )
        )
        params["p_true_scoring_mode"] = _P_TRUE_SCORING_MODE
        return params

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
        prompt = self.prompt_template.format(
            initial_prompt=context.initial_prompt,
            candidate=anchor.candidate.text,
        )
        model = get_auxiliary_model(
            self.auxiliary_model_ref,
            worker_device=self.auxiliary_worker_device,
        )
        try:
            log_prob = model.generate_one_token_logprob(prompt)
        except EstimatorInfrastructureError:
            raise
        except Exception as exc:
            raise EstimatorInfrastructureError(
                f"p_true auxiliary scoring failed for model {self.auxiliary_model_ref!r}: {exc}"
            ) from exc
        extras = {
            **_sample_diagnostics(samples),
            "anchor_source": anchor.diagnostics["anchor_source"],
            "anchor_raw_index": anchor.raw_index,
            "selection_policy": anchor.diagnostics["selection_policy"],
            "auxiliary_model_ref": self.auxiliary_model_ref,
        }
        return UQResult(
            turn_uncertainty=float(-log_prob),
            extras=extras,
            params=params,
        )


class Verbalized2SEstimator:
    name = "verbalized_2s"

    def __init__(
        self,
        *,
        anchor_mode: str = "selected",
        auxiliary_model_ref: str = "sampled_model",
        auxiliary_worker_device: str | None = None,
        prompt_template: str = DEFAULT_VERBALIZED_PROMPT,
        confidence_regex: str = DEFAULT_VERBALIZED_CONFIDENCE_REGEX,
        min_new_tokens: int = 2,
        max_new_tokens: int = 10,
    ) -> None:
        if anchor_mode != "selected":
            raise ValueError(
                f"unsupported anchor_mode {anchor_mode!r}; verbalized_2s supports only 'selected'"
            )
        self.anchor_mode = anchor_mode
        self.auxiliary_model_ref = auxiliary_model_ref
        self.auxiliary_worker_device = auxiliary_worker_device
        self.prompt_template = prompt_template
        self.confidence_regex = confidence_regex
        self.min_new_tokens = int(min_new_tokens)
        self.max_new_tokens = int(max_new_tokens)

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_auxiliary_params_from_self(
                self,
                (
                    "anchor_mode",
                    "auxiliary_model_ref",
                    "prompt_template",
                    "confidence_regex",
                    "min_new_tokens",
                    "max_new_tokens",
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
        prompt = self.prompt_template.format(
            initial_prompt=context.initial_prompt,
            candidate=anchor.candidate.text,
        )
        model = get_auxiliary_model(
            self.auxiliary_model_ref,
            worker_device=self.auxiliary_worker_device,
        )
        try:
            generated_text = model.generate_text(
                prompt,
                min_new_tokens=self.min_new_tokens,
                max_new_tokens=self.max_new_tokens,
            )
        except EstimatorInfrastructureError:
            raise
        except Exception as exc:
            raise EstimatorInfrastructureError(
                f"verbalized_2s auxiliary scoring failed for model {self.auxiliary_model_ref!r}: {exc}"
            ) from exc
        parse_text = generated_text
        if self.prompt_template.rstrip().endswith("Probability:"):
            parse_text = f"Probability:{generated_text}"
        match = re.search(self.confidence_regex, parse_text)
        if match is None:
            return UQResult(
                turn_uncertainty=float("nan"),
                extras=with_unscorable_kind(
                    {
                        **_sample_diagnostics(samples),
                        "anchor_source": anchor.diagnostics["anchor_source"],
                        "anchor_raw_index": anchor.raw_index,
                        "selection_policy": anchor.diagnostics["selection_policy"],
                        "auxiliary_model_ref": self.auxiliary_model_ref,
                        "reason": "confidence_parse_failed",
                        "generated_text": generated_text,
                    },
                    kind="confidence_parse_failed",
                ),
                params=params,
            )
        confidence = float(match.group(1))
        return UQResult(
            turn_uncertainty=float(1.0 - confidence),
            extras={
                **_sample_diagnostics(samples),
                "anchor_source": anchor.diagnostics["anchor_source"],
                "anchor_raw_index": anchor.raw_index,
                "selection_policy": anchor.diagnostics["selection_policy"],
                "auxiliary_model_ref": self.auxiliary_model_ref,
                "generated_text": generated_text,
                "parsed_confidence": confidence,
            },
            params=params,
        )
