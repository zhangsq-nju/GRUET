from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from agent_uq.records import (
    CandidateRecord,
    SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT,
    TopKEntryRec,
)
from agent_uq.uq.base import (
    UNSCORABLE_KIND_NO_VALID_CANDIDATES,
    TurnContext,
    UQResult,
    with_unscorable_kind,
)
from agent_uq.uq.lm_polygraph_constants import _RAW_TOKEN_ARRAYS_MISALIGNED

_TERMINAL_THOUGHT_RE = re.compile(r"(?m)^Thought\s+(\d+):\s*$")

def extract_task_context_text(prompt: str) -> str:

    stripped = prompt.rstrip()
    cursor_matches = list(_TERMINAL_THOUGHT_RE.finditer(stripped))
    if not cursor_matches:
        return stripped
    cursor = cursor_matches[-1]
    history_end = cursor.start()
    next_turn_idx = int(cursor.group(1))

    def _last_line_prefix_start(prefix: str, *, end: int) -> int | None:
        pattern = re.compile(rf"(?m)^{re.escape(prefix)}")
        last_match = None
        for match in pattern.finditer(stripped[:end]):
            last_match = match
        return None if last_match is None else last_match.start()


    history_anchor = history_end
    if next_turn_idx > 1:
        action_one_start = _last_line_prefix_start("Action 1:", end=history_end)
        if action_one_start is not None:
            thought_one_start = _last_line_prefix_start("Thought 1:", end=action_one_start)
            if thought_one_start is not None:
                history_anchor = thought_one_start
                observation_zero_start = _last_line_prefix_start(
                    "Observation 0:",
                    end=thought_one_start,
                )
                if observation_zero_start is not None:
                    history_anchor = observation_zero_start
    else:
        observation_zero_start = _last_line_prefix_start("Observation 0:", end=history_end)
        if observation_zero_start is not None:
            history_anchor = observation_zero_start

    question_start = _last_line_prefix_start("Question:", end=history_anchor)
    if question_start is None:
        return stripped[:history_end].rstrip()
    return stripped[question_start:history_end].rstrip()


@dataclass(frozen=True)
class AdaptedCandidate:
    raw_index: int
    candidate: CandidateRecord
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    top_k: list[list[TopKEntryRec]]


@dataclass(frozen=True)
class AdaptedTurnSamples:
    valid_items: list[AdaptedCandidate]
    excluded_indices: list[int]
    adapter_invalid_indices: list[int]
    adapter_invalid_reasons: dict[int, str]
    sample_texts: list[str]
    sample_log_probs: np.ndarray
    sample_tokens: list[list[int]]
    sample_top_k: list[list[list[TopKEntryRec]]]


@dataclass(frozen=True)
class AnchorResolution:
    raw_index: int
    local_index: int
    candidate: AdaptedCandidate
    diagnostics: dict[str, Any]


def _recover_whole_raw_candidate(
    candidate: CandidateRecord,
    *,
    raw_index: int,
) -> tuple[AdaptedCandidate | None, str | None]:
    backend = dict(candidate.backend or {})
    if backend.get("source_format") == SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT:
        response_token_ids = [int(token_id) for token_id in backend.get("response_token_ids") or []]
        response_token_logprobs = [
            float(value) for value in backend.get("response_token_logprobs") or []
        ]
        response_top_k_raw = backend.get("response_top_k") or []
        response_top_k = [
            [
                TopKEntryRec(token_id=int(entry["token_id"]), logprob=float(entry["logprob"]))
                for entry in step
            ]
            for step in response_top_k_raw
        ]
        if (
            len(response_token_ids) != len(response_token_logprobs)
            or len(response_token_ids) != len(response_top_k)
        ):
            return (None, _RAW_TOKEN_ARRAYS_MISALIGNED)
        return (
            AdaptedCandidate(
                raw_index=raw_index,
                candidate=candidate,
                text=str(candidate.raw_response or ""),
                token_ids=response_token_ids,
                token_logprobs=response_token_logprobs,
                top_k=response_top_k,
            ),
            None,
        )
    if (
        len(candidate.token_ids) != len(candidate.token_logprobs)
        or len(candidate.token_ids) != len(candidate.top_k)
    ):
        return (None, _RAW_TOKEN_ARRAYS_MISALIGNED)
    return (
        AdaptedCandidate(
            raw_index=raw_index,
            candidate=candidate,
            text=str(candidate.text or ""),
            token_ids=list(candidate.token_ids),
            token_logprobs=list(candidate.token_logprobs),
            top_k=[list(step) for step in candidate.top_k],
        ),
        None,
    )


def adapt_turn_samples(
    candidates: Sequence[CandidateRecord],
    *,
    tokenizer_name: str,
) -> AdaptedTurnSamples:

    del tokenizer_name
    valid_items: list[AdaptedCandidate] = []
    excluded_indices: list[int] = []
    adapter_invalid_indices: list[int] = []
    adapter_invalid_reasons: dict[int, str] = {}
    sample_texts: list[str] = []
    sample_log_probs: list[float] = []
    sample_tokens: list[list[int]] = []
    sample_top_k: list[list[list[TopKEntryRec]]] = []
    for raw_index, cand in enumerate(candidates):
        if cand.compliant and cand.parsed_action is not None:
            adapted, reason = _recover_whole_raw_candidate(
                cand,
                raw_index=raw_index,
            )
            if adapted is None:
                excluded_indices.append(raw_index)
                adapter_invalid_indices.append(raw_index)
                adapter_invalid_reasons[raw_index] = str(reason or "adapter_invalid")
                continue
            valid_items.append(adapted)
            sample_texts.append(adapted.text)
            sample_log_probs.append(float(sum(adapted.token_logprobs)))
            sample_tokens.append(list(adapted.token_ids))
            sample_top_k.append([list(step) for step in adapted.top_k])
        else:
            excluded_indices.append(raw_index)
    return AdaptedTurnSamples(
        valid_items=valid_items,
        excluded_indices=excluded_indices,
        adapter_invalid_indices=adapter_invalid_indices,
        adapter_invalid_reasons=adapter_invalid_reasons,
        sample_texts=sample_texts,
        sample_log_probs=np.asarray(sample_log_probs, dtype=float),
        sample_tokens=sample_tokens,
        sample_top_k=sample_top_k,
    )


def resolve_anchor(
    samples: AdaptedTurnSamples,
    context: TurnContext,
    *,
    anchor_mode: str,
) -> AnchorResolution | None:

    if anchor_mode != "selected":
        raise ValueError(
            f"unsupported anchor_mode {anchor_mode!r}; v1 supports only 'selected'"
        )
    diagnostics: dict[str, Any] = {
        "anchor_source": "selected",
        "selection_policy": context.selection_policy,
        "anchor_raw_index": context.selected_idx,
    }
    if context.selected_idx is None:
        diagnostics["reason"] = "selected_idx_missing"
        return None
    for local_index, adapted in enumerate(samples.valid_items):
        if adapted.raw_index == context.selected_idx:
            return AnchorResolution(
                raw_index=adapted.raw_index,
                local_index=local_index,
                candidate=adapted,
                diagnostics=diagnostics,
            )
    if context.selected_idx in samples.adapter_invalid_reasons:
        diagnostics["reason"] = "selected_candidate_not_adapter_valid"
        diagnostics["adapter_invalid_reason"] = samples.adapter_invalid_reasons[context.selected_idx]
        return None
    diagnostics["reason"] = "selected_candidate_not_compliant"
    return None


def _sample_diagnostics(samples: AdaptedTurnSamples) -> dict[str, Any]:
    return {
        "excluded_indices": list(samples.excluded_indices),
        "n_valid_candidates": len(samples.valid_items),
        "adapter_invalid_indices": list(samples.adapter_invalid_indices),
        "adapter_invalid_reasons": dict(samples.adapter_invalid_reasons),
    }


def _no_valid_candidates_result(
    *,
    params: dict[str, Any],
    samples: AdaptedTurnSamples,
) -> UQResult:
    return UQResult(
        turn_uncertainty=float("nan"),
        extras=with_unscorable_kind(
            {
                **_sample_diagnostics(samples),
                "reason": "no valid candidates",
            },
            kind=UNSCORABLE_KIND_NO_VALID_CANDIDATES,
        ),
        params=params,
    )


def _unscorable_anchor_result(
    *,
    params: dict[str, Any],
    samples: AdaptedTurnSamples,
    diagnostics: dict[str, Any],
) -> UQResult:
    extras = _sample_diagnostics(samples)
    extras.update(diagnostics)
    return UQResult(
        turn_uncertainty=float("nan"),
        extras=with_unscorable_kind(
            extras,
            kind=str(extras.get("reason") or "unscorable_anchor"),
        ),
        params=params,
    )


def _safe_exp(value: float) -> float:
    return float(math.exp(max(min(value, 700.0), -700.0)))


def _concat_input(task_context_text: str, text: str, *, concat_input: bool) -> str:
    task_context = task_context_text.strip()
    if not concat_input or not task_context:
        return text
    return f"{task_context}\n{text}"


def _compute_jaccard_matrix(texts: Sequence[str]) -> np.ndarray:
    matrix = np.eye(len(texts), dtype=float)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            set1 = set(texts[i].lower().split())
            set2 = set(texts[j].lower().split())
            union = len(set1 | set2)
            score = 0.0 if union == 0 else len(set1 & set2) / union
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix
