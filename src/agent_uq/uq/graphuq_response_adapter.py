from __future__ import annotations

import math
import re
import sys
from typing import Any, Sequence

from agent_uq.records import CandidateRecord, TopKEntryRec
from agent_uq.uq.graphuq_runtime import _get_tokenizer as _runtime_get_tokenizer
from agent_uq.uq.raw_turn_adapter import (
    AcceptedSpanCandidate,
    recover_accepted_span_candidate,
)


PROBINFO_KEYS = (
    "max logprob",
    "token logprob",
    "avg neg logprob",
    "avg neg logprob nor",
    "neg entropy",
    "neg entropy nor",
)


def _facade_override(name: str, original: Any) -> Any | None:
    facade = sys.modules.get("agent_uq.uq.graphuq_adapter")
    override = getattr(facade, name, None)
    if callable(override) and override is not original:
        return override
    return None


def _normalize_to_probs(logprobs: Sequence[float]) -> list[float]:

    if not logprobs:
        return []
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _make_step_probinfo(
    sampled_logprobs: Sequence[float],
    top_k_per_step: Sequence[Sequence[TopKEntryRec]],
) -> dict[str, list[float]]:

    out: dict[str, list[float]] = {k: [] for k in PROBINFO_KEYS}
    for i in range(len(sampled_logprobs)):
        step_top_k = list(top_k_per_step[i]) if i < len(top_k_per_step) else []
        if not step_top_k:
            for k in PROBINFO_KEYS:
                out[k].append(0.0)
            continue
        step_logprobs = [e.logprob for e in step_top_k]
        sampled = float(sampled_logprobs[i])

        out["max logprob"].append(max(step_logprobs))
        out["token logprob"].append(sampled)

        out["avg neg logprob"].append(-sum(step_logprobs) / len(step_logprobs))
        normed = _normalize_to_probs(step_logprobs)
        out["avg neg logprob nor"].append(
            -sum(math.log(p) for p in normed if p > 0) / max(1, len(normed))
        )
        out["neg entropy"].append(-_entropy(normed))
        out["neg entropy nor"].append(-_entropy(normed))
    return out


_STEP_PATTERN = re.compile(r"Step\s+(\d+)\s*:")


def _split_thought_into_step_text_segments(parsed_thought: str) -> list[tuple[int, int, str]]:

    matches = list(_STEP_PATTERN.finditer(parsed_thought or ""))
    if not matches:
        text = (parsed_thought or "").strip()
        if text:
            return [(0, len(parsed_thought or ""), text)]
        return []

    segments: list[tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = 0 if i == 0 and (parsed_thought or "")[: m.start()].strip() else m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(parsed_thought)
        text = parsed_thought[start:end].strip()
        if text:
            segments.append((start, end, text))
    return segments


def _accepted_span_candidate_to_graphuq_response(
    candidate: AcceptedSpanCandidate,
    *,
    tokenizer: Any,
) -> dict[str, Any]:

    parsed_thought = candidate.thought_text
    step_segments = _split_thought_into_step_text_segments(parsed_thought)
    thought_token_limit = candidate.action_token_start
    step_info_list: list[dict[str, Any]] = []
    for start_char, end_char, step_text in step_segments:
        if not step_text:
            continue
        start_token = min(
            len(tokenizer.encode(parsed_thought[:start_char], add_special_tokens=False))
            if start_char > 0 else 0,
            thought_token_limit,
        )
        end_pos = min(
            len(tokenizer.encode(parsed_thought[:end_char], add_special_tokens=False)),
            thought_token_limit,
        )
        if end_pos <= start_token:
            continue
        step_logprobs_slice = candidate.token_logprobs[start_token:end_pos]
        step_top_k_slice = candidate.top_k[start_token:end_pos]
        prob_info = _make_step_probinfo(step_logprobs_slice, step_top_k_slice)
        step_info_list.append({"text": step_text, "prob info": prob_info})

    if thought_token_limit < len(candidate.token_ids):
        answer_logprobs_slice = candidate.token_logprobs[thought_token_limit:]
        answer_top_k_slice = candidate.top_k[thought_token_limit:]
    else:
        answer_logprobs_slice = []
        answer_top_k_slice = []
    answer_prob_info = _make_step_probinfo(answer_logprobs_slice, answer_top_k_slice)

    return {
        "step-wise info": step_info_list,
        "answer info": {
            "text": candidate.action_text.strip(),
            "prob info": answer_prob_info,
        },
    }


def to_graphuq_probinfo(candidate: CandidateRecord, tokenizer_name: str) -> dict[str, Any]:

    if (
        candidate.parse_status != "ok"
        or candidate.parsed_action is None
        or not candidate.compliant
    ):
        raise ValueError(
            f"react_candidate_to_graphuq_response called on non-compliant candidate: "
            f"status={candidate.parse_status!r}, compliant={candidate.compliant!r}, "
            f"non_compliant_reason={candidate.non_compliant_reason!r}, "
            f"error={candidate.parse_error!r}"
        )
    tokenizer_factory = _facade_override("_get_tokenizer", _runtime_get_tokenizer)
    if tokenizer_factory is None:
        tokenizer_factory = _runtime_get_tokenizer
    tokenizer = tokenizer_factory(tokenizer_name)
    adapted, reason = recover_accepted_span_candidate(
        candidate,
        raw_index=0,
        tokenizer=tokenizer,
    )
    if adapted is None:
        raise ValueError(f"adapter-invalid candidate: {reason or 'adapter_invalid'}")
    return _accepted_span_candidate_to_graphuq_response(adapted, tokenizer=tokenizer)
