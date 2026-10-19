from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent_uq.prompts.parser import recover_accepted_current_turn
from agent_uq.records import (
    CandidateRecord,
    SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT,
    TopKEntryRec,
)


CURRENT_TURN_ADAPTER_CONTRACT = "parser_current_turn_span_v2"

RAW_TOKEN_ARRAYS_MISALIGNED = "raw_token_arrays_misaligned"
TOOL_CALL_NATIVE_CURRENT_TURN_SPAN_UNRECOVERABLE = (
    "tool_call_native_current_turn_span_unrecoverable"
)
ACCEPTED_SPAN_NOT_FOUND = "accepted_span_not_found"
ACCEPTED_SPAN_EMPTY = "accepted_span_empty"
ACCEPTED_SPAN_TOKEN_SLICE_UNRECOVERABLE = "accepted_span_token_slice_unrecoverable"
ACCEPTED_THOUGHT_ACTION_BOUNDARY_UNRECOVERABLE = (
    "accepted_thought_action_boundary_unrecoverable"
)


@dataclass(frozen=True)
class AcceptedSpanCandidate:
    raw_index: int
    candidate: CandidateRecord
    text: str
    thought_text: str
    action_text: str
    token_ids: list[int]
    token_logprobs: list[float]
    top_k: list[list[TopKEntryRec]]
    action_token_start: int


def _candidate_uses_tool_call_backend(candidate: CandidateRecord) -> bool:
    backend = dict(candidate.backend or {})
    return (
        isinstance(backend.get("tool_call"), dict)
        or (
            isinstance(backend.get("tool_calls"), list)
            and len(backend.get("tool_calls") or []) > 0
        )
    )


def _candidate_source_format(candidate: CandidateRecord) -> str | None:
    backend = dict(candidate.backend or {})
    value = backend.get("source_format")
    return None if value in (None, "") else str(value)


def _count_tokens(tokenizer: Any, text: str) -> int:
    if text == "":
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _recover_imported_swe_accepted_span_candidate(
    candidate: CandidateRecord,
    *,
    raw_index: int,
) -> tuple[AcceptedSpanCandidate | None, str | None]:
    backend = dict(candidate.backend or {})
    char_spans = dict(backend.get("char_spans") or {})
    token_spans = dict(backend.get("token_spans") or {})
    required_char_keys = {
        "accepted_start",
        "accepted_end",
        "thought_start",
        "thought_end",
        "action_start",
        "action_end",
    }
    required_token_keys = {"accepted_start", "accepted_end", "action_start", "action_end"}
    if not required_char_keys.issubset(char_spans) or not required_token_keys.issubset(token_spans):
        return (None, ACCEPTED_SPAN_TOKEN_SLICE_UNRECOVERABLE)

    accepted_start_char = int(char_spans["accepted_start"])
    accepted_end_char = int(char_spans["accepted_end"])
    thought_start_char = int(char_spans["thought_start"])
    thought_end_char = int(char_spans["thought_end"])
    action_start_char = int(char_spans["action_start"])
    action_end_char = int(char_spans["action_end"])
    text = str(candidate.text or "")
    accepted_len_char = accepted_end_char - accepted_start_char
    if accepted_len_char <= 0 or accepted_len_char != len(text):
        return (None, ACCEPTED_SPAN_EMPTY)
    thought_start_offset_char = thought_start_char - accepted_start_char
    thought_end_offset_char = thought_end_char - accepted_start_char
    action_offset_char = action_start_char - accepted_start_char
    action_end_offset_char = action_end_char - accepted_start_char
    if (
        thought_start_offset_char < 0
        or thought_end_offset_char < thought_start_offset_char
        or thought_end_offset_char > len(text)
        or action_offset_char < 0
        or action_offset_char < thought_end_offset_char
        or action_end_offset_char < action_offset_char
        or action_end_offset_char > len(text)
    ):
        return (None, ACCEPTED_THOUGHT_ACTION_BOUNDARY_UNRECOVERABLE)

    accepted_start_tok = int(token_spans["accepted_start"])
    accepted_end_tok = int(token_spans["accepted_end"])
    action_start_tok = int(token_spans["action_start"])
    action_end_tok = int(token_spans["action_end"])
    accepted_len_tok = accepted_end_tok - accepted_start_tok
    if accepted_len_tok <= 0 or accepted_len_tok != len(candidate.token_ids):
        return (None, ACCEPTED_SPAN_TOKEN_SLICE_UNRECOVERABLE)
    action_token_start = action_start_tok - accepted_start_tok
    action_token_end = action_end_tok - accepted_start_tok
    if (
        action_token_start < 0
        or action_token_end < action_token_start
        or action_token_end > len(candidate.token_ids)
    ):
        return (None, ACCEPTED_THOUGHT_ACTION_BOUNDARY_UNRECOVERABLE)

    return (
        AcceptedSpanCandidate(
            raw_index=raw_index,
            candidate=candidate,
            text=text,
            thought_text=text[thought_start_offset_char:thought_end_offset_char],
            action_text=text[action_offset_char:action_end_offset_char],
            token_ids=list(candidate.token_ids),
            token_logprobs=list(candidate.token_logprobs),
            top_k=[list(step) for step in candidate.top_k],
            action_token_start=action_token_start,
        ),
        None,
    )


def recover_accepted_span_candidate(
    candidate: CandidateRecord,
    *,
    raw_index: int,
    tokenizer: Any,
) -> tuple[AcceptedSpanCandidate | None, str | None]:
    text = str(candidate.text or "")
    if (
        len(candidate.token_ids) != len(candidate.token_logprobs)
        or len(candidate.token_ids) != len(candidate.top_k)
    ):
        return (None, RAW_TOKEN_ARRAYS_MISALIGNED)
    if _candidate_source_format(candidate) == SWE_THOUGHT_ACTION_CODEBLOCK_SOURCE_FORMAT:
        return _recover_imported_swe_accepted_span_candidate(
            candidate,
            raw_index=raw_index,
        )

    accepted, _ = recover_accepted_current_turn(text)
    if accepted is None:
        if _candidate_uses_tool_call_backend(candidate):
            return (None, TOOL_CALL_NATIVE_CURRENT_TURN_SPAN_UNRECOVERABLE)
        return (None, ACCEPTED_SPAN_NOT_FOUND)

    start_char = int(accepted.accepted_span_start)
    end_char = int(accepted.accepted_span_end)
    action_char = int(accepted.action_marker_start)
    if end_char <= start_char:
        return (None, ACCEPTED_SPAN_EMPTY)
    if action_char <= start_char or action_char >= end_char:
        return (None, ACCEPTED_THOUGHT_ACTION_BOUNDARY_UNRECOVERABLE)

    accepted_text = text[start_char:end_char]
    accepted_token_count = _count_tokens(tokenizer, accepted_text)
    if accepted_token_count <= 0 or accepted_token_count > len(candidate.token_ids):
        return (None, ACCEPTED_SPAN_TOKEN_SLICE_UNRECOVERABLE)
    if text[end_char:].strip() and accepted_token_count >= len(candidate.token_ids):
        return (None, ACCEPTED_SPAN_TOKEN_SLICE_UNRECOVERABLE)

    action_offset = action_char - start_char
    action_token_start = _count_tokens(tokenizer, accepted_text[:action_offset])
    if action_token_start < 0 or action_token_start > accepted_token_count:
        return (None, ACCEPTED_THOUGHT_ACTION_BOUNDARY_UNRECOVERABLE)

    return (
        AcceptedSpanCandidate(
            raw_index=raw_index,
            candidate=candidate,
            text=accepted_text,
            thought_text=accepted_text[:action_offset],
            action_text=accepted_text[action_offset:],
            token_ids=list(candidate.token_ids[:accepted_token_count]),
            token_logprobs=list(candidate.token_logprobs[:accepted_token_count]),
            top_k=[list(step) for step in candidate.top_k[:accepted_token_count]],
            action_token_start=action_token_start,
        ),
        None,
    )


def with_adapter_contract(
    params: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    merged = dict(params)
    merged[field_name] = CURRENT_TURN_ADAPTER_CONTRACT
    return merged
