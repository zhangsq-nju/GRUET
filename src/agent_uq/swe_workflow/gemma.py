from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


GEMMA_REASONING_MARKER = "REASONING"
COMMAND_FENCE = "```"
GEMMA_CURRENT_REASONING_PROJECTION = "gemma_last_reasoning_before_action_fence_v1"


@dataclass(frozen=True)
class CandidateCurrentReasoningProjection:
    accepted_text: str
    raw_response: str
    parsed_thought: str
    char_spans: dict[str, int]
    token_spans: dict[str, int]
    response_token_start: int
    response_token_end: int
    metadata: dict[str, Any]

def _is_gemma_series_name(*values: str | None) -> bool:
    return any("gemma" in str(value or "").lower() for value in values)

def _count_import_tokens(tokenizer: Any, text: str) -> int:
    if text == "":
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _project_gemma_current_reasoning_candidate(
    candidate: Mapping[str, Any],
    *,
    tokenizer: Any,
) -> CandidateCurrentReasoningProjection | None:
    accepted_text = str(candidate.get("accepted_text") or "")
    if not accepted_text:
        return None

    char_spans = {
        key: int(value)
        for key, value in dict(candidate.get("char_spans") or {}).items()
    }
    token_spans = {
        key: int(value)
        for key, value in dict(candidate.get("token_spans") or {}).items()
    }
    required_char_keys = {
        "accepted_start",
        "accepted_end",
        "thought_start",
        "thought_end",
        "action_start",
        "action_end",
    }
    required_token_keys = {
        "accepted_start",
        "accepted_end",
        "action_start",
        "action_end",
    }
    if not required_char_keys.issubset(char_spans) or not required_token_keys.issubset(
        token_spans
    ):
        return None

    accepted_start_char = char_spans["accepted_start"]
    action_start_offset = char_spans["action_start"] - accepted_start_char
    action_end_offset = char_spans["action_end"] - accepted_start_char
    thought_end_offset = char_spans["thought_end"] - accepted_start_char
    if not (0 <= action_start_offset <= action_end_offset <= len(accepted_text)):
        return None
    if not (0 <= thought_end_offset <= action_start_offset):
        return None

    action_fence_start = accepted_text.rfind(COMMAND_FENCE, 0, action_start_offset)
    if action_fence_start < 0:
        return None
    action_fence_end = accepted_text.find(COMMAND_FENCE, action_end_offset)
    if action_fence_end < 0:
        return None
    trim_start_char = accepted_text.rfind(
        GEMMA_REASONING_MARKER,
        0,
        action_fence_start,
    )
    if trim_start_char <= 0 or trim_start_char >= thought_end_offset:
        return None

    accepted_start_tok = token_spans["accepted_start"]
    accepted_end_tok = token_spans["accepted_end"]
    action_start_tok = token_spans["action_start"]
    action_end_tok = token_spans["action_end"]
    if not (accepted_start_tok <= action_start_tok <= action_end_tok <= accepted_end_tok):
        return None

    trim_prefix_tokens = _count_import_tokens(tokenizer, accepted_text[:trim_start_char])
    response_token_start = accepted_start_tok + trim_prefix_tokens
    response_token_start = min(max(response_token_start, accepted_start_tok), action_start_tok)
    response_token_end = accepted_end_tok
    if response_token_start >= response_token_end:
        return None

    projected_text = accepted_text[trim_start_char:]
    projected_action_start = action_start_offset - trim_start_char
    projected_action_end = action_end_offset - trim_start_char
    projected_thought_end = thought_end_offset - trim_start_char
    if not (
        0
        <= projected_thought_end
        <= projected_action_start
        <= projected_action_end
        <= len(projected_text)
    ):
        return None

    projected_token_count = response_token_end - response_token_start
    projected_action_start_tok = action_start_tok - response_token_start
    projected_action_end_tok = action_end_tok - response_token_start
    if not (0 <= projected_action_start_tok <= projected_action_end_tok <= projected_token_count):
        return None

    return CandidateCurrentReasoningProjection(
        accepted_text=projected_text,
        raw_response=projected_text,
        parsed_thought=projected_text[:projected_thought_end],
        char_spans={
            "accepted_start": 0,
            "accepted_end": len(projected_text),
            "thought_start": 0,
            "thought_end": projected_thought_end,
            "action_start": projected_action_start,
            "action_end": projected_action_end,
        },
        token_spans={
            "accepted_start": 0,
            "accepted_end": projected_token_count,
            "thought_start": 0,
            "thought_end": max(
                0,
                token_spans.get("thought_end", action_start_tok) - response_token_start,
            ),
            "action_start": projected_action_start_tok,
            "action_end": projected_action_end_tok,
        },
        response_token_start=response_token_start,
        response_token_end=response_token_end,
        metadata={
            "strategy": GEMMA_CURRENT_REASONING_PROJECTION,
            "trim_start_char": trim_start_char,
            "trimmed_char_count": trim_start_char,
            "source_accepted_char_count": len(accepted_text),
            "source_accepted_token_count": accepted_end_tok - accepted_start_tok,
            "projected_accepted_char_count": len(projected_text),
            "projected_accepted_token_count": projected_token_count,
        },
    )
