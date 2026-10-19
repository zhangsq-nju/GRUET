from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final
import re


@dataclass
class ParsedAction:
    verb: str
    payload: str


@dataclass
class ParsedCompletion:
    parse_status: str
    parsed_thought: str | None = None
    parsed_action: ParsedAction | None = None
    parse_error: str | None = None
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedCurrentTurn:
    thought_block: str
    verb: str
    payload: str
    accepted_span_start: int
    accepted_span_end: int
    action_marker_start: int
    action_marker_end: int


PARSE_OK: Final[str] = "ok"
PARSE_INVALID_FORMAT: Final[str] = "invalid_format"


_VERBS_BRACKETED = ("submit_program", "execute", "submit")
_VERBS_BARE = ("submit",)


_THOUGHT_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^\s*Thought(?:\s+\d+)?:\s*")
_ACTION_LINE_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^[ \t]*Action(?:\s+\d+)?:")
_STEP_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^[ \t]*Step\s+\d+\s*:")


def _strip_optional_thought_header(text: str) -> str:

    return _THOUGHT_HEADER_RE.sub("", text, count=1).strip()


def _find_action_markers(text: str) -> list[tuple[int, int]]:

    return [(m.start(), m.end()) for m in _ACTION_LINE_RE.finditer(text)]


def _scan_balanced_bracket(text: str, start: int) -> int | None:

    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def recover_accepted_current_turn(
    text: str | None,
) -> tuple[AcceptedCurrentTurn | None, str | None]:

    if text is None:
        return (None, "completion text is None")

    markers = _find_action_markers(text)
    if not markers:
        return (None, "no line-level 'Action:' marker found in completion")

    last_error: str | None = None
    for marker_start, marker_end in markers:
        thought_block = _strip_optional_thought_header(text[:marker_start].rstrip())
        if not thought_block:
            last_error = "completion did not contain a non-empty Thought block before Action"
            continue

        action_tail_raw = text[marker_end:]
        action_tail = action_tail_raw.lstrip()
        action_tail_offset = marker_end + (len(action_tail_raw) - len(action_tail))

        verb: str | None = None
        payload: str | None = None
        accepted_span_end: int | None = None

        for v in _VERBS_BRACKETED:
            if action_tail.startswith(v):
                after_verb = action_tail[len(v):]
                stripped = after_verb.lstrip()
                ws_consumed = len(after_verb) - len(stripped)
                if stripped.startswith("["):
                    bracket_open = len(v) + ws_consumed
                    close = _scan_balanced_bracket(action_tail, bracket_open)
                    if close is None:
                        last_error = (
                            f"verb {v!r} opened with '[' at offset {bracket_open} "
                            "but no matching ']' was found"
                        )
                        break
                    verb = v
                    payload = action_tail[bracket_open + 1:close]
                    accepted_span_end = action_tail_offset + close + 1
                    break

        if verb is None:
            for v in _VERBS_BARE:
                if action_tail.startswith(v):
                    after = action_tail[len(v):]
                    if after == "" or after[0] in " \t\r\n":
                        verb = v
                        payload = ""
                        accepted_span_end = action_tail_offset + len(v)
                        break

        if verb is None or accepted_span_end is None:
            if last_error is None:
                last_error = (
                    f"could not match any known verb after Action marker; "
                    f"saw {action_tail[:60]!r}"
                )
            continue

        if payload is not None and verb != "submit_program":
            payload = payload.strip()
        elif payload is not None:
            payload = payload.lstrip("\n").rstrip()

        return (
            AcceptedCurrentTurn(
                thought_block=thought_block,
                verb=verb,
                payload=payload or "",
                accepted_span_start=0,
                accepted_span_end=accepted_span_end,
                action_marker_start=marker_start,
                action_marker_end=marker_end,
            ),
            None,
        )

    return (None, last_error or "no valid current-turn Action block found in completion")


def _find_first_valid_current_turn_action(
    text: str | None,
) -> tuple[AcceptedCurrentTurn | None, str | None]:
    return recover_accepted_current_turn(text)


def find_current_turn_span(text: str | None) -> tuple[int, int] | None:

    match, _ = recover_accepted_current_turn(text)
    if match is None:
        return None
    return (match.accepted_span_start, match.accepted_span_end)


def parse_completion(text: str) -> ParsedCompletion:

    match, error = recover_accepted_current_turn(text)
    if match is None:
        return ParsedCompletion(
            parse_status=PARSE_INVALID_FORMAT,
            parse_error=error or "no valid current-turn Action block found in completion",
        )
    return ParsedCompletion(
        parse_status=PARSE_OK,
        parsed_thought=match.thought_block,
        parsed_action=ParsedAction(verb=match.verb, payload=match.payload),
    )


def is_compliant_candidate(
    parsed: ParsedCompletion,
    raw_text: str,
) -> tuple[bool, str | None]:

    if parsed.parse_status != PARSE_OK:
        return (False, "parse_not_ok")

    if raw_text is None:
        return (False, "missing_step_marker")

    markers = _find_action_markers(raw_text)
    if not markers:


        return (False, "missing_step_marker")

    first_action_start = markers[0][0]
    pre_action = raw_text[:first_action_start]
    if _STEP_MARKER_RE.search(pre_action) is None:
        return (False, "missing_step_marker")

    return (True, None)
