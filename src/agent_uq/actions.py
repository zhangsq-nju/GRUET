from __future__ import annotations

import json
from typing import Any, Mapping


ActionDict = dict[str, Any]


_SCALAR_ARG_KEYS = (
    "command",
    "payload",
    "function_name",
    "program",
    "sql",
    "query",
    "text",
)


def build_action(
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    text: str | None = None,
    kind: str = "tool",
    parser_type: str | None = None,
    tool_call_id: str | None = None,
    raw_tool_call: dict[str, Any] | None = None,
    end_name: str | None = None,
    backend_source: str | None = None,
) -> ActionDict:

    args = dict(arguments or {})
    action: ActionDict = {
        "kind": kind,
        "tool_name": tool_name,
        "arguments": args,
    }
    if parser_type is not None:
        action["parser_type"] = parser_type
    if tool_call_id is not None:
        action["tool_call_id"] = tool_call_id
    if raw_tool_call is not None:
        action["raw_tool_call"] = dict(raw_tool_call)
    if end_name is not None:
        action["end_name"] = end_name
    if backend_source is not None:
        action["backend_source"] = backend_source

    action["text"] = text or render_action(action)
    action["verb"] = tool_name
    action["payload"] = _derive_payload(tool_name, args, action["text"])
    return action


def normalize_action(action: Mapping[str, Any] | None) -> ActionDict | None:

    if action is None:
        return None
    if not isinstance(action, Mapping):
        raise TypeError(
            f"normalize_action expected a mapping or None; got {type(action).__name__}"
        )

    if "tool_name" in action or "kind" in action:
        normalized = dict(action)
        normalized.setdefault("kind", "tool")
        if "tool_name" not in normalized:
            normalized["tool_name"] = str(normalized.get("verb", ""))
        args = normalized.get("arguments")
        if not isinstance(args, Mapping):
            args = _legacy_arguments(
                str(normalized.get("tool_name", "")),
                str(normalized.get("payload", "")),
            )
        normalized["arguments"] = dict(args)
        normalized["text"] = str(
            normalized.get("text") or render_action(normalized)
        )
        normalized["verb"] = str(
            normalized.get("verb") or normalized.get("tool_name", "")
        )
        normalized["payload"] = _derive_payload(
            str(normalized["tool_name"]),
            normalized["arguments"],
            normalized["text"],
        )
        return normalized

    if "verb" in action or "payload" in action:
        verb = str(action.get("verb", ""))
        payload = str(action.get("payload", ""))
        normalized = build_action(
            verb,
            _legacy_arguments(verb, payload),
            text=_legacy_text(verb, payload),
            parser_type=str(action.get("parser_type", "legacy_text")),
        )
        for key in (
            "tool_call_id",
            "raw_tool_call",
            "end_name",
            "backend_source",
        ):
            if key in action:
                normalized[key] = action[key]
        return normalized

    raise ValueError(
        "normalize_action expected either normalized keys "
        "(`kind`, `tool_name`, `arguments`) or legacy keys (`verb`, `payload`)"
    )


def render_action(action: Mapping[str, Any] | None) -> str:

    if action is None:
        return ""
    if "text" in action and action["text"]:
        return str(action["text"])

    tool_name = str(action.get("tool_name") or action.get("verb") or "")
    arguments = action.get("arguments")
    if isinstance(arguments, Mapping):
        payload = _derive_payload(tool_name, arguments, "")
        return _legacy_text(tool_name, payload)

    payload = str(action.get("payload", ""))
    return _legacy_text(tool_name, payload)


def action_tool_name(action: Mapping[str, Any]) -> str:
    return str(normalize_action(action)["tool_name"])


def action_payload(action: Mapping[str, Any]) -> str:
    return str(normalize_action(action)["payload"])


def action_arguments(action: Mapping[str, Any]) -> dict[str, Any]:
    return dict(normalize_action(action)["arguments"])


def _legacy_arguments(verb: str, payload: str) -> dict[str, Any]:
    if verb in {"execute", "bash"}:
        return {"command": payload}
    if verb == "submit":
        return {} if not payload else {"function_name": payload}
    if verb == "submit_program":
        return {"program": payload}
    return {} if not payload else {"payload": payload}


def _legacy_text(verb: str, payload: str) -> str:
    if not payload:
        return verb
    return f"{verb}[{payload}]"


def _derive_payload(tool_name: str, arguments: Mapping[str, Any], text: str) -> str:
    for key in _SCALAR_ARG_KEYS:
        if key in arguments and isinstance(arguments[key], str):
            return arguments[key]
    if tool_name == "submit" and not arguments:
        return ""
    if not arguments:
        return ""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
