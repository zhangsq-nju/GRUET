from __future__ import annotations

import pathlib
import re

_REGISTRY_DIR = pathlib.Path(__file__).resolve().parent
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_QUESTION_MARKER_RE = re.compile(r"(?m)^Question:\s")
_DATASET_ALIASES = {
    ("nl2bash", "nl2bash"): "nl2bash_fs1",
}


def load_few_shot(
    adapter_name: str,
    dataset_key: str | None,
    *,
    num_shots: int | None = None,
) -> str:

    candidates: list[pathlib.Path] = []
    if dataset_key:
        candidates.append(_REGISTRY_DIR / f"{adapter_name}_{dataset_key}.md")
        alias = _DATASET_ALIASES.get((adapter_name, dataset_key))
        if alias:
            candidates.append(_REGISTRY_DIR / f"{adapter_name}_{alias}.md")
    candidates.append(_REGISTRY_DIR / f"{adapter_name}.md")

    for path in candidates:
        if path.is_file():
            return _slice_examples(
                _normalize_registry_text(path.read_text(encoding="utf-8")),
                num_shots=num_shots,
            )
    return ""


def _normalize_registry_text(text: str) -> str:

    return _HTML_COMMENT_RE.sub("", text).strip()


def _slice_examples(text: str, *, num_shots: int | None) -> str:

    if not text:
        return ""
    if num_shots is not None and num_shots <= 0:
        return ""
    examples = _split_examples(text)
    if not examples:
        return ""
    if num_shots is None:
        selected = examples
    else:
        selected = examples[:num_shots]
    return "\n\n".join(selected).strip()


def _split_examples(text: str) -> list[str]:

    matches = list(_QUESTION_MARKER_RE.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    out: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        example = text[start:end].strip()
        if example:
            out.append(example)
    return out
