from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_uq.prompts.few_shot import load_few_shot
from agent_uq.prompts.react_instruction import react_instruction
@dataclass
class HistoryTurn:


    thought: str
    action: str
    observation: str


@dataclass
class TaskRecord:


    record_id: str
    dataset_key: str
    adapter_name: str
    target_type: str
    question: str

_BLOCK_SEPARATOR = "\n\n"


def build(
    system: str,
    task_record: TaskRecord,
    history: Sequence[HistoryTurn],
    *,
    prompt_context: object | None = None,
    num_shots: int | None = None,
) -> str:

    blocks: list[str] = []


    system_block = system.rstrip()
    if system_block:
        blocks.append(system_block)


    instruction = react_instruction(
        adapter_name=task_record.adapter_name,
        dataset_key=task_record.dataset_key,
    )
    blocks.append(instruction.rstrip())


    fewshot = load_few_shot(
        task_record.adapter_name,
        task_record.dataset_key,
        num_shots=num_shots,
    )
    fewshot = fewshot.rstrip()
    if fewshot:
        blocks.append(fewshot)
    else:
        blocks.append("")


    history_block = _format_history(task_record, history)
    blocks.append(history_block)

    return _BLOCK_SEPARATOR.join(blocks)


def append_turn(
    history: Sequence[HistoryTurn],
    thought: str,
    action: str,
    observation: str,
) -> list[HistoryTurn]:

    return list(history) + [
        HistoryTurn(thought=thought, action=action, observation=observation)
    ]


def _format_history(task: TaskRecord, history: Sequence[HistoryTurn]) -> str:

    lines: list[str] = []
    lines.append(f"Question: {task.question}")
    next_turn_idx = 1
    for turn in history:
        lines.append("")
        if _is_observation_prologue(turn):
            lines.append(f"Observation 0: {turn.observation}")
            continue
        lines.extend(_render_numbered_turn(next_turn_idx, turn))
        next_turn_idx += 1
    lines.append("")
    lines.append(f"Thought {next_turn_idx}:")
    return "\n".join(lines)


def _is_observation_prologue(turn: HistoryTurn) -> bool:

    return turn.thought == "" and turn.action == ""


def _render_numbered_turn(turn_idx: int, turn: HistoryTurn) -> list[str]:

    thought_lines = (turn.thought or "").splitlines()
    if thought_lines:
        rendered = [f"Thought {turn_idx}: {thought_lines[0]}"]
        rendered.extend(thought_lines[1:])
    else:
        rendered = [f"Thought {turn_idx}:"]
    rendered.append(f"Action {turn_idx}: {turn.action}")
    rendered.append(f"Observation {turn_idx}: {turn.observation}")
    return rendered
