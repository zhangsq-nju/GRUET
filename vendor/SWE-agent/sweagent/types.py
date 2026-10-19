from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class StepOutput(BaseModel):
    query: list[dict] = [{}]
    prompt: str = ""
    thought: str = ""
    action: str = ""
    output: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False
    exit_status: int | str | None = None
    submission: str | None = None
    state: dict[str, str] = {}
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_ids: list[str] | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    selected_idx: int | None = None
    sampling: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []


    extra_info: dict[str, Any] = {}

    def to_template_format_dict(self) -> dict[str, str | int | float | bool | None]:

        out = {}
        for k, v in self.model_dump().items():
            if k in ("tool_calls", "tool_call_ids", "state"):
                continue
            out[k] = v
        out |= self.state
        return out


class TrajectoryStep(TypedDict, total=False):
    action: str
    candidates: list[dict[str, Any]]
    observation: str
    prompt: str
    query: list[dict[str, Any]]
    response: str
    sampling: dict[str, Any]
    selected_idx: int | None
    state: dict[str, str]
    thought: str
    execution_time: float
    extra_info: dict[str, Any]


class _HistoryItem(TypedDict):
    role: str
    content: str | list[dict[str, Any]]
    message_type: Literal["thought", "action", "observation"]


class HistoryItem(_HistoryItem, total=False):
    agent: str
    is_demo: bool
    thought: str
    action: str | None
    tool_calls: list[dict[str, str]] | None
    tool_call_ids: list[str] | None
    tags: list[str]
    cache_control: dict[str, Any] | None
    thinking_blocks: list[dict[str, Any]] | None


History = list[HistoryItem]
Trajectory = list[TrajectoryStep]


class AgentInfo(TypedDict, total=False):

    model_stats: dict[str, float]
    exit_status: str | None
    submission: str | None

    review: dict[str, Any]
    edited_files30: str
    edited_files50: str
    edited_files70: str

    summarizer: dict
    swe_agent_hash: str
    swe_agent_version: str
    swe_rex_version: str
    swe_rex_hash: str


class AgentRunResult(BaseModel):
    info: AgentInfo
    trajectory: Trajectory
