from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol


@dataclass
class TaskRecord:


    record_id: str
    dataset_key: str
    adapter_name: str
    target_type: str
    question: str
    env_backend: str = "docker"
    env_provenance: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResetResult:
    observation: str
    initial_state: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.observation

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return item in self.observation

    def __getattr__(self, name: str):
        return getattr(self.observation, name)


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    state: dict[str, Any] | None = None
    tool_call_ids: list[str] | None = None
    info: dict[str, Any] = field(default_factory=dict)


class EnvAdapter(Protocol):


    adapter_name: str
    backend_name: str

    def load_records(self) -> Iterable[TaskRecord]: ...

    def reset(self, record: TaskRecord) -> ResetResult: ...

    def step(
        self,
        action: Mapping[str, Any] | str,
        action_payload: str | None = None,
    ) -> StepResult: ...

    def close(self) -> None: ...

    def score(self, record: TaskRecord, history: list[dict[str, Any]]) -> tuple[float, bool]: ...
