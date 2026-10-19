from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from agent_uq.actions import action_payload, action_tool_name
from agent_uq.envs.base import ResetResult, StepResult, TaskRecord
from agent_uq.envs.docker_runtime import run_ephemeral_container


@dataclass
class NL2BashDockerEnvConfig:
    dataset_key: str
    target_type: str
    dataset_path: str | None = None
    dataset_splits: Iterable[Mapping[str, Any]] | None = None
    image: str = "intercode-nl2bash"
    shell: str = "/bin/bash"
    step_timeout_s: float = 5.0
    score_timeout_s: float = 5.0


class NL2BashDockerEnv:
    adapter_name = "nl2bash"
    backend_name = "docker"

    def __init__(self, cfg: NL2BashDockerEnvConfig) -> None:
        self.cfg = cfg
        self._records: list[TaskRecord] = []
        self._history_commands: list[str] = []
        self._last_record: TaskRecord | None = None

    def load_records(self) -> Iterable[TaskRecord]:
        if self._records:
            return self._records
        if self.cfg.dataset_splits:
            self._records = self._load_merged_records(self.cfg.dataset_splits)
            return self._records
        if self.cfg.dataset_path is None:
            raise ValueError(
                "NL2BashDockerEnv: either dataset_path or dataset_splits is required"
            )
        path = pathlib.Path(self.cfg.dataset_path)
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        provenance = {
            "backend": self.backend_name,
            "image": self.cfg.image,
            "dataset_key": self.cfg.dataset_key,
        }
        for i, item in enumerate(raw):
            self._records.append(
                TaskRecord(
                    record_id=f"{self.cfg.dataset_key}_{i:04d}",
                    dataset_key=self.cfg.dataset_key,
                    adapter_name=self.adapter_name,
                    target_type=self.cfg.target_type,
                    question=item["query"],
                    env_backend=self.backend_name,
                    env_provenance=dict(provenance),
                    extras={
                        "raw": item,
                        "gold_command": item["gold"],
                        "image": self.cfg.image,
                    },
                )
            )
        return self._records

    def _load_merged_records(
        self,
        dataset_splits: Iterable[Mapping[str, Any]],
    ) -> list[TaskRecord]:
        split_records: list[dict[str, Any]] = []
        for raw_split in dataset_splits:
            fs_id = str(raw_split["fs_id"])
            image = str(raw_split.get("image", self.cfg.image))
            path = pathlib.Path(str(raw_split["dataset_path"]))
            with path.open(encoding="utf-8") as fh:
                raw_records = json.load(fh)
            if not isinstance(raw_records, list):
                raise ValueError(
                    f"NL2BashDockerEnv: expected list records in {path}, "
                    f"got {type(raw_records).__name__}"
                )
            split_records.append(
                {
                    "fs_id": fs_id,
                    "image": image,
                    "path": str(path),
                    "records": raw_records,
                }
            )

        records: list[TaskRecord] = []
        max_len = max((len(split["records"]) for split in split_records), default=0)
        for source_index in range(max_len):
            for split in split_records:
                raw_records = split["records"]
                if source_index >= len(raw_records):
                    continue
                item = raw_records[source_index]
                fs_id = split["fs_id"]
                image = split["image"]
                provenance = {
                    "backend": self.backend_name,
                    "image": image,
                    "dataset_key": self.cfg.dataset_key,
                    "fs_id": fs_id,
                    "source_index": source_index,
                    "dataset_path": split["path"],
                }
                records.append(
                    TaskRecord(
                        record_id=f"{self.cfg.dataset_key}_{fs_id}_{source_index:04d}",
                        dataset_key=self.cfg.dataset_key,
                        adapter_name=self.adapter_name,
                        target_type=self.cfg.target_type,
                        question=item["query"],
                        env_backend=self.backend_name,
                        env_provenance=dict(provenance),
                        extras={
                            "raw": item,
                            "gold_command": item["gold"],
                            "fs_id": fs_id,
                            "source_index": source_index,
                            "image": image,
                        },
                    )
                )
        return records

    def reset(self, record: TaskRecord) -> ResetResult:
        self._last_record = record
        self._history_commands = []
        return ResetResult(observation="<docker-backed testbed ready>")

    def step(self, action, action_payload_: str | None = None) -> StepResult:
        if self._last_record is None:
            return StepResult(observation="<no testbed>", reward=0.0, done=False, info={"error": True})
        if isinstance(action, str):
            action_verb = action
            payload = action_payload_ or ""
        else:
            action_verb = action_tool_name(action)
            payload = action_payload(action)
        if action_verb == "execute":
            try:
                observation, returncode = self._exec_bash(payload)
            except subprocess.TimeoutExpired:
                return StepResult(
                    observation="<timeout; docker runtime reset>",
                    reward=0.0,
                    done=False,
                    info={"timeout": True, "backend_reset": True},
                )
            self._history_commands.append(payload)
            return StepResult(
                observation=observation,
                reward=0.0,
                done=False,
                info={"returncode": returncode},
            )
        if action_verb == "submit":
            return StepResult(observation="<submission queued>", reward=0.0, done=True)
        return StepResult(observation=f"<unknown verb {action_verb}>", reward=0.0, done=False, info={"error": True})

    def close(self) -> None:
        self._last_record = None
        self._history_commands = []

    def score(self, record: TaskRecord, history: list[dict[str, Any]]) -> tuple[float, bool]:
        gold = record.extras.get("gold_command", "")
        if not gold:
            return 0.0, False

        agent_output: str | None = None
        for entry in reversed(history):
            if entry.get("verb") == "execute":
                agent_output = entry.get("observation", "")
                break
        if agent_output is None:
            return 0.0, False

        try:
            gold_output, _ = self._run_script(
                gold,
                timeout_s=self.cfg.score_timeout_s,
                image=self._record_image(record),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "NL2BashDockerEnv: scoring timed out after "
                f"{self.cfg.score_timeout_s:.3f}s"
            ) from exc

        if self._normalize_output(gold_output) == self._normalize_output(agent_output):
            return 1.0, True
        return 0.0, False

    def _exec_bash(self, command: str) -> tuple[str, int]:
        replay_lines = ["set +e"]
        for previous in self._history_commands:
            replay_lines.append(f"{{ {previous}; }} >/dev/null 2>&1")
        replay_lines.append(command)
        output, returncode = self._run_script(
            "\n".join(replay_lines),
            timeout_s=self.cfg.step_timeout_s,
            image=self._record_image(self._last_record),
        )
        return self._normalize_output(output), returncode

    def _run_script(
        self,
        script: str,
        *,
        timeout_s: float,
        image: str | None = None,
    ) -> tuple[str, int]:
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        bootstrap = textwrap.dedent(
            f"""
            import base64
            import subprocess
            import sys

            script = base64.b64decode("{encoded}").decode("utf-8")
            result = subprocess.run(
                [{self.cfg.shell!r}, "-lc", script],
                capture_output=True,
                text=True,
                check=False,
            )
            sys.stdout.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
            raise SystemExit(result.returncode)
            """
        )
        result = run_ephemeral_container(
            image or self.cfg.image,
            command=["python3", "-c", bootstrap],
            timeout_s=timeout_s,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        return combined, result.returncode

    def _record_image(self, record: TaskRecord | None) -> str:
        if record is None:
            return self.cfg.image
        image = record.extras.get("image") or record.env_provenance.get("image")
        return str(image or self.cfg.image)

    def _normalize_output(self, text: str) -> str:
        cleaned = (text or "").strip()
        return cleaned or "<no output>"
