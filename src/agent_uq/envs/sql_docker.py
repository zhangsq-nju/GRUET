from __future__ import annotations

import json
import pathlib
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable

from agent_uq.actions import action_payload, action_tool_name
from agent_uq.envs.base import ResetResult, StepResult, TaskRecord
from agent_uq.envs.docker_runtime import (
    DockerRuntimeError,
    PersistentDockerContainer,
    resolve_path,
)

_STARTUP_LOG_TAIL_LINES = 40


@dataclass
class SqlDockerEnvConfig:
    dataset_key: str
    dataset_path: str
    schema_sql_path: str
    target_type: str
    image: str = "mysql:latest"
    root_password: str = "password"
    step_timeout_s: float = 5.0
    score_timeout_s: float = 10.0
    startup_timeout_s: float = 120.0
    query_row_limit: int = 200
    lower_case_table_names: int = 1


class SqlDockerEnv:
    adapter_name = "sql"
    backend_name = "docker"

    def __init__(self, cfg: SqlDockerEnvConfig) -> None:
        self.cfg = cfg
        self._records: list[TaskRecord] = []
        self._container: PersistentDockerContainer | None = None
        self._current_db: str | None = None
        self._last_query_result: list[tuple[str, ...]] | None = None
        self._last_query_error: str | None = None

    def load_records(self) -> Iterable[TaskRecord]:
        if self._records:
            return self._records
        path = pathlib.Path(self.cfg.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"SqlDockerEnv: dataset not found at {path}")
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        provenance = {
            "backend": self.backend_name,
            "image": self.cfg.image,
            "dataset_key": self.cfg.dataset_key,
            "schema_sql_path": self.cfg.schema_sql_path,
        }
        for i, item in enumerate(raw):
            self._records.append(
                TaskRecord(
                    record_id=f"{self.cfg.dataset_key}_{i}",
                    dataset_key=self.cfg.dataset_key,
                    adapter_name=self.adapter_name,
                    target_type=self.cfg.target_type,
                    question=item["query"],
                    env_backend=self.backend_name,
                    env_provenance=dict(provenance),
                    extras={
                        "raw": item,
                        "db": item.get("db"),
                        "gold_sql": item.get("gold"),
                        "db_tables": item.get("db_tables"),
                        "hardness": item.get("hardness"),
                    },
                )
            )
        return self._records

    def reset(self, record: TaskRecord) -> ResetResult:
        self._restart_backend(db_name=record.extras.get("db"))
        self._last_query_result = None
        self._last_query_error = None
        tables = list((record.extras.get("db_tables") or {}).keys())
        return ResetResult(
            observation=f"Database `{self._current_db}` is selected. Tables: {tables}"
        )

    def step(self, action, action_payload_: str | None = None) -> StepResult:
        if self._container is None or self._current_db is None:
            return StepResult(observation="<not connected>", reward=0.0, done=False, info={"error": True})

        if isinstance(action, str):
            action_verb = action
            payload = action_payload_ or ""
        else:
            action_verb = action_tool_name(action)
            payload = action_payload(action)

        if action_verb == "execute":
            return self._exec_sql(payload)
        if action_verb == "submit":
            return StepResult(
                observation="<submission queued>",
                reward=0.0,
                done=True,
                info={"submission_verb": "submit"},
            )
        return StepResult(
            observation=f"<unknown verb {action_verb!r}>",
            reward=0.0,
            done=False,
            info={"error": True},
        )

    def close(self) -> None:
        if self._container is not None:
            self._container.stop()
        self._container = None
        self._current_db = None
        self._last_query_result = None
        self._last_query_error = None

    def score(self, record: TaskRecord, history: list[dict[str, Any]]) -> tuple[float, bool]:
        gold = record.extras.get("gold_sql")
        if not gold:
            return 0.0, False
        if self._last_query_result is None:
            return 0.0, False
        agent_result = list(self._last_query_result)

        db_name = record.extras.get("db")
        if not db_name:
            return 0.0, False
        self._restart_backend(db_name=db_name)
        try:
            gold_rows = self._query_rows(
                gold,
                db_name=db_name,
                timeout_s=self.cfg.score_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            self._restart_backend(db_name=db_name)
            raise RuntimeError(
                "SqlDockerEnv: authoritative scoring timed out after "
                f"{self.cfg.score_timeout_s:.3f}s"
            ) from exc

        agent_set = sorted(_canonicalize(row) for row in agent_result)
        gold_set = sorted(_canonicalize(row) for row in gold_rows)
        if agent_set == gold_set:
            return 1.0, True
        return 0.0, False

    def _restart_backend(self, db_name: str | None = None) -> None:
        self.close()
        schema_path = resolve_path(self.cfg.schema_sql_path)
        self._container = PersistentDockerContainer(
            image_name=self.cfg.image,
            name_prefix=f"agent-uq-sql-{self.cfg.dataset_key}",
            create_timeout_s=60.0,
            environment={"MYSQL_ROOT_PASSWORD": self.cfg.root_password},
            volumes=[(schema_path, "/docker-entrypoint-initdb.d/00-schema.sql", True)],
            command=[f"--lower_case_table_names={int(self.cfg.lower_case_table_names)}"],
        )
        self._container.start()
        self._wait_until_ready()
        self._current_db = db_name

    def _wait_until_ready(self) -> None:
        if self._container is None:
            raise DockerRuntimeError("SqlDockerEnv: container missing during startup")
        deadline = time.time() + self.cfg.startup_timeout_s
        while time.time() < deadline:
            probe_error: str | None = None
            try:
                result = self._container.exec(
                    [
                        "mysqladmin",
                        "ping",
                        "-h127.0.0.1",
                        "-uroot",
                        f"-p{self.cfg.root_password}",
                        "--silent",
                    ],
                    timeout_s=5.0,
                )
            except subprocess.TimeoutExpired as exc:
                probe_error = (
                    "mysqladmin ping timed out during startup after "
                    f"{float(exc.timeout or 0.0):.1f}s"
                )
            except DockerRuntimeError as exc:
                probe_error = str(exc)
            else:
                if result.returncode == 0:
                    return
                probe_error = (result.stderr or result.stdout or "").strip() or (
                    f"mysqladmin ping exited {result.returncode}"
                )
            startup_failure = self._startup_failure_if_exited(probe_error=probe_error)
            if startup_failure is not None:
                raise startup_failure
            time.sleep(1.0)
        raise DockerRuntimeError(
            "SqlDockerEnv: timed out waiting for MySQL startup inside docker"
        )

    def _startup_failure_if_exited(
        self,
        *,
        probe_error: str | None,
    ) -> DockerRuntimeError | None:
        if self._container is None:
            raise DockerRuntimeError("SqlDockerEnv: container missing during startup")
        state = self._container.inspect_state()
        if state.running or state.status in {"created", "restarting"}:
            return None
        diagnostics = [
            "SqlDockerEnv: MySQL container exited before readiness",
            f"status={state.status}",
        ]
        if state.exit_code is not None:
            diagnostics.append(f"exit_code={state.exit_code}")
        if probe_error:
            diagnostics.append(f"probe_error={probe_error}")
        if state.error:
            diagnostics.append(f"state_error={state.error}")
        logs_tail = self._container.logs_tail(lines=_STARTUP_LOG_TAIL_LINES)
        if logs_tail:
            diagnostics.append(
                f"logs_tail({int(_STARTUP_LOG_TAIL_LINES)} lines)=\n{logs_tail}"
            )
        return DockerRuntimeError(", ".join(diagnostics))

    def _exec_sql(self, sql: str) -> StepResult:
        sql_clean = sql.strip().rstrip(";")
        if not sql_clean:
            return StepResult(observation="<empty SQL>", reward=0.0, done=False)
        if self._current_db is None:
            return StepResult(observation="<no database selected>", reward=0.0, done=False, info={"error": True})
        try:
            rows = self._query_rows(
                sql_clean,
                db_name=self._current_db,
                timeout_s=self.cfg.step_timeout_s,
            )
        except subprocess.TimeoutExpired:
            self._last_query_result = None
            self._last_query_error = (
                f"SQL timeout after {self.cfg.step_timeout_s:.3f}s; docker backend reset"
            )
            self._restart_backend(db_name=self._current_db)
            return StepResult(
                observation=self._last_query_error,
                reward=0.0,
                done=False,
                info={"timeout": True, "backend_reset": True},
            )
        except DockerRuntimeError as exc:
            self._last_query_result = None
            self._last_query_error = str(exc)
            return StepResult(
                observation=f"SQL error: {exc}",
                reward=0.0,
                done=False,
                info={"error": True},
            )

        self._last_query_result = rows
        self._last_query_error = None
        preview = rows[: self.cfg.query_row_limit]
        return StepResult(
            observation=str(preview),
            reward=0.0,
            done=False,
            info={"row_count": len(rows)},
        )

    def _query_rows(
        self,
        sql: str,
        *,
        db_name: str,
        timeout_s: float,
    ) -> list[tuple[str, ...]]:
        if self._container is None:
            raise DockerRuntimeError("SqlDockerEnv: query requested before startup")
        result = self._container.exec(
            [
                "mysql",
                "-h127.0.0.1",
                "-uroot",
                f"-p{self.cfg.root_password}",
                "-D",
                db_name,
                "--batch",
                "--raw",
                "--skip-column-names",
            ],
            timeout_s=timeout_s,
            input_text=sql,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise DockerRuntimeError(stderr or "mysql command failed")
        rows: list[tuple[str, ...]] = []
        for line in (result.stdout or "").splitlines():
            if line == "":
                continue
            rows.append(tuple(line.split("\t")))
        return rows


def _canonicalize(row: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in row)
