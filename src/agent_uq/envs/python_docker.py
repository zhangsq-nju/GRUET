from __future__ import annotations

import ast
import base64
import json
import pathlib
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable

from agent_uq.actions import action_payload, action_tool_name
from agent_uq.envs.base import ResetResult, StepResult, TaskRecord
from agent_uq.envs.docker_runtime import run_ephemeral_container


@dataclass
class PythonDockerEnvConfig:
    dataset_key: str
    dataset_path: str
    target_type: str
    terminal_action: str
    image: str = "intercode-python"
    step_timeout_s: float = 5.0
    score_timeout_s: float = 15.0
    per_case_timeout_s: float = 4.0


class PythonDockerEnv:
    adapter_name = "python"
    backend_name = "docker"

    def __init__(self, cfg: PythonDockerEnvConfig) -> None:
        self.cfg = cfg
        self._records: list[TaskRecord] = []
        self._executed_snippets: list[str] = []
        self._last_record: TaskRecord | None = None

    def load_records(self) -> Iterable[TaskRecord]:
        if self._records:
            return self._records
        path = pathlib.Path(self.cfg.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"PythonDockerEnv: dataset not found at {path}")
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(
                f"PythonDockerEnv: expected a list of records at {path}, got {type(raw)}"
            )
        provenance = {
            "backend": self.backend_name,
            "image": self.cfg.image,
            "dataset_key": self.cfg.dataset_key,
        }
        for i, item in enumerate(raw):
            rid_field = item.get("task_id", item.get("id", i))
            self._records.append(
                TaskRecord(
                    record_id=f"{self.cfg.dataset_key}_{rid_field}",
                    dataset_key=self.cfg.dataset_key,
                    adapter_name=self.adapter_name,
                    target_type=self.cfg.target_type,
                    question=item["query"],
                    env_backend=self.backend_name,
                    env_provenance=dict(provenance),
                    extras={
                        "raw": item,
                        "task_id": rid_field,
                        "tests": item.get("tests"),
                        "test_setup_code": item.get("test_setup_code", ""),
                        "input_output": item.get("input_output"),
                        "gold": item.get("gold", ""),
                    },
                )
            )
        return self._records

    def reset(self, record: TaskRecord) -> ResetResult:
        self._last_record = record
        self._executed_snippets = []
        return ResetResult(observation="")

    def step(self, action, action_payload_: str | None = None) -> StepResult:
        if self._last_record is None:
            raise RuntimeError("PythonDockerEnv.step called before reset")

        if isinstance(action, str):
            action_verb = action
            payload = action_payload_ or ""
        else:
            action_verb = action_tool_name(action)
            payload = action_payload(action)

        if action_verb == "execute":
            try:
                observation = self._exec_snippet(payload)
            except subprocess.TimeoutExpired:
                return StepResult(
                    observation=_render_exec_observation(
                        stdout_text="",
                        stderr_text="",
                        error_text=(
                            f"_ExecuteTimeout: execute snippet timed out after "
                            f"{self.cfg.step_timeout_s:.3f}s"
                        ),
                    ),
                    reward=0.0,
                    done=False,
                    info={"timeout": True, "backend_reset": True},
                )
            self._executed_snippets.append(payload)
            return StepResult(observation=observation, reward=0.0, done=False)

        if action_verb in {"submit", "submit_program"}:
            return StepResult(
                observation="<submission queued>",
                reward=0.0,
                done=True,
                info={"submission_verb": action_verb, "submission_payload": payload},
            )

        return StepResult(
            observation=f"<unknown action verb {action_verb!r}>",
            reward=0.0,
            done=False,
            info={"error": True},
        )

    def close(self) -> None:
        self._last_record = None
        self._executed_snippets = []

    def score(
        self, record: TaskRecord, history: list[dict[str, Any]]
    ) -> tuple[float, bool]:
        if not history:
            return 0.0, False

        last = history[-1]
        verb = last.get("verb")
        payload = last.get("payload", "")

        if self.cfg.dataset_key == "python_mbpp":
            return self._score_mbpp(record, verb, payload, history)
        if self.cfg.dataset_key == "python_apps":
            return self._score_apps(record, verb, payload)
        return 0.0, False

    def _score_mbpp(
        self,
        record: TaskRecord,
        verb: str | None,
        payload: str,
        history: list[dict[str, Any]],
    ) -> tuple[float, bool]:
        if verb != "submit":
            return 0.0, False
        tests: list[str] = list(record.extras.get("tests") or [])
        if not tests:
            return 0.0, False
        setup_code: str = record.extras.get("test_setup_code") or ""
        snippets = [
            entry["payload"]
            for entry in history
            if entry.get("verb") == "execute" and entry.get("payload")
        ]
        alias_code = _build_mbpp_function_alias(tests, payload, snippets)
        program_parts: list[str] = [setup_code]
        program_parts.extend(snippets)
        if alias_code:
            program_parts.append(alias_code)
        program_parts.append(
            textwrap.dedent(
                f"""
                _AUQ_PASS = 0
                _AUQ_FAIL = 0
                _AUQ_TESTS = {json.dumps(tests)}
                for _t in _AUQ_TESTS:
                    try:
                        exec(_t)
                        _AUQ_PASS += 1
                    except Exception:
                        _AUQ_FAIL += 1
                print(f"AGENT_UQ_RESULT pass={{_AUQ_PASS}} fail={{_AUQ_FAIL}}")
                """
            )
        )
        try:
            result = self._run_python_program(
                "\n".join(program_parts),
                timeout_s=self.cfg.score_timeout_s,
                stdin_text=None,
                wrap_execute_errors=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "PythonDockerEnv: MBPP scoring timed out after "
                f"{self.cfg.score_timeout_s:.3f}s"
            ) from exc

        marker = "AGENT_UQ_RESULT"
        last_line = next(
            (line for line in (result.stdout or "").splitlines() if line.startswith(marker)),
            "",
        )
        if not last_line:
            return 0.0, False
        try:
            kv = dict(part.split("=", 1) for part in last_line.split()[1:])
            n_pass = int(kv["pass"])
            n_fail = int(kv["fail"])
        except Exception:
            return 0.0, False
        total = n_pass + n_fail
        if total == 0:
            return 0.0, False
        solved = n_pass >= 1
        return (1.0 if solved else 0.0), solved

    def _score_apps(
        self,
        record: TaskRecord,
        verb: str | None,
        payload: str,
    ) -> tuple[float, bool]:
        if verb != "submit_program":
            return 0.0, False
        program = payload
        io_pair = record.extras.get("input_output") or {}
        inputs: list[str] = io_pair.get("inputs") or []
        outputs: list[str] = io_pair.get("outputs") or []
        if not inputs or len(inputs) != len(outputs):
            return 0.0, False

        n_pass = 0
        n_total = len(inputs)
        for inp, expected in zip(inputs, outputs):
            try:
                result = self._run_python_program(
                    program,
                    timeout_s=self.cfg.per_case_timeout_s,
                    stdin_text=inp,
                    wrap_execute_errors=False,
                )
            except subprocess.TimeoutExpired:
                continue
            actual = (result.stdout or "").rstrip("\n")
            expected_norm = (expected or "").rstrip("\n")
            if actual == expected_norm:
                n_pass += 1
        if n_total == 0:
            return 0.0, False
        rate = n_pass / n_total
        return rate, rate == 1.0

    def _exec_snippet(self, snippet: str) -> str:
        program = "\n".join([*self._executed_snippets, snippet])
        result = self._run_python_program(
            program,
            timeout_s=self.cfg.step_timeout_s,
            stdin_text=None,
            wrap_execute_errors=True,
        )
        return _render_exec_observation(
            stdout_text=result.stdout,
            stderr_text=result.stderr,
            error_text=None,
        )

    def _run_python_program(
        self,
        program: str,
        *,
        timeout_s: float,
        stdin_text: str | None,
        wrap_execute_errors: bool,
    ):
        encoded = base64.b64encode(program.encode("utf-8")).decode("ascii")
        if wrap_execute_errors:
            bootstrap = textwrap.dedent(
                f"""
                import base64
                import io
                import sys

                code = base64.b64decode("{encoded}").decode("utf-8")
                namespace = {{}}
                sys.stdin = io.TextIOWrapper(io.BytesIO(b""), encoding="utf-8")
                try:
                    exec(compile(code, "<execute>", "exec"), namespace)
                except EOFError:
                    sys.stderr.write(
                        "EOFError: execute snippets run without stdin; "
                        "use literal sample strings or helper functions in "
                        "execute[...] and reserve stdin-reading programs for "
                        "submit_program[...]"
                    )
                except SystemExit as exc:
                    code = exc.code if exc.code is not None else 0
                    sys.stderr.write(f"SystemExit: {{code}}")
                except Exception as exc:
                    sys.stderr.write(f"{{type(exc).__name__}}: {{exc}}")
                """
            )
        else:
            bootstrap = textwrap.dedent(
                f"""
                import base64

                code = base64.b64decode("{encoded}").decode("utf-8")
                exec(compile(code, "<agent_uq>", "exec"), {{}})
                """
            )
        return run_ephemeral_container(
            self.cfg.image,
            command=["python", "-c", bootstrap],
            timeout_s=timeout_s,
            input_text=stdin_text,
        )


def _build_mbpp_function_alias(
    tests: list[str],
    submit_payload: str,
    snippets: list[str],
) -> str:
    expected_name = _infer_mbpp_expected_function_name(tests)
    if not expected_name:
        return ""

    actual_name = _infer_mbpp_actual_function_name(submit_payload, snippets)
    if not actual_name or actual_name == expected_name:
        return ""

    if _is_identifier(actual_name):
        return f"{expected_name} = {actual_name}"

    if _is_lambda_expression(actual_name):
        return f"{expected_name} = {actual_name}"

    return ""


def _infer_mbpp_expected_function_name(tests: list[str]) -> str | None:
    names: set[str] = set()
    for test in tests:
        name = _assert_direct_call_name(test)
        if name is None:
            return None
        names.add(name)
    if len(names) != 1:
        return None
    return next(iter(names))


def _assert_direct_call_name(test: str) -> str | None:
    try:
        parsed = ast.parse(test)
    except SyntaxError:
        return None
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        return None
    expr = parsed.body[0].test
    if isinstance(expr, ast.Compare):
        expr = expr.left
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        return expr.func.id
    return None


def _infer_mbpp_actual_function_name(submit_payload: str, snippets: list[str]) -> str | None:
    payload = (submit_payload or "").strip()
    defined_names = _defined_function_names(snippets)
    if _is_identifier(payload) and (not defined_names or payload in defined_names):
        return payload
    if _is_lambda_expression(payload):
        return payload
    if defined_names:
        return defined_names[-1]
    return None


def _defined_function_names(snippets: list[str]) -> list[str]:
    names: list[str] = []
    for snippet in snippets:
        try:
            parsed = ast.parse(snippet)
        except SyntaxError:
            names.extend(
                re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", snippet, flags=re.M)
            )
            continue
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef):
                names.append(node.name)
    return names


def _is_identifier(value: str) -> bool:
    return bool(value) and value.isidentifier()


def _is_lambda_expression(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = ast.parse(value, mode="eval")
    except SyntaxError:
        return False
    return isinstance(parsed.body, ast.Lambda)


def _render_exec_observation(
    *,
    stdout_text: str,
    stderr_text: str,
    error_text: str | None = None,
) -> str:

    stdout_block = stdout_text.rstrip("\n")
    stderr_parts = [part.rstrip("\n") for part in (stderr_text, error_text or "") if part]
    if not stderr_parts:
        return stdout_block

    stderr_block = "\n".join(stderr_parts)
    if stdout_block:
        return f"{stdout_block}\n[stderr] {stderr_block}"
    return f"[stderr] {stderr_block}"
