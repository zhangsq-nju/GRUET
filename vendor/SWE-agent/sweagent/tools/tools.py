import asyncio
import json
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from swerex.runtime.abstract import Command as RexCommand
from swerex.runtime.abstract import UploadRequest
from typing_extensions import Self

from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import FormatError
from sweagent.tools.bundle import Bundle
from sweagent.tools.commands import BASH_COMMAND, Command
from sweagent.tools.parsing import (
    FunctionCallingParser,
    JsonParser,
    ParseFunction,
    ThoughtActionParser,
    has_step_marker,
)
from sweagent.tools.utils import _guard_multiline_input, generate_command_docs
from sweagent.utils.log import get_logger


class ToolFilterConfig(BaseModel):


    blocklist_error_template: str = "Operation '{{action}}' is not supported by this environment."


    blocklist: list[str] = [
        "vim",
        "vi",
        "emacs",
        "nano",
        "nohup",
        "gdb",
        "less",
        "tail -f",
        "python -m venv",
        "make",
    ]


    blocklist_standalone: list[str] = [
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "/bin/bash",
        "/bin/sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
        "su",
    ]


    block_unless_regex: dict[str, str] = {
        "radare2": r"\b(?:radare2)\b.*\s+-c\s+.*",
        "r2": r"\b(?:radare2)\b.*\s+-c\s+.*",
    }


class ToolConfig(BaseModel):


    filter: ToolFilterConfig = ToolFilterConfig()


    bundles: list[Bundle] = Field(default_factory=list)


    propagate_env_variables: list[str] = []


    env_variables: dict[str, Any] = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
    }


    registry_variables: dict[str, Any] = {}


    submit_command: str = "submit"


    parse_function: ParseFunction = Field(default_factory=FunctionCallingParser)


    require_step_marker_in_thought: bool = False


    enable_bash_tool: bool = True


    format_error_template: str = None


    command_docs: str = None


    multi_line_command_endings: dict[str, str] = {}
    submit_command_end_name: str | None = None


    reset_commands: list[str | list[str]] = []


    execution_timeout: int = 30


    install_timeout: int = 300


    total_execution_timeout: int = 1800


    max_consecutive_execution_timeouts: int = 3


    @cached_property
    def use_function_calling(self) -> bool:
        return isinstance(self.parse_function, FunctionCallingParser)

    @cached_property
    def state_commands(self) -> list[str]:

        return [bundle.state_command for bundle in self.bundles if bundle.state_command]


    @cached_property
    def commands(self) -> list[Command]:

        commands = []
        tool_sources: dict[str, Path] = {}

        if self.enable_bash_tool:
            commands.append(BASH_COMMAND)
            tool_sources[BASH_COMMAND.name] = Path("<builtin>")


        for bundle in self.bundles:
            for command in bundle.commands:
                if command.name in tool_sources:
                    existing_source = tool_sources[command.name]
                    msg = (
                        f"Tool '{command.name}' is defined multiple times:\n"
                        f"  - First definition in: {existing_source}\n"
                        f"  - Duplicate definition in: {bundle.path}"
                    )
                    raise ValueError(msg)
                commands.append(command)
                tool_sources[command.name] = bundle.path

        return commands

    @cached_property
    def tools(self) -> list[dict]:
        return [command.get_function_calling_tool() for command in self.commands]


    def model_post_init(self, __context):

        commands = self.commands
        multi_line_command_endings = {
            command.name: command.end_name for command in commands if command.end_name is not None
        }
        self.tools


        if not self.enable_bash_tool and not (
            isinstance(self.parse_function, FunctionCallingParser) or isinstance(self.parse_function, JsonParser)
        ):
            msg = f"Bash tool can only be disabled if {FunctionCallingParser.type} parser or {JsonParser.type} parser is used."
            raise ValueError(msg)

        self.multi_line_command_endings = multi_line_command_endings
        self.command_docs = generate_command_docs(
            self.commands,
            [],
            **self.env_variables,
        )
        if self.format_error_template is None:
            self.format_error_template = self.parse_function.format_error_template
        for command in commands:
            if command.name == self.submit_command:
                self.submit_command_end_name = command.end_name
                break


class ToolHandler:
    def __init__(self, tools: ToolConfig):


        self.config = tools.model_copy(deep=True)

        self._reset_commands = []
        self._command_patterns = self._get_command_patterns()
        self.logger = get_logger("swea-tools", emoji="🧰")

    @classmethod
    def from_config(cls, config: ToolConfig) -> Self:
        return cls(config)


    def install(self, env: SWEEnv) -> None:
        self._install_commands(env)
        self.reset(env)

    def reset(self, env: SWEEnv) -> None:
        self.logger.info("Resetting tools")
        env_variables = self.config.env_variables.copy() | {
            var: os.getenv(var) for var in self.config.propagate_env_variables
        }
        env.set_env_variables(env_variables)
        env.write_file("/root/.swe-agent-env", json.dumps(self.config.registry_variables))
        env.write_file("/root/state.json", "{}")
        env.communicate(" && ".join(self._reset_commands), check="raise", timeout=self.config.install_timeout)

    async def _upload_bundles(self, env: SWEEnv) -> None:
        await asyncio.gather(
            *(
                env.deployment.runtime.upload(
                    UploadRequest(source_path=bundle.path.as_posix(), target_path=f"/root/tools/{bundle.path.name}")
                )
                for bundle in self.config.bundles
            )
        )

    async def _is_command_available(self, env, command: str, env_vars: dict[str, str]) -> None:
        if command == "bash":
            return
        try:
            await env.deployment.runtime.execute(
                RexCommand(command=f"which {command}", shell=True, check=True, env=env_vars)
            )
        except Exception:
            msg = f"Tool {command} is not available in the container."
            raise RuntimeError(msg) from None

    async def _check_available_commands(self, env: SWEEnv, env_vars: dict[str, str]) -> None:
        await asyncio.gather(
            *(self._is_command_available(env, command.name, env_vars) for command in self.config.commands)
        )

    def _install_commands(self, env: SWEEnv) -> None:

        env.set_env_variables(self.config.env_variables)
        cwd = env.communicate("pwd", check="raise").strip()
        asyncio.run(self._upload_bundles(env))
        for bundle in self.config.bundles:
            cmds = [
                f"export PATH=/root/tools/{bundle.path.name}/bin:$PATH",
                f"chmod +x /root/tools/{bundle.path.name}/bin/*",
            ]
            if (bundle.path / "install.sh").exists():
                cmds.append(f"cd /root/tools/{bundle.path.name} && source install.sh")
            cmds.append(f"chmod +x /root/tools/{bundle.path.name}/bin/*")
            env.communicate(
                " && ".join(cmds),
                check="raise",
                timeout=self.config.install_timeout,
            )
        env.communicate(f"cd {cwd}", check="raise")
        path = env.communicate("echo $PATH", check="raise").strip()
        asyncio.run(self._check_available_commands(env, {"PATH": path}))


    def _get_state(self, env: SWEEnv) -> dict[str, str]:

        try:
            state_str = env.read_file("/root/state.json")
        except FileNotFoundError:
            self.logger.warning("State file not found, returning empty state")
            return {}
        if not state_str.strip():
            self.logger.warning("State file is empty, returning empty state")
            return {}
        try:
            state = json.loads(state_str)
        except json.JSONDecodeError as e:
            msg = f"State {state_str!r} is not valid json. This is an internal error, please report it."
            raise ValueError(msg) from e
        if not isinstance(state, dict):
            msg = f"State commands must return a dictionary. Got {state!r} instead."
            raise ValueError(msg)
        return state

    def get_state(self, env: SWEEnv) -> dict[str, str]:

        for state_command in self.config.state_commands:
            env.communicate(state_command, check="warn")
        combined_state = self._get_state(env)
        self.logger.debug(f"Retrieved state from environment: {combined_state}")
        return combined_state


    def should_block_action(self, action: str) -> bool:

        action = action.strip()
        if not action:
            return False
        if any(action.startswith(f) for f in self.config.filter.blocklist):
            return True
        if action in self.config.filter.blocklist_standalone:
            return True
        name = action.split()[0]
        if name in self.config.filter.block_unless_regex and not re.search(
            self.config.filter.block_unless_regex[name], action
        ):
            return True
        return False


    def check_for_submission_cmd(self, output: str) -> bool:

        if r"<<SWE_AGENT_SUBMISSION>>" in output:
            return True
        return False

    def parse_actions(self, output: dict) -> tuple[str, str]:

        analysis = self.analyze_candidate(output)
        if analysis["parse_status"] != "ok":
            msg = str(analysis["parse_error"] or "No action found in model response.")
            raise FormatError(msg)
        if not analysis["compliant"]:
            msg = "Reasoning is missing a line-start Step <n>: marker."
            raise FormatError(msg)
        return str(analysis["thought"]), str(analysis["action"])

    def analyze_candidate(self, output: dict[str, Any]) -> dict[str, Any]:

        raw_message = str(output.get("raw_response") or output.get("message") or "")
        parse_input = dict(output)
        parse_input["message"] = raw_message
        try:
            thought, action = self.config.parse_function(parse_input, self.config.commands)
        except FormatError as exc:
            parse_error = getattr(exc, "message", "") or (exc.args[0] if exc.args else "format_error")
            return {
                "parse_status": "invalid_format",
                "parse_error": str(parse_error),
                "thought": None,
                "action": None,
                "compliant": False,
                "compliance_reason": "parse_not_ok",
            }

        compliant = True
        compliance_reason = None
        if self.config.require_step_marker_in_thought and isinstance(
            self.config.parse_function, ThoughtActionParser
        ):
            compliant = has_step_marker(thought)
            if not compliant:
                compliance_reason = "missing_step_marker"

        return {
            "parse_status": "ok",
            "parse_error": None,
            "thought": thought,
            "action": action,
            "compliant": compliant,
            "compliance_reason": compliance_reason,
        }

    def guard_multiline_input(self, action: str) -> str:

        return _guard_multiline_input(action, self._get_first_multiline_cmd)

    def _get_first_multiline_cmd(self, action: str) -> re.Match | None:

        patterns = {
            k: v
            for k, v in self._command_patterns.items()
            if k in self.config.multi_line_command_endings or k == self.config.submit_command
        }
        matches = list()
        for _, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]

    def _get_command_patterns(self) -> dict[str, re.Pattern]:


        _command_patterns = {}
        for command in self.config.commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                _command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                _command_patterns[command.name] = pat
        submit_pat = re.compile(
            rf"^\s*({self.config.submit_command})\s*(.*?)^({self.config.submit_command_end_name})\s*$",
            re.DOTALL | re.MULTILINE,
        )
        _command_patterns[self.config.submit_command] = submit_pat
        return _command_patterns
