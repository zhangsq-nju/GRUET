import asyncio
import logging
import shlex
from pathlib import PurePath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import DeploymentConfig, DockerDeploymentConfig, get_deployment
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    CreateBashSessionRequest,
    ReadFileRequest,
    WriteFileRequest,
)
from swerex.runtime.abstract import Command as RexCommand

from sweagent.environment.hooks.abstract import CombinedEnvHooks, EnvHook
from sweagent.environment.repo import Repo, RepoConfig
from sweagent.utils.log import get_logger


class EnvironmentConfig(BaseModel):


    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11", python_standalone_dir="/root"),
        description="Deployment options.",
    )
    repo: RepoConfig | None = Field(
        default=None,
        description="Repository options.",
    )
    post_startup_commands: list[str] = []

    post_startup_command_timeout: int = 500


    model_config = ConfigDict(extra="forbid")

    name: str = "main"


class SWEEnv:
    def __init__(
        self,
        *,
        deployment: AbstractDeployment,
        repo: Repo | RepoConfig | None,
        post_startup_commands: list[str],
        post_startup_command_timeout: int = 500,
        hooks: list[EnvHook] | None = None,
        name: str = "main",
    ):

        super().__init__()
        self.deployment = deployment
        self.repo = repo
        self._post_startup_commands = post_startup_commands
        self.post_startup_command_timeout = post_startup_command_timeout
        self.logger = get_logger("swea-env", emoji="🪴")
        self.name = name
        self.clean_multi_line_functions = lambda x: x
        self._chook = CombinedEnvHooks()
        for hook in hooks or []:
            self.add_hook(hook)

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> Self:


        config = config.model_copy(deep=True)
        return cls(
            deployment=get_deployment(config.deployment),
            repo=config.repo,
            post_startup_commands=config.post_startup_commands,
            post_startup_command_timeout=config.post_startup_command_timeout,
            name=config.name,
        )

    def add_hook(self, hook: EnvHook) -> None:

        hook.on_init(env=self)
        self._chook.add_hook(hook)

    def start(self) -> None:

        self._init_deployment()
        self.reset()
        for command in self._post_startup_commands:
            self.communicate(command, check="raise", timeout=self.post_startup_command_timeout)

    def _copy_repo(self) -> None:

        if self.repo is None:
            return

        folders = self.communicate(input="ls", check="raise").split("\n")
        if self.repo.repo_name in folders:
            return

        self._chook.on_copy_repo_started(repo=self.repo)
        self.repo.copy(self.deployment)

    def hard_reset(self):

        self.close()
        self.start()

    def reset(self):

        self.communicate(input="cd /", check="raise")
        self._copy_repo()
        self._reset_repository()
        self._chook.on_environment_startup()

    def _reset_repository(self) -> None:

        if self.repo is not None:
            self.logger.debug("Resetting repository %s to commit %s", self.repo.repo_name, self.repo.base_commit)


            startup_commands = [
                f"cd /{self.repo.repo_name}",
                "export ROOT=$(pwd -P)",
                'git config --global --add safe.directory "$ROOT"',
                *self.repo.get_reset_commands(),
            ]
            self.communicate(
                input=" && ".join(startup_commands),
                check="raise",
                error_msg="Failed to clean repository",

                timeout=120,
            )

    def close(self) -> None:

        self.logger.info("Beginning environment shutdown...")
        asyncio.run(self.deployment.stop())
        self._chook.on_close()


    def _init_deployment(
        self,
    ) -> None:

        self._chook.on_start_deployment()
        asyncio.run(self.deployment.start())
        asyncio.run(
            self.deployment.runtime.create_session(
                CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=10)
            )
        )
        self.set_env_variables({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PIP_PROGRESS_BAR": "off", "PAGER": "cat"})
        self.logger.info("Environment Initialized")

    def interrupt_session(self):
        self.logger.info("Interrupting session")
        asyncio.run(self.deployment.runtime.run_in_session(BashInterruptAction()))


    def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:

        self.logger.log(logging.TRACE, "Input:\n%s", input)
        rex_check = "silent" if check else "ignore"
        r = asyncio.run(
            self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        )
        output = r.output
        self.logger.log(logging.TRACE, "Output:\n%s", output)
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            self.logger.error(msg)
            if check == "raise":
                self.close()
                raise RuntimeError(msg)
        return output

    def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:

        r = asyncio.run(
            self.deployment.runtime.read_file(ReadFileRequest(path=str(path), encoding=encoding, errors=errors))
        )
        return r.content

    def write_file(self, path: str | PurePath, content: str) -> None:

        asyncio.run(self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content)))

    def set_env_variables(self, env_variables: dict[str, str]) -> None:

        if not env_variables:
            self.logger.debug("No environment variables to set")
            return
        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
        command = " && ".join(_env_setters)
        self.communicate(command, check="raise")

    def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:

        asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=shell, check=check, env=env, cwd=cwd))
        )
