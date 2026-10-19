import asyncio
import os
import shlex
from pathlib import Path
from typing import Any, Literal, Protocol

from git import InvalidGitRepositoryError
from git import Repo as GitRepo
from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.abstract import AbstractDeployment
from swerex.runtime.abstract import Command, UploadRequest
from typing_extensions import Self

from sweagent.utils.github import _parse_gh_repo_url
from sweagent.utils.log import get_logger

logger = get_logger("swea-config", emoji="🔧")


class Repo(Protocol):


    base_commit: str
    repo_name: str

    def copy(self, deployment: AbstractDeployment): ...

    def get_reset_commands(self) -> list[str]: ...


def _get_git_reset_commands(base_commit: str) -> list[str]:
    return [
        "git fetch",
        "git status",
        "git restore .",
        "git reset --hard",
        f"git checkout {shlex.quote(base_commit)}",
        "git clean -fdq",
    ]


class PreExistingRepoConfig(BaseModel):


    repo_name: str

    base_commit: str = Field(default="HEAD")


    type: Literal["preexisting"] = "preexisting"


    reset: bool = True


    model_config = ConfigDict(extra="forbid")

    def copy(self, deployment: AbstractDeployment):

        pass

    def get_reset_commands(self) -> list[str]:

        if self.reset:
            return _get_git_reset_commands(self.base_commit)
        return []


class LocalRepoConfig(BaseModel):
    path: Path
    base_commit: str = Field(default="HEAD")


    type: Literal["local"] = "local"


    model_config = ConfigDict(extra="forbid")

    @property
    def repo_name(self) -> str:

        return Path(self.path).resolve().name.replace(" ", "-").replace("'", "")


    def check_valid_repo(self) -> Self:
        try:
            repo = GitRepo(self.path, search_parent_directories=True)
        except InvalidGitRepositoryError as e:
            msg = f"Could not find git repository at {self.path=}."
            raise ValueError(msg) from e
        if repo.is_dirty():
            msg = f"Local git repository {self.path} is dirty. Please commit or stash changes."
            raise ValueError(msg)
        return self

    def copy(self, deployment: AbstractDeployment):
        self.check_valid_repo()
        asyncio.run(
            deployment.runtime.upload(UploadRequest(source_path=str(self.path), target_path=f"/{self.repo_name}"))
        )
        r = asyncio.run(
            deployment.runtime.execute(Command(command=f"chown -R root:root /{self.repo_name}", shell=True))
        )
        if r.exit_code != 0:
            msg = f"Failed to change permissions on copied repository (exit code: {r.exit_code}, stdout: {r.stdout}, stderr: {r.stderr})"
            raise RuntimeError(msg)

    def get_reset_commands(self) -> list[str]:

        return _get_git_reset_commands(self.base_commit)


class GithubRepoConfig(BaseModel):
    github_url: str

    base_commit: str = Field(default="HEAD")


    clone_timeout: float = 500


    type: Literal["github"] = "github"


    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        if self.github_url.count("/") == 1:
            self.github_url = f"https://github.com/{self.github_url}"

    @property
    def repo_name(self) -> str:
        org, repo = _parse_gh_repo_url(self.github_url)
        return f"{org}__{repo}"

    def _get_url_with_token(self, token: str) -> str:

        if not token:
            return self.github_url
        if "@" in self.github_url:
            logger.warning("Cannot prepend token to URL. '@' found in URL")
            return self.github_url
        _, _, url_no_protocol = self.github_url.partition("://")
        return f"https://{token}@{url_no_protocol}"

    def copy(self, deployment: AbstractDeployment):

        base_commit = self.base_commit
        github_token = os.getenv("GITHUB_TOKEN", "")
        url = self._get_url_with_token(github_token)
        asyncio.run(
            deployment.runtime.execute(
                Command(
                    command=" && ".join(
                        (
                            f"mkdir /{self.repo_name}",
                            f"cd /{self.repo_name}",
                            "git init",
                            f"git remote add origin {shlex.quote(url)}",
                            f"git fetch --depth 1 origin {shlex.quote(base_commit)}",
                            "git checkout FETCH_HEAD",
                            "cd ..",
                        )
                    ),
                    timeout=self.clone_timeout,
                    shell=True,
                    check=True,
                )
            ),
        )

    def get_reset_commands(self) -> list[str]:

        return _get_git_reset_commands(self.base_commit)


class SWESmithRepoConfig(BaseModel):


    repo_name: str
    base_commit: str = Field(default="HEAD")
    mirror_url: str = ""


    type: Literal["swesmith_preexisting"] = "swesmith_preexisting"


    model_config = ConfigDict(extra="forbid")

    def copy(self, deployment: AbstractDeployment):
        pass

    @staticmethod
    def _get_url_with_token(url: str, token: str) -> str:
        if not token or not url:
            return url
        _, _, url_no_protocol = url.partition("://")
        return f"https://{token}@{url_no_protocol}"

    def get_reset_commands(self) -> list[str]:
        if self.mirror_url:
            github_token = os.getenv("GITHUB_TOKEN", "")
            url = self._get_url_with_token(self.mirror_url, github_token)
            return [
                "git restore .",
                "git reset --hard",
                f"git fetch {shlex.quote(url)} {shlex.quote(self.base_commit)}",
                "git checkout FETCH_HEAD",
                "git clean -fdq",
            ]
        return _get_git_reset_commands(self.base_commit)


RepoConfig = LocalRepoConfig | GithubRepoConfig | PreExistingRepoConfig | SWESmithRepoConfig


def repo_from_simplified_input(
    *, input: str, base_commit: str = "HEAD", type: Literal["local", "github", "preexisting", "auto"] = "auto"
) -> RepoConfig:

    if type == "local":
        return LocalRepoConfig(path=Path(input), base_commit=base_commit)
    if type == "github":
        return GithubRepoConfig(github_url=input, base_commit=base_commit)
    if type == "preexisting":
        return PreExistingRepoConfig(repo_name=input, base_commit=base_commit)
    if type == "auto":
        if input.startswith("https://github.com/"):
            return GithubRepoConfig(github_url=input, base_commit=base_commit)
        else:
            return LocalRepoConfig(path=Path(input), base_commit=base_commit)
    msg = f"Unknown repo type: {type}"
    raise ValueError(msg)
