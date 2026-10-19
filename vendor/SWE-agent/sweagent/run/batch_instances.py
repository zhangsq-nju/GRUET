import json
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    DummyDeploymentConfig,
    LocalDeploymentConfig,
)
from typing_extensions import Self

from sweagent.agent.problem_statement import (
    ProblemStatementConfig,
    SWEBenchMultimodalProblemStatement,
    TextProblemStatement,
)
from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig, SWESmithRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.utils.files import load_file
from sweagent.utils.github import _is_repo_private
from sweagent.utils.log import get_logger

logger = get_logger("swea-config", emoji="🔧")


class AbstractInstanceSource(ABC):


    @abstractmethod
    def get_instance_configs(self) -> list[EnvironmentConfig]: ...


class BatchInstance(BaseModel):


    env: EnvironmentConfig
    problem_statement: ProblemStatementConfig


def _slice_spec_to_slice(slice_spec: str) -> slice:
    if slice_spec == "":
        return slice(None)
    parts = slice_spec.split(":")
    values = [None if p == "" else int(p) for p in parts]
    if len(parts) == 1:
        return slice(values[0])
    if len(parts) == 2:
        return slice(values[0], values[1])
    if len(parts) == 3:
        return slice(values[0], values[1], values[2])
    msg = (
        f"Invalid slice specification: {slice_spec!r}. "
        "Here's the expected format: stop or start:stop or start:stop:step "
        "(i.e., it behaves exactly like python's list slicing `list[slice]`)."
    )
    raise ValueError(msg)


def _filter_batch_items(
    instances: list[BatchInstance], *, filter_: str, slice_: str = "", shuffle: bool = False
) -> list[BatchInstance]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x.problem_statement.id)
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_, instance.problem_statement.id)]
    after_filter = len(instances)
    if before_filter != after_filter:
        logger.info("Instance filter: %d -> %d instances", before_filter, after_filter)
    if slice_:
        instances = instances[_slice_spec_to_slice(slice_)]
        after_slice = len(instances)
        if before_filter != after_slice:
            logger.info("Instance slice: %d -> %d instances", before_filter, after_slice)
    return instances


class SimpleBatchInstance(BaseModel):


    image_name: str
    problem_statement: str
    instance_id: str
    repo_name: str = ""

    base_commit: str = "HEAD"

    extra_fields: dict[str, Any] = Field(default_factory=dict)


    model_config = ConfigDict(extra="ignore")

    def to_full_batch_instance(self, deployment: DeploymentConfig) -> BatchInstance:


        deployment = deployment.model_copy(deep=True)

        if "issue_images" in self.extra_fields:
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=self.problem_statement,
                issue_images=self.extra_fields.pop("issue_images"),
                id=self.instance_id,
                extra_fields=self.extra_fields,
            )
        else:
            problem_statement = TextProblemStatement(
                text=self.problem_statement, id=self.instance_id, extra_fields=self.extra_fields
            )

        if not self.repo_name:
            repo = None
        elif "github" in self.repo_name:
            repo = GithubRepoConfig(github_url=self.repo_name, base_commit=self.base_commit)
        elif "/" not in self.repo_name:
            repo = PreExistingRepoConfig(repo_name=self.repo_name, base_commit=self.base_commit)
        else:
            repo = LocalRepoConfig(path=Path(self.repo_name), base_commit=self.base_commit)
        if isinstance(deployment, LocalDeploymentConfig):
            if self.image_name:
                msg = "Local deployment does not support image_name"
                raise ValueError(msg)
            return BatchInstance(
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )
        if isinstance(deployment, DummyDeploymentConfig):
            return BatchInstance(
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )

        deployment.image = self.image_name

        if isinstance(deployment, DockerDeploymentConfig) and deployment.python_standalone_dir is None:

            deployment.python_standalone_dir = "/root"

        return BatchInstance(
            env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
        )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_id(cls, data):

        if isinstance(data, dict):
            if "id" in data and "instance_id" not in data:
                data["instance_id"] = data["id"]
                data.pop("id")
        return data


    @classmethod
    def from_swe_bench(cls, instance: dict[str, Any]) -> Self:

        iid = instance["instance_id"]
        image_name = instance.get("image_name", None)
        if image_name is None:

            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        extra_fields = {}
        if "image_assets" in instance:
            issue_images = json.loads(instance["image_assets"])["problem_statement"]
            extra_fields["issue_images"] = issue_images
        return cls(
            image_name=image_name,
            problem_statement=instance["problem_statement"],
            instance_id=iid,
            repo_name="testbed",
            base_commit=instance["base_commit"],
            extra_fields=extra_fields,
        )


class InstancesFromFile(BaseModel, AbstractInstanceSource):


    path: Path
    filter: str = ".*"

    slice: str = ""

    shuffle: bool = False


    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
        description="Deployment options.",
    )


    simple: Literal[True] = True


    type: Literal["file"] = "file"


    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        simple_instances = [SimpleBatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class InstancesFromHuggingFace(BaseModel, AbstractInstanceSource):


    dataset_name: str

    split: str = "dev"
    filter: str = ".*"

    slice: str = ""

    shuffle: bool = False


    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )

    type: Literal["huggingface"] = "huggingface"


    def get_instance_configs(self) -> list[BatchInstance]:
        from datasets import load_dataset

        ds: list[dict[str, Any]] = load_dataset(self.dataset_name, split=self.split)
        simple_instances: list[SimpleBatchInstance] = [SimpleBatchInstance.model_validate(instance) for instance in ds]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        ds_name = "".join(l for l in self.dataset_name if l.isalnum() or l in ["-", "_"])
        return f"{ds_name}_{self.split}"


class SWEBenchInstances(BaseModel, AbstractInstanceSource):


    subset: Literal["lite", "verified", "full", "multimodal", "multilingual"] = "lite"


    path_override: str | Path | None = None


    split: Literal["dev", "test"] = "dev"

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )


    type: Literal["swe_bench"] = "swe_bench"


    filter: str = ".*"

    slice: str = ""

    shuffle: bool = False


    evaluate: bool = False


    def _get_dataset_path(self) -> str:
        if self.path_override is not None:
            return str(self.path_override)
        dataset_mapping = {
            "full": "princeton-nlp/SWE-Bench",
            "verified": "princeton-nlp/SWE-Bench_Verified",
            "lite": "princeton-nlp/SWE-Bench_Lite",
            "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
            "multilingual": "swe-bench/SWE-Bench_Multilingual",
        }

        if self.subset not in dataset_mapping:
            msg = f"Unsupported subset: {self.subset}"
            raise ValueError(msg)

        return dataset_mapping[self.subset]

    def get_instance_configs(self) -> list[BatchInstance]:
        from datasets import load_dataset

        ds: list[dict[str, Any]] = load_dataset(self._get_dataset_path(), split=self.split)

        if isinstance(self.deployment, DockerDeploymentConfig):
            self.deployment.platform = "linux/amd64"

        instances = [
            SimpleBatchInstance.from_swe_bench(instance).to_full_batch_instance(self.deployment) for instance in ds
        ]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return f"swe_bench_{self.subset}_{self.split}"


class ExpertInstancesFromFile(BaseModel, AbstractInstanceSource):


    path: Path
    filter: str = ".*"

    slice: str = ""

    shuffle: bool = False


    type: Literal["expert_file"] = "expert_file"


    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        instances = [BatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class SWESmithInstances(BaseModel, AbstractInstanceSource):


    path: Path

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )


    filter: str = ".*"

    slice: str = ""

    shuffle: bool = False


    type: Literal["swesmith"] = "swesmith"


    def get_instance_configs(self) -> list[BatchInstance]:
        github_token = os.getenv("GITHUB_TOKEN", "")

        instance_dicts = load_file(self.path)
        instances = []

        for instance_dict in instance_dicts:
            deployment = self.deployment.model_copy(deep=True)
            deployment.image = instance_dict["image_name"]

            if isinstance(deployment, DockerDeploymentConfig) and deployment.python_standalone_dir is None:
                deployment.python_standalone_dir = "/root"

            instance_id = instance_dict["instance_id"]
            repo_field = instance_dict.get("repo", "")

            mirror_url = ""
            if repo_field and _is_repo_private(repo_field, github_token):
                if not github_token:
                    msg = (
                        f"Repo '{repo_field}' appears to be private but GITHUB_TOKEN is not set. "
                        "Set GITHUB_TOKEN with 'repo' scope to access private repositories."
                    )
                    raise ValueError(msg)
                mirror_url = f"https://github.com/{repo_field}.git"

            repo = SWESmithRepoConfig(
                repo_name="testbed",
                base_commit=instance_id,
                mirror_url=mirror_url,
            )

            problem_statement = TextProblemStatement(
                text=instance_dict.get("problem_statement", ""),
                id=instance_id,
                extra_fields={"fail_to_pass": instance_dict.get("FAIL_TO_PASS", [])},
            )

            instances.append(
                BatchInstance(
                    env=EnvironmentConfig(deployment=deployment, repo=repo),
                    problem_statement=problem_statement,
                )
            )

        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return f"swesmith_{self.path.stem}"


BatchInstanceSourceConfig = (
    InstancesFromHuggingFace | InstancesFromFile | SWEBenchInstances | ExpertInstancesFromFile | SWESmithInstances
)
