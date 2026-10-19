from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator
from simple_parsing.helpers.fields import field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from tenacity import RetryError
from typing_extensions import Self
from unidiff import UnidiffParseError

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    HumanModel,
    HumanThoughtModel,
    InstanceStats,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.agent.reviewer import (
    ChooserRetryLoop,
    RetryLoopConfig,
    ReviewSubmission,
    ScoreRetryLoop,
    get_retry_loop_from_config,
)
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    TotalCostLimitExceededError,
)
from sweagent.tools.parsing import (
    ActionOnlyParser,
    ThoughtActionParser,
    extract_thought_action_spans,
)
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _convert_paths_to_abspath, _strip_abspath_from_dict
from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from sweagent.utils.log import get_logger
from sweagent.utils.patch_formatter import PatchFormatter


class TemplateConfig(BaseModel):


    system_template: str = ""
    instance_template: str = ""
    next_step_template: str = "Observation: {{observation}}"

    next_step_truncated_observation_template: str = (
        "Observation: {{observation[:max_observation_length]}}<response clipped>"
        "<NOTE>Observations should not exceeded {{max_observation_length}} characters. "
        "{{elided_chars}} characters were elided. Please try a different command that produces less output "
        "or use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
    )


    max_observation_length: int = 100_000


    next_step_no_output_template: str = None


    strategy_template: str | None = None
    demonstration_template: str | None = None

    demonstrations: list[Path] = field(default_factory=list)


    put_demos_in_history: bool = False


    disable_image_processing: bool = False


    shell_check_error_template: str = (
        "Your bash command contained syntax errors and was NOT executed. "
        "Please fix the syntax errors and try again. This can be the result "
        "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
        "{{bash_stdout}}\n{{bash_stderr}}"
    )


    command_cancelled_timeout_template: str = (
        "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. "
        "Please try a different command that completes more quickly. "
        "Note: A common source of this error is if the command is interactive or requires user input "
        "(it is impossible to receive user input in the current environment, so the command will never complete)."
    )


    def model_post_init(self, __context):
        self.demonstrations = _convert_paths_to_abspath(self.demonstrations)
        if self.next_step_no_output_template is None:
            self.next_step_no_output_template = self.next_step_template

    @model_validator(mode="after")
    def validate_template_jinja_syntax(self) -> Self:
        template_fields = [field for field in self.model_fields.keys() if field.endswith("_template")]
        for field in template_fields:
            value = getattr(self, field)
            _warn_probably_wrong_jinja_syntax(value)
        return self

    @model_validator(mode="after")
    def warnings(self) -> Self:
        logger = get_logger("swea-config", emoji="🔧")
        if self.put_demos_in_history and self.demonstration_template is not None:
            logger.warning("demonstration_template is ignored when put_demos_in_history is True")
        if not self.system_template or not self.instance_template:
            logger.warning(
                "system_template/instance_template is not set, using empty string. Perhaps you were"
                " overwriting the default config? See https://swe-agent.com/latest/usage/cl_tutorial/"
                " for more information. Note: You can ignore this warning in human mode."
            )
        return self


class CandidateSamplingConfig(BaseModel):
    enabled: bool = False
    num_candidates: int = 1
    max_retry_candidates: int | None = None
    selection_policy: Literal["first_compliant"] = "first_compliant"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_candidate_budget(self) -> Self:
        if int(self.num_candidates) < 1:
            raise ValueError("candidate_sampling.num_candidates must be >= 1")
        if self.max_retry_candidates is None:
            self.max_retry_candidates = 3 * int(self.num_candidates)
        if int(self.max_retry_candidates) < int(self.num_candidates):
            raise ValueError(
                "candidate_sampling.max_retry_candidates must be >= "
                "candidate_sampling.num_candidates"
            )
        return self


class DefaultAgentConfig(BaseModel):


    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3

    max_steps: int = Field(default=0, ge=0)

    action_sampler: ActionSamplerConfig | None = None
    candidate_sampling: "CandidateSamplingConfig" = Field(default_factory=lambda: CandidateSamplingConfig())

    type: Literal["default"] = "default"


    model_config = ConfigDict(extra="forbid")


class ShellAgentConfig(BaseModel):
    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3

    max_steps: int = Field(default=0, ge=0)


    type: Literal["shell"] = "shell"


    model_config = ConfigDict(extra="forbid")


class RetryAgentConfig(BaseModel):
    name: str = "retry_main"
    agent_configs: list[DefaultAgentConfig]
    retry_loop: RetryLoopConfig
    type: Literal["retry"] = "retry"
    model_config = ConfigDict(extra="forbid")


AgentConfig = Annotated[DefaultAgentConfig | RetryAgentConfig | ShellAgentConfig, Field(union_mode="left_to_right")]


class _BlockedActionError(Exception):
    pass


class _RetryWithOutput(Exception):
    pass


class _RetryWithoutOutput(Exception):
    pass


class _ExitForfeit(Exception):
    pass


class _TotalExecutionTimeExceeded(Exception):
    pass


RETRY_WITH_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITH-OUTPUT###"
RETRY_WITHOUT_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITHOUT-OUTPUT###"
EXIT_FORFEIT_TOKEN = "###SWE-AGENT-EXIT-FORFEIT###"


class AbstractAgent:
    def __init__(self, *args, **kwargs):
        model: AbstractModel
        replay_config: BaseModel | None
        logger: logging.Logger

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self: ...

    def add_hook(self, hook: AbstractAgentHook) -> None: ...

    def get_trajectory_data(self) -> dict[str, Any]: ...

    def step(self) -> StepOutput: ...

    def run(self, *args, **kwargs) -> AgentRunResult: ...


def get_agent_from_config(config: AgentConfig) -> AbstractAgent:
    if config.type == "default":
        return DefaultAgent.from_config(config)
    elif config.type == "retry":
        return RetryAgent.from_config(config)
    elif config.type == "shell":

        from sweagent.agent.extra.shell_agent import ShellAgent

        return ShellAgent.from_config(config)
    else:
        msg = f"Unknown agent type: {config.type}"
        raise ValueError(msg)


class RetryAgent(AbstractAgent):
    def __init__(self, config: RetryAgentConfig):

        self.config = config.model_copy(deep=True)
        self._hooks = []
        self._i_attempt = 0
        self.logger = get_logger("swea-agent", emoji="🤠")
        self._agent: DefaultAgent | None = None
        self._attempt_data: list[dict[str, Any]] = []
        self._total_instance_attempt_stats = InstanceStats()

        self._chook = CombinedAgentHook()
        self._traj_path: Path | None = None
        self._problem_statement: ProblemStatement | None = None
        self._env: SWEEnv | None = None
        self._output_dir: Path | None = None
        self._rloop: ScoreRetryLoop | ChooserRetryLoop | None = None

    @property
    def _total_instance_stats(self) -> InstanceStats:
        assert self._rloop is not None
        return self._total_instance_attempt_stats + self._rloop.review_model_stats

    @classmethod
    def from_config(cls, config: RetryAgentConfig) -> Self:
        return cls(config)

    def add_hook(self, hook: AbstractAgentHook) -> None:
        self._chook.add_hook(hook)
        self._hooks.append(hook)

    def setup(
        self, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig, output_dir: Path = Path(".")
    ) -> None:

        self._total_instance_attempt_stats = InstanceStats()
        self._problem_statement = problem_statement
        self._traj_path = output_dir / (self._problem_statement.id + ".traj")
        self._env = env
        self._output_dir = output_dir
        self._rloop = get_retry_loop_from_config(self.config.retry_loop, problem_statement=problem_statement)

    def _setup_agent(self) -> AbstractAgent:


        agent_config = self.config.agent_configs[self._i_attempt % len(self.config.agent_configs)].model_copy(deep=True)
        remaining_budget = self.config.retry_loop.cost_limit - self._total_instance_stats.instance_cost
        if remaining_budget < agent_config.model.per_instance_cost_limit:
            self.logger.debug("Setting agent per-attempt cost limit to remaining budget: %s", remaining_budget)
            agent_config.model.per_instance_cost_limit = remaining_budget
        self._agent = DefaultAgent.from_config(agent_config)
        for hook in self._hooks:
            self._agent.add_hook(hook)
        assert self._output_dir is not None
        sub_agent_output_dir = self._output_dir / f"attempt_{self._i_attempt}"
        assert self._problem_statement is not None
        assert self._env is not None
        self._agent.setup(env=self._env, problem_statement=self._problem_statement, output_dir=sub_agent_output_dir)
        return self._agent

    def _next_attempt(self) -> None:

        assert self._env is not None
        self._i_attempt += 1
        self._env.hard_reset()
        self._setup_agent()

    def step(self) -> StepOutput:

        assert self._agent is not None


        if self._total_instance_stats.instance_cost > 1.1 * self.config.retry_loop.cost_limit > 0:
            msg = "Total instance cost exceeded cost limit. This should not happen, please report this. Triggering autosubmit."
            self.logger.critical(msg)
            return self._agent.attempt_autosubmission_after_error(step=StepOutput())
        try:
            step = self._agent.step()
        except TotalCostLimitExceededError:

            raise
        except Exception as e:
            msg = "Error in agent step: %s. This really shouldn't happen, please report this. Triggering autosubmit."
            self.logger.critical(msg, e, exc_info=True)
            step = self._agent.attempt_autosubmission_after_error(step=StepOutput())
        return step

    def _finalize_agent_run(self) -> None:

        assert self._agent is not None
        self._agent.save_trajectory()
        self._attempt_data.append(self._agent.get_trajectory_data())
        self._total_instance_attempt_stats += self._agent.model.stats

    def get_trajectory_data(self, choose: bool) -> dict[str, Any]:

        assert self._rloop is not None

        data = {
            "attempts": self._attempt_data,
        }

        if choose:
            try:
                best_attempt_idx = self._rloop.get_best()
            except TotalCostLimitExceededError:
                raise
            except Exception as e:
                self.logger.critical(f"Error getting best attempt index: {e}. Setting to 0.", exc_info=True)
                best_attempt_idx = 0
            data |= copy.deepcopy(self._attempt_data[best_attempt_idx])
            data["info"]["best_attempt_idx"] = best_attempt_idx
            data["info"]["rloop_model_stats"] = self._rloop.review_model_stats.model_dump()

            data["info"]["model_stats"] = self._total_instance_stats.model_dump()
            if isinstance(self._rloop, ChooserRetryLoop):
                data["info"]["chooser"] = (
                    self._rloop._chooser_output.model_dump() if self._rloop._chooser_output else {}
                )
        return data

    def save_trajectory(self, choose: bool) -> None:
        data = self.get_trajectory_data(choose=choose)
        assert self._traj_path is not None
        self._traj_path.write_text(json.dumps(data, indent=2))

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:

        output_dir.mkdir(parents=True, exist_ok=True)
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)
        assert self._rloop is not None


        self._chook.on_run_start()
        step_output = StepOutput()
        self._setup_agent()
        assert self._agent is not None
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory(choose=False)
            if step_output.done:
                self._rloop.on_submit(
                    ReviewSubmission(
                        trajectory=self._agent.trajectory,
                        info=self._agent.info,
                        model_stats=self._agent.model.stats,
                    )
                )
                if isinstance(self._rloop, ScoreRetryLoop):
                    self._agent.info["review"] = self._rloop.reviews[-1].model_dump()
                self._finalize_agent_run()
                self.save_trajectory(choose=False)
                if self._rloop.retry():
                    assert self._env is not None
                    self._next_attempt()
                    step_output.done = False
        self.save_trajectory(choose=True)
        self._chook.on_run_done(trajectory=self._agent.trajectory, info=self._agent.info)

        self.logger.info("Trajectory saved to %s", self._traj_path)


        data = self.get_trajectory_data(choose=True)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])


class DefaultAgent(AbstractAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        max_steps: int = 0,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
        candidate_sampling_config: CandidateSamplingConfig | None = None,
    ):

        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.max_steps = max_steps
        self.logger = get_logger("swea-agent", emoji="🤠")

        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None


        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None


        self._action_sampler: AbstractActionSampler | None = None
        self.candidate_sampling = candidate_sampling_config or CandidateSamplingConfig()
        if self.candidate_sampling.enabled and action_sampler_config is not None:
            msg = "candidate_sampling and action_sampler cannot be enabled at the same time"
            raise ValueError(msg)
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)


        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0
        self._n_completed_visible_steps = 0

    @classmethod
    def from_config(cls, config: DefaultAgentConfig, *, model: AbstractModel | None = None) -> Self:


        config = config.model_copy(deep=True)
        if model is None:
            model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            max_steps=config.max_steps,
            action_sampler_config=config.action_sampler,
            candidate_sampling_config=config.candidate_sampling,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:

        hook.on_init(agent=self)
        self._chook.add_hook(hook)


    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):

        from sweagent.run.run_single import RunSingleConfig

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:

        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]


        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages


    def _append_history(self, item: dict[str, Any]) -> None:

        self._chook.on_query_message_added(**item)
        self.history.append(item)

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)


        if hasattr(problem_statement, "type") and problem_statement.type == "swe_bench_multimodal":
            from sweagent.agent.problem_statement import SWEBenchMultimodalProblemStatement

            if isinstance(problem_statement, SWEBenchMultimodalProblemStatement):

                if not problem_statement.disable_image_processing and self.templates.disable_image_processing:
                    problem_statement.disable_image_processing = True

        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)


        self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)
        self._n_completed_visible_steps = 0

        self._chook.on_tools_installation_started()
        self.tools.install(self._env)
        self._chook.on_setup_attempt()
        self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        assert self._env is not None
        assert self._problem_statement is not None
        self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self.add_system_message_to_history()
        self.add_demonstrations_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(self._env))
        self._chook.on_setup_done()

    def add_system_message_to_history(self) -> None:

        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def add_demonstrations_to_history(self) -> None:

        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:

        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)


        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(_demo_text)["history"]
        else:
            demo_history = json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:

            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:

            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:

        assert self._problem_statement is not None
        assert self._env is not None
        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:

        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)


        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_step_to_history(self, step: StepOutput) -> None:

        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":

            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:

            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:

        templates: list[str] = []

        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)

        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)

    def get_trajectory_data(self) -> dict[str, Any]:


        assert self._env is not None


        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None
        attempt_data["environment"] = self._env.name
        if self.candidate_sampling.enabled:
            attempt_data["trajectory_format"] = "candidate_v1"
        return attempt_data

    def save_trajectory(
        self,
    ) -> None:

        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_model_requery_history(
        self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:

        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")

        return self.messages + [
            {"role": "assistant", "content": output, "agent": self.name, "message_type": "assistant"},
            {"role": "user", "content": error_template, "agent": self.name, "message_type": "user"},
        ]

    def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:

        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not asyncio.run(self._env.deployment.is_alive(timeout=10)):


            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step

        repo_name = "/"
        if self._env.repo is not None:
            repo_name = f"/{self._env.repo.repo_name}"
        submission_command = "git add -A && git diff --cached > /root/model.patch"
        self.logger.info("Executing submission command %s in %s", submission_command, repo_name)
        try:
            self._env.execute_command(submission_command, check=True, cwd=repo_name)
        except Exception as e:
            self.logger.error("Failed to execute submission command, got %s", e)


        step = self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        return step

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:

        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            try:
                submission = self._env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                self.logger.exception("Failed to read submission file, got %s", e)
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:

        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: self._env.read_file(
                            PurePosixPath("/") / self._env.repo.repo_name / path
                        ),
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None

        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    def handle_action(self, step: StepOutput) -> StepOutput:

        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = self.tools.get_state(env=self._env)
            return step

        assert self._env is not None
        self._chook.on_action_started(step=step)
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        except CommandTimeoutError:
            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception as f:
                self.logger.exception("Failed to interrupt session after command timeout: %s", f, exc_info=True)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        self._chook.on_action_executed(step=step)
        step.state = self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        return self.handle_submission(step)

    def forward(self, history: list[dict[str, str]]) -> StepOutput:

        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()


        step = StepOutput()
        step.query = copy.deepcopy(history)
        step.prompt = self.model.render_prompt(history)
        try:

            self._chook.on_model_query(messages=history, agent=self.name)

            if self.candidate_sampling.enabled:
                requested_candidates = max(1, int(self.candidate_sampling.num_candidates))
                max_retry_candidates = max(
                    requested_candidates,
                    int(self.candidate_sampling.max_retry_candidates or 0),
                )
                sampled_outputs = self._query_candidate_outputs(history, requested_candidates)
                step.prompt = str(sampled_outputs[0].get("prompt", step.prompt)) if sampled_outputs else step.prompt
                self._populate_step_from_candidate_pool(
                    step,
                    sampled_outputs,
                    requested_candidates=requested_candidates,
                    max_retry_candidates=max_retry_candidates,
                    candidate_idx_offset=0,
                    select_first_compliant=True,
                    is_top_up=False,
                )
                if step.selected_idx is None:
                    step.output = self._render_candidate_requery_output(sampled_outputs)
                    err = FormatError("No compliant candidate survived candidate sampling.")
                    err.message = "No compliant candidate survived candidate sampling."
                    raise err
                selected_candidate = step.candidates[step.selected_idx]
                selected_attempt_idx = int(selected_candidate["candidate_idx"])
                selected_output = sampled_outputs[selected_attempt_idx]
                while (
                    len(step.candidates) < requested_candidates
                    and int(step.sampling["attempts_used"]) < max_retry_candidates
                ):
                    missing_candidates = requested_candidates - len(step.candidates)
                    remaining_budget = max_retry_candidates - int(step.sampling["attempts_used"])
                    top_up_outputs = self._query_candidate_outputs(history, min(missing_candidates, remaining_budget))
                    self._populate_step_from_candidate_pool(
                        step,
                        top_up_outputs,
                        requested_candidates=requested_candidates,
                        max_retry_candidates=max_retry_candidates,
                        candidate_idx_offset=int(step.sampling["attempts_used"]),
                        select_first_compliant=False,
                        is_top_up=True,
                    )
                step.output = str(selected_candidate.get("raw_response") or "")
                step.thinking_blocks = selected_output.get("thinking_blocks", [])
                if selected_output.get("tool_calls") is not None:
                    step.tool_call_ids = [call["id"] for call in selected_output["tool_calls"]]
                    step.tool_calls = selected_output["tool_calls"]
            elif self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=history,
                )
                output = best.completion

                step.extra_info.update(best.extra_info)
            else:
                output = self.model.query(history)
            if not self.candidate_sampling.enabled:
                step.output = output["message"]

                step.thought, step.action = self.tools.parse_actions(output)
                step.thinking_blocks = output.get("thinking_blocks", [])
                if output.get("tool_calls") is not None:
                    step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                    step.tool_calls = output["tool_calls"]
            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n🎬 ACTION\n{step.action.strip()}")
            self._chook.on_actions_generated(step=step)
            return self.handle_action(step)
        except Exception as e:
            if step.action == step.thought == "" and not step.sampling:


                step.thought = step.output

            e.step = step
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:


        def handle_error_with_autosubmission(exit_status: str, message: str) -> StepOutput:

            self.logger.warning(message)
            return self.attempt_autosubmission_after_error(
                StepOutput(
                    thought=message,
                    exit_status=exit_status,
                    output=message,
                    done=True,
                )
            )

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[dict[str, str]]:

            self.logger.warning("Requerying model after %s (%dth requery)", type(exception).__name__, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)


            except KeyboardInterrupt:
                raise
            except EOFError:
                raise


            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails
                )
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1

                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass


            except _ExitForfeit:
                self.logger.info("Exiting due to forfeit")
                return handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )

            except _TotalExecutionTimeExceeded:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )

            except CommandTimeoutError:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )

            except ContextWindowExceededError:
                return handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError:
                return handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_payload: dict[str, Any] = {
            "action": step.action,
            "observation": step.observation,
            "response": step.output,
            "thought": step.thought,
            "execution_time": step.execution_time,
            "state": step.state,
            "query": step.query,
            "extra_info": step.extra_info,
        }
        if self.candidate_sampling.enabled:
            trajectory_payload["candidates"] = copy.deepcopy(step.candidates)
            trajectory_payload["prompt"] = step.prompt
            trajectory_payload["sampling"] = copy.deepcopy(step.sampling)
            trajectory_payload["selected_idx"] = step.selected_idx
        trajectory_step = TrajectoryStep(trajectory_payload)
        self.trajectory.append(trajectory_step)

    def _finalize_step_output(
        self,
        step_output: StepOutput,
        *,
        counted_visible_step: bool,
    ) -> StepOutput:
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))
        self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)
        if counted_visible_step:
            self._n_completed_visible_steps += 1

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    def _exit_for_max_steps(self) -> StepOutput:
        message = f"Exit due to max steps limit ({self.max_steps})"
        self.logger.warning(message)
        return self.attempt_autosubmission_after_error(
            StepOutput(
                thought=message,
                output=message,
                observation=message,
                exit_status="exit_max_steps",
                done=True,
            )
        )

    def step(self) -> StepOutput:


        assert self._env is not None
        if 0 < self.max_steps <= self._n_completed_visible_steps:
            return self._finalize_step_output(
                self._exit_for_max_steps(),
                counted_visible_step=False,
            )
        self._chook.on_step_start()

        n_step = self._n_completed_visible_steps + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        return self._finalize_step_output(
            step_output,
            counted_visible_step=True,
        )

    def _query_candidate_outputs(self, history: list[dict[str, str]], n: int) -> list[dict[str, Any]]:
        sampled_outputs = self.model.query(history, n=max(1, int(n)))
        if isinstance(sampled_outputs, dict):
            return [sampled_outputs]
        return sampled_outputs

    def _populate_step_from_candidate_pool(
        self,
        step: StepOutput,
        sampled_outputs: list[dict[str, Any]],
        *,
        requested_candidates: int,
        max_retry_candidates: int,
        candidate_idx_offset: int,
        select_first_compliant: bool,
        is_top_up: bool,
    ) -> None:
        if not step.sampling:
            step.sampling = {
                "requested_candidates": requested_candidates,
                "attempts_used": 0,
                "compliant_count": 0,
                "reject_count": 0,
                "top_up_attempts_used": 0,
                "max_retry_candidates": max_retry_candidates,
            }
        for sample_idx, output in enumerate(sampled_outputs):
            candidate = self._materialize_candidate(output, candidate_idx=candidate_idx_offset + sample_idx)
            step.sampling["attempts_used"] += 1
            if is_top_up:
                step.sampling["top_up_attempts_used"] += 1
            if candidate is None or not bool(candidate.get("compliant", False)):
                step.sampling["reject_count"] += 1
                continue

            candidate.pop("compliant", None)
            candidate.pop("compliance_reason", None)
            persisted_idx = len(step.candidates)
            step.candidates.append(candidate)
            step.sampling["compliant_count"] += 1
            if select_first_compliant and step.selected_idx is None:
                if self.candidate_sampling.selection_policy != "first_compliant":
                    msg = f"Unsupported candidate selection policy: {self.candidate_sampling.selection_policy}"
                    raise ValueError(msg)
                step.selected_idx = persisted_idx
                step.thought = str(candidate["thought"])
                step.action = str(candidate["action"])

        step.extra_info["selection_policy"] = self.candidate_sampling.selection_policy

    def _materialize_candidate(self, output: dict[str, Any], *, candidate_idx: int) -> dict[str, Any] | None:
        analysis = self.tools.analyze_candidate(output)
        if analysis["parse_status"] != "ok":
            return None

        raw_response = str(output.get("raw_response") or output.get("message") or "")
        try:
            char_spans = extract_thought_action_spans(raw_response)
        except FormatError:
            return None

        token_ids = output.get("token_ids")
        if token_ids is None:
            encoded = self.model.encode_text(raw_response)
            if encoded is None:
                return None
            token_ids = encoded
        token_ids = [int(tok) for tok in token_ids]

        token_logprobs = output.get("token_logprobs")
        if token_logprobs is None:
            token_logprobs = [0.0 for _ in token_ids]
        token_logprobs = [float(lp) for lp in token_logprobs]
        if len(token_logprobs) != len(token_ids):
            return None

        top_k = output.get("top_k")
        if top_k is None:
            top_k = [[] for _ in token_ids]
        if len(top_k) != len(token_ids):
            return None

        token_spans = self._token_spans_from_char_spans(raw_response, char_spans, len(token_ids))
        if token_spans is None:
            return None

        return {
            "candidate_idx": candidate_idx,
            "raw_response": raw_response,
            "accepted_text": raw_response[char_spans["accepted_start"] : char_spans["accepted_end"]],
            "thought": str(analysis["thought"]),
            "action": str(analysis["action"]),
            "parse_status": "ok",
            "parse_error": None,
            "compliant": bool(analysis["compliant"]),
            "compliance_reason": analysis["compliance_reason"],
            "finish_reason": str(output.get("finish_reason") or "unknown"),
            "char_spans": char_spans,
            "token_spans": token_spans,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs,
            "top_k": self._normalize_top_k(top_k),
        }

    def _token_spans_from_char_spans(
        self,
        raw_response: str,
        char_spans: dict[str, int],
        max_token_count: int,
    ) -> dict[str, int] | None:
        keys = (
            "accepted_start",
            "accepted_end",
            "thought_start",
            "thought_end",
            "action_start",
            "action_end",
        )
        token_spans: dict[str, int] = {}
        for key in keys:
            prefix = raw_response[: int(char_spans[key])]
            token_ids = self.model.encode_text(prefix)
            if token_ids is None:
                return None
            token_idx = len(token_ids)
            if token_idx < 0 or token_idx > max_token_count:
                return None
            token_spans[key] = token_idx

        if not (
            token_spans["accepted_start"]
            <= token_spans["thought_start"]
            <= token_spans["thought_end"]
            <= token_spans["action_start"]
            <= token_spans["action_end"]
            <= token_spans["accepted_end"]
        ):
            return None
        return token_spans

    def _normalize_top_k(self, top_k: Any) -> list[list[dict[str, float | int]]]:
        normalized: list[list[dict[str, float | int]]] = []
        for step in top_k:
            entries: list[dict[str, float | int]] = []
            for item in step or []:
                if isinstance(item, dict):
                    entries.append({"token_id": int(item["token_id"]), "logprob": float(item["logprob"])})
                else:
                    entries.append({"token_id": int(item.token_id), "logprob": float(item.logprob)})
            normalized.append(entries)
        return normalized

    def _render_candidate_requery_output(self, sampled_outputs: list[dict[str, Any]]) -> str:
        rendered = []
        for idx, output in enumerate(sampled_outputs):
            raw_response = str(output.get("raw_response") or output.get("message") or "")
            rendered.append(f"[candidate {idx}]\n{raw_response}")
        return "\n\n".join(rendered)

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:

        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)


        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)


        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])
