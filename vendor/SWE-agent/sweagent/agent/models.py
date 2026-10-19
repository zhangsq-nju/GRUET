from __future__ import annotations

import copy
import json
import os
import random
import shlex
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

import litellm
import litellm.types.utils
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, SecretStr
from swerex.exceptions import SwerexException
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent import REPO_ROOT, __version__
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FunctionCallingFormatError,
    InstanceCallLimitExceededError,
    InstanceCostLimitExceededError,
    ModelConfigurationError,
    TotalCostLimitExceededError,
)
from sweagent.tools.tools import ToolConfig
from sweagent.types import History, HistoryItem
from sweagent.utils.log import get_logger

try:
    import readline
except ImportError:
    readline = None

litellm.suppress_debug_info = True


_THREADS_THAT_USED_API_KEYS = []


class RetryConfig(PydanticBaseModel):


    retries: int = 20

    min_wait: float = 10

    max_wait: float = 120


class GenericAPIModelConfig(PydanticBaseModel):


    name: str = Field(description="Name of the model.")

    per_instance_cost_limit: float = Field(
        default=3.0,
        description="Cost limit for every instance (task).",
    )
    total_cost_limit: float = Field(default=0.0, description="Total cost limit.")
    per_instance_call_limit: int = Field(default=0, description="Per instance call limit.")
    temperature: float = 0.0

    top_p: float | None = 1.0

    top_k: int | None = None

    api_base: str | None = None
    api_version: str | None = None
    api_key: SecretStr | None = None

    stop: list[str] = []


    completion_kwargs: dict[str, Any] = {}


    convert_system_to_user: bool = False


    retry: RetryConfig = RetryConfig()


    delay: float = 0.0


    fallbacks: list[dict[str, Any]] = []


    choose_api_key_by_thread: bool = True


    max_input_tokens: int | None = None


    max_output_tokens: int | None = None


    litellm_model_registry: str | None = None


    custom_tokenizer: dict[str, Any] | None = None


    backend: Literal["litellm", "vllm"] = "litellm"


    tokenizer_path: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    max_model_len: int | None = None


    model_config = ConfigDict(extra="forbid")

    def get_api_keys(self) -> list[str]:

        if self.api_key is None:
            return []
        api_key = self.api_key.get_secret_value()
        if not api_key:
            return []
        if api_key.startswith("$"):
            env_var_name = api_key[1:]
            api_key = os.getenv(env_var_name, "")
            if not api_key:
                get_logger("swea-config", emoji="🔧").warning(f"Environment variable {env_var_name} not set")
                return []
        return api_key.split(":::")

    def choose_api_key(self) -> str | None:

        api_keys = self.get_api_keys()
        if not api_keys:
            return None
        if not self.choose_api_key_by_thread:
            return random.choice(api_keys)
        thread_name = threading.current_thread().name
        if thread_name not in _THREADS_THAT_USED_API_KEYS:
            _THREADS_THAT_USED_API_KEYS.append(thread_name)
        thread_idx = _THREADS_THAT_USED_API_KEYS.index(thread_name)
        key_idx = thread_idx % len(api_keys)
        get_logger("config", emoji="🔧").debug(
            f"Choosing API key {key_idx} for thread {thread_name} (idx {thread_idx})"
        )
        return api_keys[key_idx]

    @property
    def id(self) -> str:
        name = self.name.replace("/", "--")
        if self.top_p is not None:
            top_p = f"{self.top_p:.2f}"
        else:
            top_p = "None"
        temperature = f"{self.temperature:.2f}"
        per_instance_cost_limit = f"{self.per_instance_cost_limit:.2f}"
        return f"{name}__t-{temperature}__p-{top_p}__c-{per_instance_cost_limit}"


class ReplayModelConfig(GenericAPIModelConfig):
    replay_path: Path = Field(description="Path to replay file when using the replay model.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )

    name: Literal["replay"] = Field(default="replay", description="Model name.")

    model_config = ConfigDict(extra="forbid")


class HumanModelConfig(GenericAPIModelConfig):
    name: Literal["human"] = Field(default="human", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(default=0.0, description="Cost limit for all instances (tasks).")
    cost_per_call: float = 0.0
    catch_eof: bool = True

    model_config = ConfigDict(extra="forbid")


class HumanThoughtModelConfig(HumanModelConfig):
    name: Literal["human_thought"] = Field(default="human_thought", description="Model name.")

    per_instance_cost_limit: float = Field(
        default=0.0, description="Cost limit for every instance (task). This is a dummy value here."
    )
    total_cost_limit: float = Field(
        default=0.0, description="Cost limit for all instances (tasks). This is a dummy value here."
    )
    cost_per_call: float = 0.0

    model_config = ConfigDict(extra="forbid")


ModelConfig = Annotated[
    GenericAPIModelConfig
    | ReplayModelConfig
    | HumanModelConfig
    | HumanThoughtModelConfig,
    Field(union_mode="left_to_right"),
]


class GlobalStats(PydanticBaseModel):


    total_cost: float = 0


    last_query_timestamp: float = 0


GLOBAL_STATS = GlobalStats()


GLOBAL_STATS_LOCK = Lock()


@dataclass
class WhiteboxTopKEntry:
    token_id: int
    logprob: float


class InstanceStats(PydanticBaseModel):


    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) + getattr(other, field) for field in self.model_fields.keys()},
        )

    def __sub__(self, other: InstanceStats) -> InstanceStats:
        return InstanceStats(
            **{field: getattr(self, field) - getattr(other, field) for field in self.model_fields.keys()},
        )


def _normalize_vllm_step_dict(step_logprobs: dict[int, Any], top_k: int) -> list[dict[str, float | int]]:
    items = [(tok_id, lp) for tok_id, lp in step_logprobs.items()]
    items.sort(key=lambda kv: (kv[1].rank if kv[1].rank is not None else 10**9))
    return [
        {"token_id": int(tok_id), "logprob": float(lp.logprob)}
        for tok_id, lp in items[:top_k]
    ]


def _normalize_vllm_logprobs(
    sample_logprobs: list[dict[int, Any] | None] | None,
    token_ids: list[int],
    top_k: int,
) -> tuple[list[float], list[list[dict[str, float | int]]]]:
    if sample_logprobs is None:
        return [0.0 for _ in token_ids], [[] for _ in token_ids]

    token_logprobs: list[float] = []
    top_k_per_step: list[list[dict[str, float | int]]] = []
    for i, tok_id in enumerate(token_ids):
        step_dict = sample_logprobs[i] if i < len(sample_logprobs) else None
        if not step_dict:
            token_logprobs.append(0.0)
            top_k_per_step.append([])
            continue
        sampled_lp = step_dict.get(tok_id)
        token_logprobs.append(0.0 if sampled_lp is None else float(sampled_lp.logprob))
        top_k_per_step.append(_normalize_vllm_step_dict(step_dict, top_k))
    return token_logprobs, top_k_per_step


def _extract_vllm_message_text(completion_output: Any) -> str:
    for carrier in (
        completion_output,
        getattr(completion_output, "message", None),
        getattr(completion_output, "assistant_message", None),
    ):
        if carrier is None:
            continue
        if isinstance(carrier, dict):
            value = carrier.get("content")
        else:
            value = getattr(carrier, "content", None)
        if isinstance(value, str):
            return value
    return str(getattr(completion_output, "text", "") or "")


class AbstractModel(ABC):
    def __init__(self, config: ModelConfig, tools: ToolConfig):
        self.config: ModelConfig
        self.stats: InstanceStats

    def reset_stats(self):
        self.stats = InstanceStats()

    @abstractmethod
    def query(self, history: History, action_prompt: str = "> ") -> dict: ...

    def encode_text(self, text: str) -> list[int] | None:
        return None

    def render_prompt(self, history: History) -> str:
        return json.dumps(history, ensure_ascii=False, indent=2)

    @property
    def instance_cost_limit(self) -> float:

        return 0


def _handle_raise_commands(action: str) -> None:
    if action == "raise_runtime":
        raise SwerexException()
    elif action == "raise_cost":
        raise CostLimitExceededError()
    elif action == "raise_context":
        raise ContextWindowExceededError()
    elif action.startswith("raise_function_calling"):
        parts = shlex.split(action)
        error_code = parts[1]
        if len(parts) == 3:
            error_message = parts[2]
        assert len(parts) < 4
        raise FunctionCallingFormatError(error_message, error_code)


class HumanModel(AbstractModel):
    def __init__(self, config: HumanModelConfig, tools: ToolConfig):

        self.logger = get_logger("swea-lm", emoji="🤖")
        self.config: HumanModelConfig = config
        self.stats = InstanceStats()


        self.multi_line_command_endings = {
            command.name: command.end_name for command in tools.commands if command.end_name is not None
        }
        self._readline_histfile = REPO_ROOT / ".swe-agent-human-history"
        self._load_readline_history()

    def _load_readline_history(self) -> None:

        if readline is None:
            return
        if self._readline_histfile.is_file():
            self.logger.debug(f"Loading readline history from {self._readline_histfile}")
            readline.read_history_file(self._readline_histfile)

    def _save_readline_history(self) -> None:

        if readline is None:
            return
        readline.write_history_file(self._readline_histfile)

    def _update_stats(
        self,
    ) -> None:
        self.stats.instance_cost += self.config.cost_per_call
        self.stats.api_calls += 1
        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            msg = f"Instance cost limit exceeded: {self.stats.instance_cost} > {self.config.per_instance_cost_limit}"
            raise InstanceCostLimitExceededError(msg)
        if 0 < self.config.total_cost_limit < self.stats.instance_cost:
            msg = f"Total cost limit exceeded: {self.stats.instance_cost} > {self.config.total_cost_limit}"
            raise TotalCostLimitExceededError(msg)

    def _query(
        self,
        history: History,
        action_prompt: str = "> ",
    ) -> dict:

        action = input(action_prompt)
        self._save_readline_history()
        command_name = action.split()[0] if action.strip() else ""


        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:

                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        else:

            action = action.encode("utf8").decode("unicode_escape")
        if action.strip() and action.strip().split()[0] == "spend_money":
            money = float(action.strip().split()[1])
            self.stats.instance_cost += money
            action = f"echo 'Spent {money} dollars'"
        _handle_raise_commands(action)
        self._update_stats()
        return {"message": action}

    def query(self, history: History, action_prompt: str = "> ", n: int | None = None, **kwargs) -> dict | list[dict]:

        out = []
        n_samples = n or 1
        for _ in range(n_samples):
            try:
                out.append(self._query(history, action_prompt))
            except KeyboardInterrupt:
                print("^C (exit with ^D)")
                out.append(self.query(history, action_prompt))
            except EOFError:
                if self.config.catch_eof:
                    print("\nGoodbye!")
                    out.append({"message": "exit"})
                else:

                    raise
        if n is None:
            return out[0]
        return out


class HumanThoughtModel(HumanModel):
    def query(self, history: History, **kwargs) -> dict:

        thought_all = ""
        thought = input("Thought (end w/ END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super()._query(history, action_prompt="Action: ")["message"]

        return {"message": f"{thought_all}\n```\n{action}\n```"}


class ReplayModel(AbstractModel):
    def __init__(self, config: ReplayModelConfig, tools: ToolConfig):

        self.config = config
        self.stats = InstanceStats()

        if not self.config.replay_path.exists():
            msg = f"Replay file {self.config.replay_path} not found"
            raise FileNotFoundError(msg)

        self._replays = [
            list(json.loads(x).values())[0] for x in Path(self.config.replay_path).read_text().splitlines(keepends=True)
        ]
        self._replay_idx = 0
        self._action_idx = 0
        self.use_function_calling = tools.use_function_calling
        self.submit_command = tools.submit_command
        self.logger = get_logger("swea-lm", emoji="🤖")

    def _next_replay(self) -> None:

        self._replay_idx += 1
        self._action_idx = 0

    def query(self, history: History) -> dict:

        self.stats.api_calls += 1
        actions = self._replays[self._replay_idx]
        try:
            action = actions[self._action_idx]
        except IndexError:

            self.logger.error("Reached end of replay trajectory without submitting. Submitting now.")
            if self.use_function_calling:
                action = {
                    "message": f"Calling `{self.submit_command}` to submit.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_submit",
                            "function": {
                                "name": self.submit_command,
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            else:
                action = f"```\n{self.submit_command}\n```"

        self._action_idx += 1


        if isinstance(action, str) and action == "submit":
            self._next_replay()
            return {"message": action}


        if isinstance(action, dict):
            return action
        return {"message": action}


class LiteLLMModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):


        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if tools.use_function_calling:
            if not litellm.utils.supports_function_calling(model=self.config.name):
                msg = (
                    f"Model {self.config.name} does not support function calling. If your model"
                    " does not support function calling, you can use `parse_function='thought_action'` instead. "
                    "See https://swe-agent.com/latest/faq/ for more information."
                )
                self.logger.warning(msg)
        if self.config.litellm_model_registry is not None:
            with open(self.config.litellm_model_registry) as f:
                model_costs = json.load(f)
                litellm.register_model(model_costs)
        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            self.model_max_input_tokens = litellm.model_cost.get(self.config.name, {}).get("max_input_tokens")

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            self.model_max_output_tokens = litellm.model_cost.get(self.config.name, {}).get("max_output_tokens")


            is_claude_3_7 = "claude-3-7-sonnet" in self.config.name or "claude-sonnet-4" in self.config.name
            has_128k_beta_header = (
                self.config.completion_kwargs.get("extra_headers", {}).get("anthropic-beta") == "output-128k-2025-02-19"
            )
            if is_claude_3_7 and not has_128k_beta_header:
                self.model_max_output_tokens = 64000
                self.logger.warning(
                    "Claude 3.7/4 models do not support 128k context by default. "
                    "Setting max output tokens to 64k. To enable 128k context, please set the "
                    "completion_kwargs to {'extra_headers': {'anthropic-beta': 'output-128k-2025-02-19'}}."
                )

        self.lm_provider = litellm.model_cost.get(self.config.name, {}).get("litellm_provider", self.config.name)
        self.custom_tokenizer = None
        if self.config.custom_tokenizer is not None:
            self.custom_tokenizer = litellm.utils.create_pretrained_tokenizer(**self.config.custom_tokenizer)

    @property
    def instance_cost_limit(self) -> float:

        return self.config.per_instance_cost_limit

    def _update_stats(self, *, input_tokens: int, output_tokens: int, cost: float) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1


        self.logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.instance_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        self.logger.debug(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={GLOBAL_STATS.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )


        if 0 < self.config.total_cost_limit < GLOBAL_STATS.total_cost:
            self.logger.warning(f"Cost {GLOBAL_STATS.total_cost:.2f} exceeds limit {self.config.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise TotalCostLimitExceededError(msg)

        if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost:
            self.logger.warning(
                f"Cost {self.stats.instance_cost:.2f} exceeds limit {self.config.per_instance_cost_limit:.2f}"
            )
            msg = "Instance cost limit exceeded"
            raise InstanceCostLimitExceededError(msg)

        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            self.logger.warning(f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}")
            msg = "Per instance call limit exceeded"
            raise InstanceCallLimitExceededError(msg)

    def _sleep(self) -> None:
        elapsed_time = time.time() - GLOBAL_STATS.last_query_timestamp
        if elapsed_time < self.config.delay:
            time.sleep(self.config.delay - elapsed_time)
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.last_query_timestamp = time.time()

    def _single_query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        self._sleep()

        messages_no_cache_control = copy.deepcopy(messages)
        for message in messages_no_cache_control:
            if "cache_control" in message:
                del message["cache_control"]
            if "thinking_blocks" in message:
                del message["thinking_blocks"]
        input_tokens: int = litellm.utils.token_counter(
            messages=messages_no_cache_control,
            model=self.custom_tokenizer["identifier"] if self.custom_tokenizer is not None else self.config.name,
            custom_tokenizer=self.custom_tokenizer,
        )
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.config.api_base:

            extra_args["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools

        completion_kwargs = copy.deepcopy(self.config.completion_kwargs)
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens


        if "extra_headers" not in completion_kwargs:
            completion_kwargs["extra_headers"] = {}
        if "User-Agent" not in completion_kwargs["extra_headers"]:
            completion_kwargs["extra_headers"]["User-Agent"] = f"swe-agent/{__version__}"

        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(
                model=self.config.name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                **completion_kwargs,
                **extra_args,
                n=n,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        self.logger.debug(f"Response: {response}")
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.name)
        except Exception as e:
            self.logger.debug(f"Error calculating cost: {e}, setting cost to 0.")
            if self.config.per_instance_cost_limit > 0 or self.config.total_cost_limit > 0:
                msg = (
                    f"Error calculating cost: {e} for your model {self.config.name}. If this is ok "
                    "(local models, etc.), please make sure you set `per_instance_cost_limit` and "
                    "`total_cost_limit` to 0 to disable this safety check."
                )
                self.logger.error(msg)
                raise ModelConfigurationError(msg)
            cost = 0
        choices: litellm.types.utils.Choices = response.choices
        n_choices = n if n is not None else 1
        outputs = []
        output_tokens = 0
        for i in range(n_choices):
            output = choices[i].message.content or ""
            output_tokens += litellm.utils.token_counter(
                text=output,
                model=self.custom_tokenizer["identifier"] if self.custom_tokenizer is not None else self.config.name,
                custom_tokenizer=self.custom_tokenizer,
            )
            output_dict = {"message": output}
            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:
                    tool_calls = [call.to_dict() for call in response.choices[i].message.tool_calls]
                else:
                    tool_calls = []
                output_dict["tool_calls"] = tool_calls
            if (
                hasattr(response.choices[i].message, "thinking_blocks")
                and response.choices[i].message.thinking_blocks
            ):
                output_dict["thinking_blocks"] = response.choices[i].message.thinking_blocks
            outputs.append(output_dict)
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return outputs

    def _query(
        self, messages: list[dict[str, str]], n: int | None = None, temperature: float | None = None
    ) -> list[dict]:
        if n is None:
            return self._single_query(messages, temperature=temperature)
        outputs = []

        for _ in range(n):
            outputs.extend(self._single_query(messages))
        return outputs

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception_info = ""
            if attempt.retry_state.outcome is not None and attempt.retry_state.outcome.exception() is not None:
                exception = attempt.retry_state.outcome.exception()
                exception_info = f" due to {exception.__class__.__name__}: {str(exception)}"

            self.logger.warning(
                f"Retrying LM query: attempt {attempt.retry_state.attempt_number} "
                f"(slept for {attempt.retry_state.idle_for:.2f}s)"
                f"{exception_info}"
            )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(min=self.config.retry.min_wait, max=self.config.retry.max_wait),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    ContextWindowExceededError,
                    CostLimitExceededError,
                    RuntimeError,
                    litellm.exceptions.UnsupportedParamsError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.PermissionDeniedError,
                    litellm.exceptions.ContextWindowExceededError,
                    litellm.exceptions.APIError,
                    litellm.exceptions.ContentPolicyViolationError,
                    TypeError,
                    litellm.exceptions.AuthenticationError,
                    ContentPolicyViolationError,
                    ModelConfigurationError,
                    KeyboardInterrupt,
                    IndexError,
                )
            ),
            before_sleep=retry_warning,
        ):
            with attempt:
                result = self._query(messages, n=n, temperature=temperature)
        if n is None or n == 1:
            return result[0]
        return result

    def render_prompt(self, history: History) -> str:
        return json.dumps(self._history_to_messages(history), ensure_ascii=False, indent=2)

    def _history_to_messages(
        self,
        history: History,
    ) -> list[dict[str, str]]:
        history = copy.deepcopy(history)

        def get_role(history_item: HistoryItem) -> str:
            if history_item["role"] == "system":
                return "user" if self.config.convert_system_to_user else "system"
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)
            if role == "tool":
                message = {
                    "role": role,
                    "content": history_item["content"],

                    "tool_call_id": history_item["tool_call_ids"][0],
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {"role": role, "content": history_item["content"], "tool_calls": tool_calls}
                if thinking_blocks := history_item.get("thinking_blocks"):
                    message["thinking_blocks"] = thinking_blocks
            else:
                message = {"role": role, "content": history_item["content"]}
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]
            messages.append(message)
        n_cache_control = str(messages).count("cache_control")
        self.logger.debug(f"n_cache_control: {n_cache_control}")
        return messages


def _validate_vllm_tensor_parallel_size(tensor_parallel_size: int) -> int:
    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int):
        raise ValueError(
            "agent.model.tensor_parallel_size must be a positive integer; "
            f"got {type(tensor_parallel_size).__name__} {tensor_parallel_size!r}"
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            "agent.model.tensor_parallel_size must be >= 1; "
            f"got {tensor_parallel_size}"
        )
    if tensor_parallel_size > 1:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return int(tensor_parallel_size)


class VLLMModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")
        self.tensor_parallel_size = _validate_vllm_tensor_parallel_size(self.config.tensor_parallel_size)

        from transformers import AutoTokenizer
        from vllm import LLM
        try:
            from agent_uq.uq.hf_fallback import (
                configure_hf_home_cache_env,
                install_transformers_extra_special_tokens_list_compat,
                load_with_hf_home_fallback,
                resolve_hf_home_model_ref,
            )
        except ModuleNotFoundError:
            def configure_hf_home_cache_env():
                return None

            def install_transformers_extra_special_tokens_list_compat():
                return None

            def load_with_hf_home_fallback(loader, model_ref, *args, **kwargs):
                return loader(model_ref, *args, **kwargs)

            def resolve_hf_home_model_ref(model_ref, *, required_files=()):
                return model_ref

        tokenizer_name = self.config.tokenizer_path or self.config.name
        configure_hf_home_cache_env()
        install_transformers_extra_special_tokens_list_compat()
        self._tokenizer = load_with_hf_home_fallback(
            AutoTokenizer.from_pretrained,
            tokenizer_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        llm_kwargs: dict[str, Any] = {}
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len
        self._llm = LLM(
            model=resolve_hf_home_model_ref(self.config.name, required_files=("config.json",)),
            tokenizer=resolve_hf_home_model_ref(
                tokenizer_name,
                required_files=("tokenizer_config.json",),
            ),
            dtype="bfloat16",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=self.config.trust_remote_code,
            **llm_kwargs,
        )
        self.model_max_output_tokens = self.config.max_output_tokens or 512

    @property
    def instance_cost_limit(self) -> float:
        return self.config.per_instance_cost_limit

    def encode_text(self, text: str) -> list[int] | None:
        if text == "":
            return []
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _history_to_messages(self, history: History) -> list[dict[str, Any]]:

        return LiteLLMModel._history_to_messages(self, history)

    def _chat_template_kwargs(self) -> dict[str, Any] | None:
        raw_value = self.config.completion_kwargs.get("chat_template_kwargs")
        if not isinstance(raw_value, dict) or not raw_value:
            return None
        return copy.deepcopy(raw_value)

    def render_prompt(self, history: History) -> str:
        messages = self._history_to_messages(history)
        chat_template_kwargs = self._chat_template_kwargs() or {}
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                rendered = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            except TypeError:
                rendered = apply_chat_template(
                    messages,
                    tokenize=False,
                    **chat_template_kwargs,
                )
            if isinstance(rendered, str):
                return rendered
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def query(self, history: History, n: int = 1, temperature: float | None = None) -> list[dict] | dict:
        from vllm import SamplingParams

        messages = self._history_to_messages(history)
        prompt_snapshot = self.render_prompt(history)
        chat_template_kwargs = self._chat_template_kwargs()
        top_k = int(self.config.completion_kwargs.get("top_k_logprobs", 5))
        sampling_params = SamplingParams(
            n=n,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=1.0 if self.config.top_p is None else self.config.top_p,
            top_k=0 if self.config.top_k is None else int(self.config.top_k),
            max_tokens=self.model_max_output_tokens,
            logprobs=top_k,
            stop=self.config.stop or None,
        )
        outputs = self._llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        self.stats.api_calls += 1
        if not outputs:
            result = {"message": ""}
            return result if n == 1 else [result]
        request_output = outputs[0]
        completions: list[dict[str, Any]] = []
        for co in request_output.outputs:
            token_ids = list(co.token_ids)
            token_logprobs, top_k_per_step = _normalize_vllm_logprobs(co.logprobs, token_ids, top_k)
            message_text = _extract_vllm_message_text(co)
            completions.append(
                {
                    "message": message_text,
                    "raw_response": message_text,
                    "prompt": prompt_snapshot,
                    "token_ids": token_ids,
                    "token_logprobs": token_logprobs,
                    "top_k": top_k_per_step,
                    "finish_reason": co.finish_reason or "unknown",
                    "vllm_index": getattr(co, "index", 0),
                    "cumulative_logprob": getattr(co, "cumulative_logprob", None),
                }
            )
        if n == 1:
            return completions[0]
        return completions


def get_model(args: ModelConfig, tools: ToolConfig) -> AbstractModel:


    if isinstance(args, GenericAPIModelConfig) and not isinstance(
        args, HumanModelConfig | HumanThoughtModelConfig | ReplayModelConfig
    ):
        if args.name == "human":
            args = HumanModelConfig(**args.model_dump())
        elif args.name == "human_thought":
            args = HumanThoughtModelConfig(**args.model_dump())
        elif args.name == "replay":
            args = ReplayModelConfig(**args.model_dump())

    if args.name == "human":
        assert isinstance(args, HumanModelConfig), f"Expected {HumanModelConfig}, got {args}"
        return HumanModel(args, tools)
    if args.name == "human_thought":
        assert isinstance(args, HumanThoughtModelConfig), f"Expected {HumanThoughtModelConfig}, got {args}"
        return HumanThoughtModel(args, tools)
    if args.name == "replay":
        assert isinstance(args, ReplayModelConfig), f"Expected {ReplayModelConfig}, got {args}"
        return ReplayModel(args, tools)
    assert isinstance(args, GenericAPIModelConfig), f"Expected {GenericAPIModelConfig}, got {args}"
    if args.backend == "vllm":
        return VLLMModel(args, tools)
    return LiteLLMModel(args, tools)
