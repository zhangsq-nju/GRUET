from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from agent_uq.uq.hf_fallback import (
    configure_hf_home_cache_env,
    install_transformers_extra_special_tokens_list_compat,
    resolve_hf_home_model_ref,
)
import agent_uq.vllm_runtime as vllm_runtime


@dataclass
class TopKEntry:


    token_id: int
    logprob: float


@dataclass
class Candidate:


    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    top_k: list[list[TopKEntry]]
    finish_reason: str
    backend: dict[str, Any] = field(default_factory=dict)


def _normalize_step_dict(step_logprobs: dict[int, Any], top_k: int) -> list[TopKEntry]:

    items = [(tok_id, lp) for tok_id, lp in step_logprobs.items()]

    items.sort(key=lambda kv: (kv[1].rank if kv[1].rank is not None else 10**9))
    return [TopKEntry(token_id=tok_id, logprob=float(lp.logprob)) for tok_id, lp in items[:top_k]]


def _normalize_logprobs_to_top_k(
    sample_logprobs: Sequence[dict[int, Any] | None] | None,
    token_ids: Sequence[int],
    top_k: int,
) -> tuple[list[float], list[list[TopKEntry]]]:

    if sample_logprobs is None:


        return [0.0 for _ in token_ids], [[] for _ in token_ids]

    token_logprobs: list[float] = []
    top_k_per_step: list[list[TopKEntry]] = []

    for i, tok_id in enumerate(token_ids):
        step_dict = sample_logprobs[i] if i < len(sample_logprobs) else None
        if not step_dict:
            token_logprobs.append(0.0)
            top_k_per_step.append([])
            continue

        sampled_lp = step_dict.get(tok_id)
        if sampled_lp is None:
            token_logprobs.append(0.0)
        else:
            token_logprobs.append(float(sampled_lp.logprob))
        top_k_per_step.append(_normalize_step_dict(step_dict, top_k))

    return token_logprobs, top_k_per_step


def candidate_from_completion_output(
    completion_output: Any,
    top_k: int,
    backend_extra: dict[str, Any] | None = None,
) -> Candidate:

    token_ids = list(completion_output.token_ids)
    token_logprobs, top_k_per_step = _normalize_logprobs_to_top_k(
        completion_output.logprobs, token_ids, top_k
    )
    return Candidate(
        text=completion_output.text,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        top_k=top_k_per_step,
        finish_reason=completion_output.finish_reason or "unknown",
        backend=dict(backend_extra or {}),
    )


def candidate_from_chat_completion_output(
    completion_output: Any,
    top_k: int,
    backend_extra: dict[str, Any] | None = None,
) -> Candidate:

    backend = dict(backend_extra or {})
    tool_calls = _extract_native_tool_calls(completion_output)
    if tool_calls:
        backend["tool_calls"] = tool_calls
    message_text = _message_text(completion_output)
    if message_text is not None:
        backend["raw_assistant_message"] = message_text
    return candidate_from_completion_output(
        completion_output,
        top_k=top_k,
        backend_extra=backend,
    )


class VLLMSampler:


    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        **extra_llm_kwargs: Any,
    ) -> None:


        from vllm import LLM

        self.model_path = model_path
        self.tensor_parallel_size = vllm_runtime.validate_tensor_parallel_size(
            tensor_parallel_size
        )
        vllm_runtime.configure_vllm_worker_multiproc_method(
            self.tensor_parallel_size
        )
        configure_hf_home_cache_env()
        install_transformers_extra_special_tokens_list_compat()
        runtime_model_path = resolve_hf_home_model_ref(model_path, required_files=("config.json",))
        runtime_tokenizer_path = resolve_hf_home_model_ref(
            tokenizer_path or model_path,
            required_files=("tokenizer_config.json",),
        )
        self._llm = LLM(
            model=runtime_model_path,
            tokenizer=runtime_tokenizer_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            **extra_llm_kwargs,
        )

    def sample(
        self,
        prompt: str,
        *,
        n: int,
        top_k: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        decode_top_k: int | None = None,
        max_new_tokens: int = 512,
        stop: list[str] | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        compatibility_profile: str | None = None,
    ) -> list[Candidate]:

        if tools or tool_choice or compatibility_profile:
            return self._sample_chat_with_tools(
                tools=tools,
                tool_choice=tool_choice,
                compatibility_profile=compatibility_profile,
                prompt=prompt,
                n=n,
                top_k=top_k,
                temperature=temperature,
                top_p=top_p,
                decode_top_k=decode_top_k,
                max_new_tokens=max_new_tokens,
                stop=stop,
                seed=seed,
            )

        from vllm import SamplingParams

        sampling_kwargs = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "logprobs": top_k,
            "stop": stop or None,
            "seed": seed,
        }
        if decode_top_k is not None:
            sampling_kwargs["top_k"] = int(decode_top_k)
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self._llm.generate([prompt], sampling_params)
        if not outputs:
            return []
        request_output = outputs[0]
        candidates = [
            candidate_from_completion_output(
                co,
                top_k=top_k,
                backend_extra={
                    "vllm_index": co.index,
                    "cumulative_logprob": co.cumulative_logprob,
                    "compatibility_profile": compatibility_profile,
                },
            )
            for co in request_output.outputs
        ]
        return candidates

    def _validate_tool_calling_request(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        compatibility_profile: str | None,
    ) -> None:
        if not tools:
            return
        if not hasattr(self._llm, "chat"):
            raise RuntimeError(
                "VLLMSampler backend does not expose a native tool-calling execution "
                "path via LLM.chat(...); refusing to silently degrade the SWE "
                "function-calling request into plain text sampling"
            )
        if tool_choice not in {None, "required", "auto"}:
            raise RuntimeError(
                f"unsupported tool_choice {tool_choice!r} for compatibility profile "
                f"{compatibility_profile!r}"
            )

    def _sample_chat_with_tools(
        self,
        *,
        prompt: str,
        n: int,
        top_k: int,
        temperature: float,
        top_p: float,
        decode_top_k: int | None,
        max_new_tokens: int,
        stop: list[str] | None,
        seed: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        compatibility_profile: str | None,
    ) -> list[Candidate]:
        self._validate_tool_calling_request(
            tools=tools,
            tool_choice=tool_choice,
            compatibility_profile=compatibility_profile,
        )

        from vllm import SamplingParams

        sampling_kwargs = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "logprobs": top_k,
            "stop": stop or None,
            "seed": seed,
        }
        if decode_top_k is not None:
            sampling_kwargs["top_k"] = int(decode_top_k)
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=sampling_params,
            use_tqdm=False,
            tools=tools,
        )
        if not outputs:
            return []
        request_output = outputs[0]
        candidates = [
            candidate_from_chat_completion_output(
                co,
                top_k=top_k,
                backend_extra={
                    "vllm_index": co.index,
                    "cumulative_logprob": co.cumulative_logprob,
                    "compatibility_profile": compatibility_profile,
                    "tool_choice": tool_choice,
                },
            )
            for co in request_output.outputs
        ]
        if tools and not any(candidate.backend.get("tool_calls") for candidate in candidates):
            raise RuntimeError(
                "vLLM chat sampling completed, but the response did not expose native "
                "tool-call metadata required by the SWE function-calling contract"
            )
        return candidates


def _extract_native_tool_calls(completion_output: Any) -> list[dict[str, Any]]:
    for carrier in (
        completion_output,
        getattr(completion_output, "message", None),
        getattr(completion_output, "assistant_message", None),
    ):
        if carrier is None:
            continue
        if isinstance(carrier, dict):
            raw_tool_calls = carrier.get("tool_calls")
        else:
            raw_tool_calls = getattr(carrier, "tool_calls", None)
        if raw_tool_calls:
            return [_normalize_tool_call(tool_call) for tool_call in raw_tool_calls]
    return []


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return dict(tool_call)
    out: dict[str, Any] = {}
    for key in ("id", "type", "function", "name", "arguments"):
        value = getattr(tool_call, key, None)
        if value is not None:
            out[key] = value
    function = out.get("function")
    if function is not None and not isinstance(function, dict):
        out["function"] = {
            key: getattr(function, key)
            for key in ("name", "arguments")
            if getattr(function, key, None) is not None
        }
    return out


def _message_text(completion_output: Any) -> str | None:
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
    return None
