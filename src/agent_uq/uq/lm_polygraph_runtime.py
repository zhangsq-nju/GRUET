from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from agent_uq.uq.base import EstimatorInfrastructureError
from agent_uq.uq.hf_fallback import (
    configure_hf_home_cache_env,
    install_transformers_extra_special_tokens_list_compat,
    load_with_hf_home_fallback,
    resolve_hf_home_model_ref,
)
from agent_uq.uq.lm_polygraph_constants import _MODEL_CONF_DIR
from agent_uq.vllm_runtime import (
    configure_vllm_worker_multiproc_method as _configure_vllm_worker_multiproc_method,
    validate_tensor_parallel_size as _validate_tensor_parallel_size,
)


_AUXILIARY_GPU_MEMORY_UTILIZATION_ENV = "AGENT_UQ_AUXILIARY_VLLM_GPU_MEMORY_UTILIZATION"


def _facade_override(name: str, original: Any) -> Any | None:
    facade = sys.modules.get("agent_uq.uq.lm_polygraph")
    override = getattr(facade, name, None)
    if callable(override) and override is not original:
        return override
    return None


def _model_conf_dir():
    facade = sys.modules.get("agent_uq.uq.lm_polygraph")
    return getattr(facade, "_MODEL_CONF_DIR", _MODEL_CONF_DIR)


def _is_cuda_device(device: str | None) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def _resolve_torch_device(preferred: str | None) -> str:
    override = _facade_override("_resolve_torch_device", _resolve_torch_device)
    if override is not None:
        return override(preferred)
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def validate_tensor_parallel_size(tensor_parallel_size: int) -> int:
    override = _facade_override(
        "validate_tensor_parallel_size",
        validate_tensor_parallel_size,
    )
    if override is not None:
        return override(tensor_parallel_size)
    return _validate_tensor_parallel_size(tensor_parallel_size)


def configure_vllm_worker_multiproc_method(tensor_parallel_size: int) -> None:
    override = _facade_override(
        "configure_vllm_worker_multiproc_method",
        configure_vllm_worker_multiproc_method,
    )
    if override is not None:
        override(tensor_parallel_size)
        return
    _configure_vllm_worker_multiproc_method(tensor_parallel_size)


class _NLIBackend:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str | None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = _resolve_torch_device(device)
        load_kwargs: dict[str, Any] = {
            key: value
            for key, value in {
                "cache_dir": cache_dir,
                "local_files_only": True if local_files_only else None,
            }.items()
            if value is not None
        }
        self.tokenizer = load_with_hf_home_fallback(
            AutoTokenizer.from_pretrained,
            model_name,
            model_max_length=512,
            **load_kwargs,
        )
        self.model = load_with_hf_home_fallback(
            AutoModelForSequenceClassification.from_pretrained,
            model_name,
            **load_kwargs,
        )
        if _is_cuda_device(self.device):
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        label2id = {
            str(label).lower(): int(idx)
            for label, idx in self.model.config.label2id.items()
        }
        self.contradiction_id = label2id["contradiction"]
        self.neutral_id = label2id["neutral"]
        self.entailment_id = label2id["entailment"]
        self._torch = torch

    def predict_proba(self, lefts: Sequence[str], rights: Sequence[str]) -> np.ndarray:
        if len(lefts) != len(rights):
            raise ValueError("lefts and rights must have the same length")
        if not lefts:
            return np.empty((0, 3), dtype=float)
        probs: list[np.ndarray] = []
        for start in range(0, len(lefts), self.batch_size):
            batch_left = list(lefts[start : start + self.batch_size])
            batch_right = list(rights[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch_left,
                batch_right,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
            probs.append(
                self._torch.softmax(logits, dim=-1).detach().cpu().numpy()
            )
        return np.concatenate(probs, axis=0)


class _CrossEncoderBackend:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str | None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.batch_size = int(batch_size)
        self.cross_encoder = load_with_hf_home_fallback(
            CrossEncoder,
            model_name,
            cache_kwarg="cache_folder",
            device=_resolve_torch_device(device),
        )

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=float)
        scores = self.cross_encoder.predict(
            list(pairs),
            batch_size=self.batch_size,
        )
        return np.asarray(scores, dtype=float)


@dataclass(frozen=True)
class ResolvedAuxiliaryModelConfig:
    model_ref: str
    model_path: str
    tokenizer_path: str
    dtype: str
    trust_remote_code: bool
    auxiliary_vllm: dict[str, Any]

    @property
    def tensor_parallel_size(self) -> int:
        return int(self.auxiliary_vllm["tensor_parallel_size"])

    @property
    def gpu_memory_utilization(self) -> float:
        return float(self.auxiliary_vllm["gpu_memory_utilization"])


@contextlib.contextmanager
def _temporary_cuda_visible_devices(visible_devices: str | None):
    if visible_devices is None:
        yield
        return
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def _parse_auxiliary_worker_gpu(worker_device: str | None) -> str | None:
    if not _is_cuda_device(worker_device):
        return None
    parts = str(worker_device).split(":", 1)
    if len(parts) != 2 or not parts[1]:
        return "0"
    return parts[1]


def _auxiliary_runtime_signature(config: ResolvedAuxiliaryModelConfig) -> dict[str, Any]:
    return {
        "model_path": config.model_path,
        "tokenizer_path": config.tokenizer_path,
        "dtype": config.dtype,
        "trust_remote_code": bool(config.trust_remote_code),
        "tensor_parallel_size": int(config.tensor_parallel_size),
    }


def _freeze_auxiliary_runtime_signature(signature: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(key), value) for key, value in signature.items()))


def _auxiliary_cache_key(
    config: ResolvedAuxiliaryModelConfig,
    *,
    worker_device: str | None,
) -> tuple[str, tuple[tuple[str, Any], ...], str | None]:
    placement_key = (
        _parse_auxiliary_worker_gpu(worker_device)
        if int(config.tensor_parallel_size) == 1
        else None
    )
    return (
        config.model_ref,
        _freeze_auxiliary_runtime_signature(_auxiliary_runtime_signature(config)),
        placement_key,
    )


def _load_auxiliary_tokenizer(config: ResolvedAuxiliaryModelConfig):
    override = _facade_override("_load_auxiliary_tokenizer", _load_auxiliary_tokenizer)
    if override is not None:
        return override(config)
    from transformers import AutoTokenizer

    cache_key = (config.tokenizer_path, bool(config.trust_remote_code))
    cached = _AUXILIARY_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tokenizer = load_with_hf_home_fallback(
        AutoTokenizer.from_pretrained,
        config.tokenizer_path,
        trust_remote_code=bool(config.trust_remote_code),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _AUXILIARY_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _construct_auxiliary_vllm_engine(
    config: ResolvedAuxiliaryModelConfig,
    *,
    worker_device: str | None,
):
    override = _facade_override(
        "_construct_auxiliary_vllm_engine",
        _construct_auxiliary_vllm_engine,
    )
    if override is not None:
        return override(config, worker_device=worker_device)
    from vllm import LLM

    visible_device = None
    if int(config.tensor_parallel_size) == 1:
        visible_device = _parse_auxiliary_worker_gpu(worker_device)
    with _temporary_cuda_visible_devices(visible_device):
        tensor_parallel_size = validate_tensor_parallel_size(int(config.tensor_parallel_size))
        configure_vllm_worker_multiproc_method(tensor_parallel_size)
        configure_hf_home_cache_env()
        install_transformers_extra_special_tokens_list_compat()
        return LLM(
            model=resolve_hf_home_model_ref(config.model_path, required_files=("config.json",)),
            tokenizer=resolve_hf_home_model_ref(
                config.tokenizer_path,
                required_files=("tokenizer_config.json",),
            ),
            dtype=config.dtype,
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            trust_remote_code=bool(config.trust_remote_code),
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
        )


class _AuxiliaryVLLMModel:
    def __init__(
        self,
        *,
        config: ResolvedAuxiliaryModelConfig,
        worker_device: str | None,
    ) -> None:
        self.config = config
        self.worker_device = worker_device
        self.tokenizer = _load_auxiliary_tokenizer(config)
        self.llm = _construct_auxiliary_vllm_engine(
            config,
            worker_device=worker_device,
        )

    def generate_one_token_logprob(self, prompt: str) -> float:
        from vllm import SamplingParams

        outputs = self.llm.generate(
            [prompt],
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                logprobs=1,
            ),
        )
        if not outputs or not outputs[0].outputs:
            raise EstimatorInfrastructureError(
                "auxiliary vLLM one-token generation returned no output"
            )
        completion = outputs[0].outputs[0]
        token_ids = list(getattr(completion, "token_ids", []) or [])
        if len(token_ids) != 1:
            raise EstimatorInfrastructureError(
                "auxiliary vLLM one-token generation did not return exactly one token"
            )
        logprobs = getattr(completion, "logprobs", None)
        if not logprobs or not logprobs[0]:
            raise EstimatorInfrastructureError(
                "auxiliary vLLM one-token generation returned no token logprobs"
            )
        token_id = int(token_ids[0])
        sampled = logprobs[0].get(token_id)
        if sampled is None:
            raise EstimatorInfrastructureError(
                "auxiliary vLLM one-token generation omitted the sampled token logprob"
            )
        return float(sampled.logprob)

    def generate_text(
        self,
        prompt: str,
        *,
        min_new_tokens: int = 0,
        max_new_tokens: int,
    ) -> str:
        from vllm import SamplingParams

        outputs = self.llm.generate(
            [prompt],
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                min_tokens=int(min_new_tokens),
                max_tokens=int(max_new_tokens),
            ),
        )
        if not outputs or not outputs[0].outputs:
            raise EstimatorInfrastructureError("auxiliary vLLM generation returned no output")
        return str(outputs[0].outputs[0].text)


_NLI_CACHE: dict[tuple[str, int, str], _NLIBackend] = {}
_CROSS_ENCODER_CACHE: dict[tuple[str, int, str], _CrossEncoderBackend] = {}
_AUXILIARY_CONFIG_CACHE: dict[tuple[str, str | None, str], ResolvedAuxiliaryModelConfig] = {}
_AUXILIARY_TOKENIZER_CACHE: dict[tuple[str, bool], Any] = {}
_AUXILIARY_MODEL_CACHE: dict[
    tuple[str, tuple[tuple[str, Any], ...], str | None],
    _AuxiliaryVLLMModel,
] = {}


def _release_lm_polygraph_gpu_state() -> None:

    for model in list(_AUXILIARY_MODEL_CACHE.values()):
        llm = getattr(model, "llm", None)
        llm_engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            with contextlib.suppress(Exception):
                shutdown()
    _NLI_CACHE.clear()
    _CROSS_ENCODER_CACHE.clear()
    _AUXILIARY_MODEL_CACHE.clear()


def get_nli_backend(
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> _NLIBackend:
    override = _facade_override("get_nli_backend", get_nli_backend)
    if override is not None:
        return override(model_name=model_name, batch_size=batch_size, device=device)
    resolved_device = _resolve_torch_device(device)
    key = (model_name, int(batch_size), resolved_device)
    if key not in _NLI_CACHE:
        try:
            _NLI_CACHE[key] = _NLIBackend(
                model_name=model_name,
                batch_size=int(batch_size),
                device=resolved_device,
            )
        except Exception as exc:
            raise EstimatorInfrastructureError(
                f"failed to load NLI backend {model_name!r}: {exc}"
            ) from exc
    return _NLI_CACHE[key]


def get_cross_encoder_backend(
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> _CrossEncoderBackend:
    override = _facade_override(
        "get_cross_encoder_backend",
        get_cross_encoder_backend,
    )
    if override is not None:
        return override(model_name=model_name, batch_size=batch_size, device=device)
    resolved_device = _resolve_torch_device(device)
    key = (model_name, int(batch_size), resolved_device)
    if key not in _CROSS_ENCODER_CACHE:
        try:
            _CROSS_ENCODER_CACHE[key] = _CrossEncoderBackend(
                model_name=model_name,
                batch_size=int(batch_size),
                device=resolved_device,
            )
        except Exception as exc:
            raise EstimatorInfrastructureError(
                f"failed to load cross-encoder backend {model_name!r}: {exc}"
            ) from exc
    return _CROSS_ENCODER_CACHE[key]


def resolve_auxiliary_model_config(model_ref: str) -> ResolvedAuxiliaryModelConfig:
    override = _facade_override(
        "resolve_auxiliary_model_config",
        resolve_auxiliary_model_config,
    )
    if override is not None:
        return override(model_ref)
    if model_ref == "sampled_model":
        raise EstimatorInfrastructureError(
            "auxiliary_model_ref=sampled_model must be materialized before estimator construction"
        )
    gpu_memory_override = os.environ.get(_AUXILIARY_GPU_MEMORY_UTILIZATION_ENV)
    source_override = json.loads(os.environ.get("AGENT_UQ_AUXILIARY_SOURCE_MODEL") or "{}")
    if source_override.get("model_ref") != model_ref:
        source_override = {}
    cache_key = (model_ref, gpu_memory_override, json.dumps(source_override, sort_keys=True))
    cached = _AUXILIARY_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    path = _model_conf_dir() / f"{model_ref}.yaml"
    if not path.is_file():
        raise EstimatorInfrastructureError(
            f"unknown auxiliary model_ref {model_ref!r}; expected config at {path}"
        )
    try:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise EstimatorInfrastructureError(
            f"failed to read auxiliary model config {path}: {exc}"
        ) from exc
    if not isinstance(cfg, dict):
        raise EstimatorInfrastructureError(
            f"auxiliary model config at {path} did not resolve to a mapping"
        )
    auxiliary_vllm = cfg.get("auxiliary_vllm")
    if not isinstance(auxiliary_vllm, dict):
        raise EstimatorInfrastructureError(
            f"auxiliary model config {path} must define an auxiliary_vllm mapping"
        )
    if "tensor_parallel_size" not in auxiliary_vllm:
        raise EstimatorInfrastructureError(
            f"auxiliary model config {path} must define auxiliary_vllm.tensor_parallel_size"
        )
    if "gpu_memory_utilization" not in auxiliary_vllm:
        raise EstimatorInfrastructureError(
            f"auxiliary model config {path} must define auxiliary_vllm.gpu_memory_utilization"
        )
    auxiliary_vllm = dict(auxiliary_vllm)
    if source_override:
        cfg.update(model_path=source_override["model_path"], tokenizer_path=source_override["tokenizer_path"])
        auxiliary_vllm["tensor_parallel_size"] = int(source_override["tensor_parallel_size"])
    if gpu_memory_override not in (None, ""):
        try:
            auxiliary_vllm["gpu_memory_utilization"] = float(gpu_memory_override)
        except ValueError as exc:
            raise EstimatorInfrastructureError(
                f"{_AUXILIARY_GPU_MEMORY_UTILIZATION_ENV} must be a float"
            ) from exc

    resolved = ResolvedAuxiliaryModelConfig(
        model_ref=model_ref,
        model_path=str(cfg["model_path"]),
        tokenizer_path=str(cfg.get("tokenizer_path") or cfg["model_path"]),
        dtype=str(cfg.get("dtype") or "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        auxiliary_vllm=auxiliary_vllm,
    )
    _AUXILIARY_CONFIG_CACHE[cache_key] = resolved
    return resolved


def get_auxiliary_model(
    model_ref: str,
    *,
    worker_device: str | None = None,
) -> _AuxiliaryVLLMModel:
    override = _facade_override("get_auxiliary_model", get_auxiliary_model)
    if override is not None:
        return override(model_ref, worker_device=worker_device)
    config = resolve_auxiliary_model_config(model_ref)
    key = _auxiliary_cache_key(config, worker_device=worker_device)
    if key not in _AUXILIARY_MODEL_CACHE:
        try:
            _AUXILIARY_MODEL_CACHE[key] = _AuxiliaryVLLMModel(
                config=config,
                worker_device=worker_device,
            )
        except Exception as exc:
            raise EstimatorInfrastructureError(
                f"failed to load auxiliary causal model {model_ref!r}: {exc}"
            ) from exc
    return _AUXILIARY_MODEL_CACHE[key]
