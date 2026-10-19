from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

import agent_uq.uq.graphuq_adapter
import agent_uq.uq.lm_polygraph
import agent_uq.uq.reasoning_contenders
from agent_uq.scoring.execution import ScoreExecutionPlan
from agent_uq.uq.base import get_estimator
from agent_uq.uq.lm_polygraph import (
    derive_cross_encoder_execution_mode,
    resolve_cross_encoder_device,
    resolve_auxiliary_model_config,
)
from agent_uq.uq.pairwise_nli_artifact import derive_nli_execution_mode


_ESTIMATOR_METADATA_FIELDS = frozenset({"name", "class_path"})


@dataclass(frozen=True)
class EstimatorParamViews:


    constructor_kwargs: dict[str, Any]
    persisted_params: dict[str, Any]


@dataclass(frozen=True)
class RuntimeEstimatorViews:


    constructor_kwargs: dict[str, Any]
    persisted_params: dict[str, Any]
    runtime_metadata: dict[str, Any]


def _score_execution_runtime_metadata(plan: ScoreExecutionPlan) -> dict[str, Any]:
    return {
        "score_execution": {
            "num_workers": int(plan.num_workers),
            "worker_index": int(
                0
                if plan.worker_index is None and plan.num_workers == 1
                else plan.worker_index
                if plan.worker_index is not None
                else 0
            ),
            "worker_device": plan.assigned_device,
        }
    }


def _resolve_graphuq_runtime_binding(
    *,
    explicit_device: str | None,
    assigned_device: str | None,
    execution_mode_hint: str | None,
) -> tuple[str, dict[str, Any]]:
    from agent_uq.uq.graphuq_adapter import resolve_graphuq_nli_device

    resolved_device = assigned_device or resolve_graphuq_nli_device(explicit_device)
    execution_mode = execution_mode_hint or derive_nli_execution_mode(resolved_device)
    return execution_mode, {
        "configured_device": explicit_device,
        "resolved_device": resolved_device,
        "execution_mode": execution_mode,
    }


def _resolve_graphuq_action_similarity_runtime_binding(
    *,
    explicit_device: str | None,
    assigned_device: str | None,
) -> dict[str, Any]:
    from agent_uq.uq.graphuq_adapter import resolve_graphuq_action_similarity_device

    resolved_device = assigned_device or resolve_graphuq_action_similarity_device(
        explicit_device
    )
    return {
        "configured_device": explicit_device,
        "resolved_device": resolved_device,
    }


def _score_execution_cross_encoder_device_hint(
    plan: ScoreExecutionPlan,
) -> str | None:
    if plan.assigned_device is not None:
        return plan.assigned_device
    if plan.devices:
        return plan.devices[0]
    return None


def _resolve_cross_encoder_runtime_binding(
    *,
    explicit_device: str | None,
    assigned_device: str | None,
    default_device_hint: str | None,
    execution_mode_hint: str | None,
) -> tuple[str, dict[str, Any]]:
    resolved_device = (
        assigned_device
        or explicit_device
        or default_device_hint
        or _resolve_cross_encoder_device(None)
    )
    execution_mode = execution_mode_hint or derive_cross_encoder_execution_mode(
        resolved_device
    )
    return execution_mode, {
        "configured_device": explicit_device,
        "resolved_device": resolved_device,
        "execution_mode": execution_mode,
    }


def _resolve_cross_encoder_device(preferred: str | None) -> str:
    return resolve_cross_encoder_device(preferred)


def _get_estimator(estimator_name: str, **kwargs: Any) -> Any:
    return get_estimator(estimator_name, **kwargs)


def apply_score_execution_to_param_views(
    estimator_name: str,
    param_views: EstimatorParamViews,
    *,
    plan: ScoreExecutionPlan,
) -> RuntimeEstimatorViews:

    constructor_kwargs = dict(param_views.constructor_kwargs)
    persisted_params = dict(param_views.persisted_params)
    runtime_metadata = _score_execution_runtime_metadata(plan)

    explicit_nli_device = constructor_kwargs.get("nli_device")
    supports_explicit_nli_device = (
        "nli_device" in constructor_kwargs or estimator_name == "gut_up"
    )
    explicit_action_similarity_device = constructor_kwargs.get(
        "action_similarity_device"
    )
    supports_explicit_action_similarity_device = (
        "action_similarity_device" in constructor_kwargs
        or estimator_name == "gut_up"
    )
    explicit_cross_encoder_device = constructor_kwargs.get("cross_encoder_device")
    supports_explicit_cross_encoder_device = "cross_encoder_device" in constructor_kwargs
    supports_auxiliary_worker_device = (
        estimator_name in {"p_true", "verbalized_2s"}
        and "auxiliary_model_ref" in constructor_kwargs
    )

    if supports_auxiliary_worker_device:
        constructor_kwargs["auxiliary_worker_device"] = plan.assigned_device
        auxiliary = resolve_auxiliary_model_config(constructor_kwargs["auxiliary_model_ref"])
        runtime_metadata["auxiliary"] = {
            "worker_device": plan.assigned_device,
            "model_ref": auxiliary.model_ref,
            "model_path": auxiliary.model_path,
            "tokenizer_path": auxiliary.tokenizer_path,
            "tensor_parallel_size": auxiliary.tensor_parallel_size,
            "gpu_memory_utilization": auxiliary.gpu_memory_utilization,
        }

    if plan.assigned_device is not None and supports_explicit_nli_device:
        if (
            explicit_nli_device is not None
            and explicit_nli_device != plan.assigned_device
        ):
            raise ValueError(
                "score_execution worker device conflicts with estimator.nli_device: "
                f"worker={plan.assigned_device!r}, estimator={explicit_nli_device!r}"
            )
        constructor_kwargs["nli_device"] = plan.assigned_device

    if plan.assigned_device is not None and supports_explicit_action_similarity_device:
        if (
            explicit_action_similarity_device is not None
            and explicit_action_similarity_device != plan.assigned_device
        ):
            raise ValueError(
                "score_execution worker device conflicts with "
                "estimator.action_similarity_device: "
                f"worker={plan.assigned_device!r}, "
                f"estimator={explicit_action_similarity_device!r}"
            )
        constructor_kwargs["action_similarity_device"] = plan.assigned_device

    if plan.assigned_device is not None and supports_explicit_cross_encoder_device:
        if (
            explicit_cross_encoder_device is not None
            and explicit_cross_encoder_device != plan.assigned_device
        ):
            raise ValueError(
                "score_execution worker device conflicts with "
                "estimator.cross_encoder_device: "
                f"worker={plan.assigned_device!r}, "
                f"estimator={explicit_cross_encoder_device!r}"
            )
        constructor_kwargs["cross_encoder_device"] = plan.assigned_device

    if estimator_name == "gut_up":
        execution_mode, nli_metadata = _resolve_graphuq_runtime_binding(
            explicit_device=explicit_nli_device,
            assigned_device=plan.assigned_device,
            execution_mode_hint=plan.execution_mode,
        )
        if "nli_batch_size" in constructor_kwargs:
            nli_metadata["batch_size"] = int(constructor_kwargs["nli_batch_size"])
        persisted_params["nli_execution_mode"] = execution_mode
        runtime_metadata["nli"] = nli_metadata
        action_similarity_metadata = (
            _resolve_graphuq_action_similarity_runtime_binding(
                explicit_device=explicit_action_similarity_device,
                assigned_device=plan.assigned_device,
            )
        )
        if "action_similarity_batch_size" in constructor_kwargs:
            action_similarity_metadata["batch_size"] = int(
                constructor_kwargs["action_similarity_batch_size"]
            )
        if "action_similarity_model_name" in constructor_kwargs:
            action_similarity_metadata["model_name"] = str(
                constructor_kwargs["action_similarity_model_name"]
            )
        runtime_metadata["action_similarity"] = action_similarity_metadata
    elif "nli_execution_mode" in persisted_params and plan.execution_mode is not None:
        persisted_params["nli_execution_mode"] = plan.execution_mode
        nli_metadata = {
            "configured_device": explicit_nli_device,
            "resolved_device": plan.assigned_device or explicit_nli_device,
            "execution_mode": plan.execution_mode,
        }
        if "nli_batch_size" in constructor_kwargs:
            nli_metadata["batch_size"] = int(constructor_kwargs["nli_batch_size"])
        runtime_metadata["nli"] = nli_metadata

    if supports_explicit_cross_encoder_device:
        cross_encoder_device_hint = _score_execution_cross_encoder_device_hint(plan)
        execution_mode_hint = (
            derive_cross_encoder_execution_mode(cross_encoder_device_hint)
            if cross_encoder_device_hint is not None
            else None
        )
        execution_mode, cross_encoder_metadata = _resolve_cross_encoder_runtime_binding(
            explicit_device=explicit_cross_encoder_device,
            assigned_device=plan.assigned_device,
            default_device_hint=cross_encoder_device_hint,
            execution_mode_hint=execution_mode_hint,
        )
        persisted_params["cross_encoder_execution_mode"] = execution_mode
        if "cross_encoder_batch_size" in constructor_kwargs:
            cross_encoder_metadata["batch_size"] = int(
                constructor_kwargs["cross_encoder_batch_size"]
            )
        runtime_metadata["cross_encoder"] = cross_encoder_metadata

    return RuntimeEstimatorViews(
        constructor_kwargs=constructor_kwargs,
        persisted_params=persisted_params,
        runtime_metadata=runtime_metadata,
    )


def _derive_constructor_kwargs(
    cfg_estimator: DictConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    sampled_model_name: str | None = None,
) -> tuple[str | None, dict[str, Any]]:

    if isinstance(cfg_estimator, DictConfig):
        composed = OmegaConf.to_container(cfg_estimator, resolve=True)
    else:
        composed = dict(cfg_estimator)
    if not isinstance(composed, dict):
        raise TypeError(
            f"derive_effective_estimator_params: cfg_estimator must resolve to a dict; "
            f"got {type(composed).__name__}"
        )
    estimator_name = composed.get("name")
    stripped = {
        k: v for k, v in composed.items() if k not in _ESTIMATOR_METADATA_FIELDS
    }
    if overrides:


        for k, v in overrides.items():
            stripped[k] = v
    if stripped.get("auxiliary_model_ref") == "sampled_model":
        if not sampled_model_name:
            raise ValueError(
                "derive_effective_estimator_params: auxiliary_model_ref=sampled_model "
                "requires sampled_model_name for pre-construction materialization"
            )
        stripped["auxiliary_model_ref"] = sampled_model_name
    return (
        estimator_name if isinstance(estimator_name, str) and estimator_name else None,
        stripped,
    )


def _derive_persisted_params(
    estimator_name: str | None,
    constructor_kwargs: Mapping[str, Any],
) -> dict[str, Any]:

    persisted = dict(constructor_kwargs)
    if estimator_name is None:
        return persisted
    estimator = _get_estimator(estimator_name, **persisted)
    effective_params = getattr(estimator, "_effective_params", None)
    if callable(effective_params):
        return dict(effective_params())
    return persisted


def derive_estimator_param_views(
    cfg_estimator: DictConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    sampled_model_name: str | None = None,
) -> EstimatorParamViews:

    estimator_name, constructor_kwargs = _derive_constructor_kwargs(
        cfg_estimator,
        overrides=overrides,
        sampled_model_name=sampled_model_name,
    )
    return EstimatorParamViews(
        constructor_kwargs=constructor_kwargs,
        persisted_params=_derive_persisted_params(estimator_name, constructor_kwargs),
    )


def derive_effective_estimator_params(
    cfg_estimator: DictConfig | dict[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    sampled_model_name: str | None = None,
) -> dict[str, Any]:

    return derive_estimator_param_views(
        cfg_estimator,
        overrides=overrides,
        sampled_model_name=sampled_model_name,
    ).persisted_params
