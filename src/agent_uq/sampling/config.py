from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from agent_uq.vllm_runtime import validate_tensor_parallel_size

_DEFAULT_RETRY_BUDGET_MULTIPLIER_PER_TURN = 3
_REJECTION_DERIVATION_ONLY_FIELDS = frozenset({"retry_budget_multiplier_per_turn"})


def _resolve_retry_budget_multiplier_per_turn(rejection: DictConfig | dict[str, Any]) -> int:

    raw = rejection.get(
        "retry_budget_multiplier_per_turn",
        _DEFAULT_RETRY_BUDGET_MULTIPLIER_PER_TURN,
    )
    if raw is None:
        raw = _DEFAULT_RETRY_BUDGET_MULTIPLIER_PER_TURN
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            "sampling.rejection.retry_budget_multiplier_per_turn must be a "
            f"positive integer; got {type(raw).__name__} {raw!r}"
        )
    if raw <= 0:
        raise ValueError(
            "sampling.rejection.retry_budget_multiplier_per_turn must be > 0; "
            f"got {raw}"
        )
    return int(raw)


def apply_rejection_defaults_and_validate(cfg: DictConfig) -> None:

    sampling = cfg.sampling
    rejection = sampling.get("rejection")
    if rejection is None:
        raise ValueError(
            "sampling.rejection block is missing from the composed config - "
            "expected `sampling.rejection.{enabled,max_retry_candidates_per_turn,"
            "max_retry_candidates_per_episode}` to be present"
        )

    num_samples = int(sampling.num_samples)
    max_turns = int(sampling.max_turns)
    multiplier = _resolve_retry_budget_multiplier_per_turn(rejection)

    if rejection.get("max_retry_candidates_per_turn", None) is None:
        rejection.max_retry_candidates_per_turn = multiplier * num_samples
    if rejection.get("max_retry_candidates_per_episode", None) is None:
        rejection.max_retry_candidates_per_episode = (
            int(rejection.max_retry_candidates_per_turn) * max_turns
        )

    per_turn = int(rejection.max_retry_candidates_per_turn)
    per_episode = int(rejection.max_retry_candidates_per_episode)

    if per_turn < num_samples:
        raise ValueError(
            "sampling.rejection.max_retry_candidates_per_turn >= sampling.num_samples "
            f"constraint violated: got per_turn={per_turn}, num_samples={num_samples}"
        )
    if per_episode < per_turn:
        raise ValueError(
            "sampling.rejection.max_retry_candidates_per_episode >= "
            "sampling.rejection.max_retry_candidates_per_turn constraint violated: "
            f"got per_episode={per_episode}, per_turn={per_turn}"
        )


def extract_effective_sampling_manifest_views(
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:

    raw = OmegaConf.to_container(cfg.sampling, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"cfg.sampling must resolve to a dict; got {type(raw).__name__}"
        )
    if "rejection" not in raw:
        raise ValueError(
            "extract_effective_sampling_config: cfg.sampling.rejection is missing "
            "after apply_rejection_defaults_and_validate; this is a bug in cli.run "
            "wiring"
        )
    rejection = raw.get("rejection")
    if isinstance(rejection, dict):
        raw["rejection"] = {
            k: v for k, v in rejection.items()
            if k not in _REJECTION_DERIVATION_ONLY_FIELDS
        }
    raw["model_path"] = str(cfg.model.name)
    raw["env_backend"] = str(cfg.env.get("backend", "docker"))
    runtime = {
        "sampling": {
            "tensor_parallel_size": validate_tensor_parallel_size(
                int(cfg.model.get("tensor_parallel_size", 1))
            ),
        }
    }
    return raw, runtime


def extract_effective_sampling_config(cfg: DictConfig) -> dict[str, Any]:

    sampling, _ = extract_effective_sampling_manifest_views(cfg)
    return sampling


__all__ = [
    "_DEFAULT_RETRY_BUDGET_MULTIPLIER_PER_TURN",
    "_REJECTION_DERIVATION_ONLY_FIELDS",
    "_resolve_retry_budget_multiplier_per_turn",
    "apply_rejection_defaults_and_validate",
    "extract_effective_sampling_config",
    "extract_effective_sampling_manifest_views",
]
