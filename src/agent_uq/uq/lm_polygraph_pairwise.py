from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from agent_uq.uq.base import EstimatorInfrastructureError, TurnContext
from agent_uq.uq.lm_polygraph_adapter import (
    AdaptedTurnSamples,
    _compute_jaccard_matrix,
    _concat_input,
)
from agent_uq.uq.lm_polygraph_constants import _LM_POLYGRAPH_ADAPTER_CONTRACT
from agent_uq.uq.lm_polygraph_runtime import (
    _auxiliary_runtime_signature,
    _resolve_torch_device,
    get_cross_encoder_backend,
    get_nli_backend,
    resolve_auxiliary_model_config,
)
from agent_uq.uq.pairwise_nli_artifact import (
    PairwiseNLIArtifactEntry,
    derive_build_key as derive_pairwise_nli_build_key,
    derive_nli_execution_mode,
)

@dataclass(frozen=True)
class PairwiseNLIResolution:
    entailment: np.ndarray
    neutral: np.ndarray
    contradiction: np.ndarray
    build_key: str
    cache_status: str
    nli_execution_mode: str


def _effective_nli_execution_mode(device: str | None) -> str:
    resolved_device = _resolve_torch_device(device)
    return derive_nli_execution_mode(resolved_device)


def resolve_cross_encoder_device(device: str | None) -> str:
    return _resolve_torch_device(device)


def derive_cross_encoder_execution_mode(resolved_device: str | None) -> str:
    device = str(resolved_device or "cpu").strip().lower()
    if not device or device.startswith("cpu"):
        return "cpu"
    if device.startswith("cuda"):
        return "cuda"
    return device.split(":", 1)[0]


def _effective_cross_encoder_execution_mode(device: str | None) -> str:
    resolved_device = resolve_cross_encoder_device(device)
    return derive_cross_encoder_execution_mode(resolved_device)


def _effective_cross_encoder_params_from_self(
    instance: Any,
    names: Sequence[str],
) -> dict[str, Any]:
    params = _effective_params_from_self(instance, names)
    params["cross_encoder_execution_mode"] = _effective_cross_encoder_execution_mode(
        getattr(instance, "cross_encoder_device", None)
    )
    return params


def _pairwise_entry_to_arrays(
    entry: PairwiseNLIArtifactEntry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(entry.entailment_matrix, dtype=float),
        np.asarray(entry.neutral_matrix, dtype=float),
        np.asarray(entry.contradiction_matrix, dtype=float),
    )


def _resolve_pairwise_nli_artifact(
    sample_texts: Sequence[str],
    *,
    valid_candidate_indices: Sequence[int],
    context: TurnContext,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> PairwiseNLIResolution:
    resolved_device = _resolve_torch_device(device)
    nli_execution_mode = derive_nli_execution_mode(resolved_device)
    build_key = derive_pairwise_nli_build_key(
        sample_texts=list(sample_texts),
        nli_model_name=model_name,
        nli_execution_mode=nli_execution_mode,
    )

    session = context.pairwise_nli_artifact_session
    if session is not None:
        cached = session.lookup(context.turn_idx, build_key)
        if cached is not None:
            entailment, neutral, contradiction = _pairwise_entry_to_arrays(cached)
            return PairwiseNLIResolution(
                entailment=entailment,
                neutral=neutral,
                contradiction=contradiction,
                build_key=build_key,
                cache_status="hit",
                nli_execution_mode=nli_execution_mode,
            )

    entailment, neutral, contradiction, _ = _pairwise_nli_probabilities(
        sample_texts,
        model_name=model_name,
        batch_size=batch_size,
        device=resolved_device,
    )

    if session is not None:
        session.stage(
            context.turn_idx,
            PairwiseNLIArtifactEntry(
                build_key=build_key,
                build_params={
                    "nli_model_name": model_name,
                    "nli_execution_mode": nli_execution_mode,
                },
                valid_candidate_indices=[int(idx) for idx in valid_candidate_indices],
                sample_texts=list(sample_texts),
                entailment_matrix=entailment.tolist(),
                neutral_matrix=neutral.tolist(),
                contradiction_matrix=contradiction.tolist(),
                n_valid_candidates=len(sample_texts),
            ),
        )
        cache_status = "miss"
    else:
        cache_status = "transient"

    return PairwiseNLIResolution(
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
        build_key=build_key,
        cache_status=cache_status,
        nli_execution_mode=nli_execution_mode,
    )


def _pairwise_nli_probabilities(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    backend = get_nli_backend(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    unique_texts, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    pairs = list(itertools.product(unique_texts.tolist(), repeat=2))
    lefts = [left for left, _ in pairs]
    rights = [right for _, right in pairs]
    try:
        probs = backend.predict_proba(lefts, rights).reshape(
            len(unique_texts),
            len(unique_texts),
            -1,
        )
    except EstimatorInfrastructureError:
        raise
    except Exception as exc:
        raise EstimatorInfrastructureError(
            f"NLI backend runtime failed for {model_name!r}: {exc}"
        ) from exc
    contra = probs[:, :, backend.contradiction_id]
    neutral = probs[:, :, backend.neutral_id]
    entail = probs[:, :, backend.entailment_id]
    labels = np.argmax(probs, axis=-1)
    entail_full = entail[inverse, :][:, inverse]
    neutral_full = neutral[inverse, :][:, inverse]
    contra_full = contra[inverse, :][:, inverse]
    labels_full = labels[inverse, :][:, inverse]
    np.fill_diagonal(entail_full, 1.0)
    np.fill_diagonal(neutral_full, 0.0)
    np.fill_diagonal(contra_full, 0.0)
    np.fill_diagonal(labels_full, backend.entailment_id)
    return entail_full, neutral_full, contra_full, labels_full


def _semantic_classes_from_matrices(
    *,
    entailment: np.ndarray,
    neutral: np.ndarray,
    contradiction: np.ndarray,
) -> tuple[list[list[int]], dict[int, int]]:
    labels = np.argmax(
        np.stack([entailment, neutral, contradiction], axis=-1),
        axis=-1,
    )
    is_entail = labels == 0
    n_samples = entailment.shape[0]
    class_to_sample: list[list[int]] = []
    sample_to_class: dict[int, int] = {}
    for sample_idx in range(n_samples):
        assigned = False
        for class_idx, sample_indices in enumerate(class_to_sample):
            representative = sample_indices[0]
            if is_entail[representative, sample_idx] and is_entail[sample_idx, representative]:
                sample_indices.append(sample_idx)
                sample_to_class[sample_idx] = class_idx
                assigned = True
                break
        if not assigned:
            sample_to_class[sample_idx] = len(class_to_sample)
            class_to_sample.append([sample_idx])
    return class_to_sample, sample_to_class


def _sample_similarity_matrix(
    samples: AdaptedTurnSamples,
    context: TurnContext,
    *,
    similarity_score: str,
    affinity: str,
    nli_model_name: str,
    nli_batch_size: int,
    nli_device: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if similarity_score == "Jaccard_score":
        return _compute_jaccard_matrix(samples.sample_texts), {}
    resolution = _resolve_pairwise_nli_artifact(
        samples.sample_texts,
        valid_candidate_indices=[item.raw_index for item in samples.valid_items],
        context=context,
        model_name=nli_model_name,
        batch_size=nli_batch_size,
        device=nli_device,
    )
    if affinity == "entail":
        matrix = resolution.entailment
    else:
        matrix = 1.0 - resolution.contradiction
    return (
        (matrix + matrix.T) / 2.0,
        {
            "pairwise_nli_artifact_build_key": resolution.build_key,
            "pairwise_nli_artifact_cache_status": resolution.cache_status,
        },
    )


def _cross_encoder_similarity_matrix(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    backend = get_cross_encoder_backend(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    unique_texts, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    pairs = list(itertools.product(unique_texts.tolist(), repeat=2))
    try:
        unique_matrix = np.asarray(
            backend.predict(pairs),
            dtype=float,
        ).reshape(len(unique_texts), len(unique_texts))
    except EstimatorInfrastructureError:
        raise
    except Exception as exc:
        raise EstimatorInfrastructureError(
            f"cross-encoder backend runtime failed for {model_name!r}: {exc}"
        ) from exc
    matrix = unique_matrix[inverse, :][:, inverse]
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _anchor_nli_forward_scores(
    anchor_text: str,
    sample_texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    backend = get_nli_backend(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    unique_texts, inverse = np.unique(np.asarray(sample_texts, dtype=object), return_inverse=True)
    try:
        probs = backend.predict_proba(
            [anchor_text] * len(unique_texts),
            unique_texts.tolist(),
        )
    except EstimatorInfrastructureError:
        raise
    except Exception as exc:
        raise EstimatorInfrastructureError(
            f"NLI backend runtime failed for {model_name!r}: {exc}"
        ) from exc
    contra = probs[:, backend.contradiction_id][inverse]
    neutral = probs[:, backend.neutral_id][inverse]
    entail = probs[:, backend.entailment_id][inverse]
    return entail, neutral, contra


def _normalized_laplacian(similarity: np.ndarray) -> np.ndarray:
    degrees = np.sum(similarity, axis=1)
    safe = np.where(degrees > 0, degrees, 1.0)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe))
    return np.eye(similarity.shape[0]) - inv_sqrt @ similarity @ inv_sqrt


def _with_adapter_contract(params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    merged["lm_polygraph_adapter_contract"] = _LM_POLYGRAPH_ADAPTER_CONTRACT
    return merged


def _effective_params_from_self(instance: Any, names: Sequence[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, np.generic):
            value = value.item()
        params[name] = value
    return params


def _effective_auxiliary_params_from_self(
    instance: Any,
    names: Sequence[str],
) -> dict[str, Any]:
    params = _effective_params_from_self(instance, names)
    config = resolve_auxiliary_model_config(str(params["auxiliary_model_ref"]))
    params["auxiliary_backend"] = "vllm"
    params["auxiliary_runtime_signature"] = _auxiliary_runtime_signature(config)
    return params


def _effective_nli_params_from_self(
    instance: Any,
    names: Sequence[str],
    *,
    include_execution_mode: bool,
) -> dict[str, Any]:
    params = _effective_params_from_self(instance, names)
    if include_execution_mode:
        params["nli_execution_mode"] = _effective_nli_execution_mode(
            getattr(instance, "nli_device", None)
        )
    return params
