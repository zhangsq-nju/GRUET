from __future__ import annotations

from typing import Any

import numpy as np

from agent_uq.records import CandidateRecord, TopKEntryRec
from agent_uq.uq.base import (
    EstimatorInfrastructureError,
    TurnContext,
    UQResult,
    register_estimator,
)
from agent_uq.uq.graphuq_adapter import _get_tokenizer
from agent_uq.uq.lm_polygraph_adapter import (
    AdaptedCandidate,
    AdaptedTurnSamples,
    AnchorResolution,
    adapt_turn_samples,
    extract_task_context_text,
    resolve_anchor,
    _concat_input,
    _compute_jaccard_matrix,
    _no_valid_candidates_result,
    _recover_whole_raw_candidate,
    _safe_exp,
    _sample_diagnostics,
    _unscorable_anchor_result,
)
from agent_uq.uq.lm_polygraph_constants import (
    DEFAULT_P_TRUE_PROMPT,
    DEFAULT_VERBALIZED_CONFIDENCE_REGEX,
    DEFAULT_VERBALIZED_PROMPT,
    _DEFAULT_CROSS_ENCODER_NAME,
    _DEFAULT_NLI_MODEL_NAME,
    _LM_POLYGRAPH_ADAPTER_CONTRACT,
    _MODEL_CONF_DIR,
    _P_TRUE_SCORING_MODE,
    _RAW_TOKEN_ARRAYS_MISALIGNED,
)
from agent_uq.uq.lm_polygraph_estimators import (
    DegMatEstimator,
    EccentricityEstimator,
    EigValLaplacianEstimator,
    PPLEstimator,
    PTrueEstimator,
    PredictiveEntropy,
    PredictiveEntropyEstimator,
    Perplexity,
    SemanticDensityEstimator,
    SemanticEntropyEstimator,
    SentenceSAREstimator,
    TokenSAREstimator,
    Verbalized2SEstimator,
    _anchor_text_without_one_token,
    _token_similarity_scores,
)
from agent_uq.uq.lm_polygraph_pairwise import (
    PairwiseNLIArtifactEntry,
    PairwiseNLIResolution,
    derive_cross_encoder_execution_mode,
    derive_nli_execution_mode,
    derive_pairwise_nli_build_key,
    resolve_cross_encoder_device,
    _anchor_nli_forward_scores,
    _cross_encoder_similarity_matrix,
    _effective_auxiliary_params_from_self,
    _effective_cross_encoder_execution_mode,
    _effective_cross_encoder_params_from_self,
    _effective_nli_execution_mode,
    _effective_nli_params_from_self,
    _effective_params_from_self,
    _normalized_laplacian,
    _pairwise_entry_to_arrays,
    _pairwise_nli_probabilities,
    _resolve_pairwise_nli_artifact,
    _sample_similarity_matrix,
    _semantic_classes_from_matrices,
    _with_adapter_contract,
)
from agent_uq.uq.lm_polygraph_runtime import (
    ResolvedAuxiliaryModelConfig,
    _AUXILIARY_CONFIG_CACHE,
    _AUXILIARY_MODEL_CACHE,
    _AUXILIARY_TOKENIZER_CACHE,
    _AuxiliaryVLLMModel,
    _CROSS_ENCODER_CACHE,
    _CrossEncoderBackend,
    _NLIBackend,
    _NLI_CACHE,
    _auxiliary_cache_key,
    _auxiliary_runtime_signature,
    _construct_auxiliary_vllm_engine,
    _freeze_auxiliary_runtime_signature,
    _is_cuda_device,
    _load_auxiliary_tokenizer,
    _parse_auxiliary_worker_gpu,
    _release_lm_polygraph_gpu_state,
    _resolve_torch_device,
    _temporary_cuda_visible_devices,
    configure_vllm_worker_multiproc_method,
    get_auxiliary_model,
    get_cross_encoder_backend,
    get_nli_backend,
    resolve_auxiliary_model_config,
    validate_tensor_parallel_size,
)


@register_estimator("deg_mat")
def _make_deg_mat(**kwargs: Any) -> DegMatEstimator:
    return DegMatEstimator(**kwargs)


@register_estimator("eccentricity")
def _make_eccentricity(**kwargs: Any) -> EccentricityEstimator:
    return EccentricityEstimator(**kwargs)


@register_estimator("eig_val_laplacian")
def _make_eig_val_laplacian(**kwargs: Any) -> EigValLaplacianEstimator:
    return EigValLaplacianEstimator(**kwargs)


@register_estimator("semantic_entropy")
def _make_semantic_entropy(**kwargs: Any) -> SemanticEntropyEstimator:
    return SemanticEntropyEstimator(**kwargs)


@register_estimator("ppl")
def _make_ppl(**kwargs: Any) -> PPLEstimator:
    return PPLEstimator(**kwargs)


@register_estimator("predictive_entropy")
def _make_predictive_entropy(**kwargs: Any) -> PredictiveEntropyEstimator:
    return PredictiveEntropyEstimator(**kwargs)


@register_estimator("sentence_sar")
def _make_sentence_sar(**kwargs: Any) -> SentenceSAREstimator:
    return SentenceSAREstimator(**kwargs)


@register_estimator("token_sar")
def _make_token_sar(**kwargs: Any) -> TokenSAREstimator:
    return TokenSAREstimator(**kwargs)


@register_estimator("semantic_density")
def _make_semantic_density(**kwargs: Any) -> SemanticDensityEstimator:
    return SemanticDensityEstimator(**kwargs)


@register_estimator("p_true")
def _make_p_true(**kwargs: Any) -> PTrueEstimator:
    return PTrueEstimator(**kwargs)


@register_estimator("verbalized_2s")
def _make_verbalized_2s(**kwargs: Any) -> Verbalized2SEstimator:
    return Verbalized2SEstimator(**kwargs)
