from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.linalg import eigh

from agent_uq.records import CandidateRecord
from agent_uq.uq.base import TurnContext, UQResult
from agent_uq.uq.lm_polygraph_adapter import (
    adapt_turn_samples,
    _no_valid_candidates_result,
    _safe_exp,
    _sample_diagnostics,
)
from agent_uq.uq.lm_polygraph_constants import _DEFAULT_NLI_MODEL_NAME
from agent_uq.uq.lm_polygraph_pairwise import (
    _effective_nli_params_from_self,
    _normalized_laplacian,
    _resolve_pairwise_nli_artifact,
    _sample_similarity_matrix,
    _semantic_classes_from_matrices,
    _with_adapter_contract,
)


class DegMatEstimator:
    name = "deg_mat"

    def __init__(
        self,
        *,
        similarity_score: str = "NLI_score",
        affinity: str = "entail",
        nli_model_name: str = _DEFAULT_NLI_MODEL_NAME,
        nli_batch_size: int = 32,
        nli_device: str | None = None,
    ) -> None:
        self.similarity_score = similarity_score
        self.affinity = affinity
        self.nli_model_name = nli_model_name
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_nli_params_from_self(
                self,
                (
                    "similarity_score",
                    "affinity",
                    "nli_model_name",
                    "nli_device",
                ),
                include_execution_mode=self.similarity_score != "Jaccard_score",
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        matrix, pairwise_extras = _sample_similarity_matrix(
            samples,
            context,
            similarity_score=self.similarity_score,
            affinity=self.affinity,
            nli_model_name=self.nli_model_name,
            nli_batch_size=self.nli_batch_size,
            nli_device=self.nli_device,
        )
        degree = np.diag(np.sum(matrix, axis=1))
        scalar = float(np.trace(len(samples.sample_texts) - degree) / (len(samples.sample_texts) ** 2))
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                **_sample_diagnostics(samples),
                **pairwise_extras,
            },
            params=params,
        )


class EccentricityEstimator:
    name = "eccentricity"

    def __init__(
        self,
        *,
        similarity_score: str = "NLI_score",
        affinity: str = "entail",
        thres: float = 0.9,
        nli_model_name: str = _DEFAULT_NLI_MODEL_NAME,
        nli_batch_size: int = 32,
        nli_device: str | None = None,
    ) -> None:
        self.similarity_score = similarity_score
        self.affinity = affinity
        self.thres = float(thres)
        self.nli_model_name = nli_model_name
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_nli_params_from_self(
                self,
                (
                    "similarity_score",
                    "affinity",
                    "thres",
                    "nli_model_name",
                    "nli_device",
                ),
                include_execution_mode=self.similarity_score != "Jaccard_score",
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        matrix, pairwise_extras = _sample_similarity_matrix(
            samples,
            context,
            similarity_score=self.similarity_score,
            affinity=self.affinity,
            nli_model_name=self.nli_model_name,
            nli_batch_size=self.nli_batch_size,
            nli_device=self.nli_device,
        )
        laplacian = _normalized_laplacian(matrix)
        eigenvalues, eigenvectors = eigh(laplacian)
        keep_mask = eigenvalues < self.thres
        if not np.any(keep_mask):
            keep_mask = eigenvalues == np.min(eigenvalues)
        smallest = eigenvectors[:, keep_mask].T
        c_ecc = (-1.0) * np.asarray(
            [np.linalg.norm(vec - vec.mean(0), 2) for vec in smallest],
            dtype=float,
        )
        scalar = float(np.linalg.norm(c_ecc, 2))
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                **_sample_diagnostics(samples),
                **pairwise_extras,
            },
            params=params,
        )


class EigValLaplacianEstimator:
    name = "eig_val_laplacian"

    def __init__(
        self,
        *,
        similarity_score: str = "NLI_score",
        affinity: str = "entail",
        nli_model_name: str = _DEFAULT_NLI_MODEL_NAME,
        nli_batch_size: int = 32,
        nli_device: str | None = None,
    ) -> None:
        self.similarity_score = similarity_score
        self.affinity = affinity
        self.nli_model_name = nli_model_name
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_nli_params_from_self(
                self,
                (
                    "similarity_score",
                    "affinity",
                    "nli_model_name",
                    "nli_device",
                ),
                include_execution_mode=self.similarity_score != "Jaccard_score",
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        matrix, pairwise_extras = _sample_similarity_matrix(
            samples,
            context,
            similarity_score=self.similarity_score,
            affinity=self.affinity,
            nli_model_name=self.nli_model_name,
            nli_batch_size=self.nli_batch_size,
            nli_device=self.nli_device,
        )
        laplacian = _normalized_laplacian(matrix)
        scalar = float(sum(max(0.0, 1.0 - value) for value in eigh(laplacian, eigvals_only=True)))
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                **_sample_diagnostics(samples),
                **pairwise_extras,
            },
            params=params,
        )


class SemanticEntropyEstimator:
    name = "semantic_entropy"

    def __init__(
        self,
        *,
        class_probability_estimation: str = "sum",
        entropy_estimation: str = "mean",
        use_unique_responses: bool = False,
        nli_model_name: str = _DEFAULT_NLI_MODEL_NAME,
        nli_batch_size: int = 32,
        nli_device: str | None = None,
    ) -> None:
        self.class_probability_estimation = class_probability_estimation
        self.entropy_estimation = entropy_estimation
        self.use_unique_responses = bool(use_unique_responses)
        self.nli_model_name = nli_model_name
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device

    def _effective_params(self) -> dict[str, Any]:
        return _with_adapter_contract(
            _effective_nli_params_from_self(
                self,
                (
                    "class_probability_estimation",
                    "entropy_estimation",
                    "use_unique_responses",
                    "nli_model_name",
                    "nli_device",
                ),
                include_execution_mode=True,
            )
        )

    def score(self, candidates: Sequence[CandidateRecord], context: TurnContext) -> UQResult:
        samples = adapt_turn_samples(candidates, tokenizer_name=context.tokenizer_name)
        params = self._effective_params()
        if not samples.valid_items:
            return _no_valid_candidates_result(
                params=params,
                samples=samples,
            )
        resolution = _resolve_pairwise_nli_artifact(
            samples.sample_texts,
            valid_candidate_indices=[item.raw_index for item in samples.valid_items],
            context=context,
            model_name=self.nli_model_name,
            batch_size=self.nli_batch_size,
            device=self.nli_device,
        )
        class_to_sample, sample_to_class = _semantic_classes_from_matrices(
            entailment=resolution.entailment,
            neutral=resolution.neutral,
            contradiction=resolution.contradiction,
        )
        if self.class_probability_estimation == "sum":
            class_log_probs: list[float] = []
            for class_indices in class_to_sample:
                class_likelihoods = samples.sample_log_probs[np.asarray(class_indices, dtype=int)]
                if self.use_unique_responses:
                    class_texts = np.asarray(samples.sample_texts, dtype=object)[np.asarray(class_indices, dtype=int)]
                    unique_ids = np.unique(class_texts, return_index=True)[1]
                    class_likelihoods = class_likelihoods[unique_ids]
                class_log_probs.append(float(np.logaddexp.reduce(class_likelihoods)))
        elif self.class_probability_estimation == "frequency":
            class_log_probs = [
                float(math.log(len(class_indices) / len(samples.sample_texts)))
                for class_indices in class_to_sample
            ]
        else:
            raise ValueError(
                f"unknown class_probability_estimation {self.class_probability_estimation!r}"
            )
        if self.entropy_estimation == "mean":
            terms = [
                class_log_probs[sample_to_class[idx]]
                for idx in range(len(samples.sample_texts))
            ]
            scalar = float(-np.mean(terms))
        elif self.entropy_estimation == "direct":
            scalar = float(
                -sum(
                    class_log_probs[sample_to_class[idx]]
                    * _safe_exp(class_log_probs[sample_to_class[idx]])
                    for idx in range(len(samples.sample_texts))
                )
            )
        else:
            raise ValueError(
                f"unknown entropy_estimation {self.entropy_estimation!r}"
            )
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                **_sample_diagnostics(samples),
                "class_to_sample": class_to_sample,
                "sample_to_class": sample_to_class,
                "pairwise_nli_artifact_build_key": resolution.build_key,
                "pairwise_nli_artifact_cache_status": resolution.cache_status,
            },
            params=params,
        )
