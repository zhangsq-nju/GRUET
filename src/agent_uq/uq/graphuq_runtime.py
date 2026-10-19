from __future__ import annotations

import contextlib
from typing import Any

from agent_uq.uq.hf_fallback import load_with_hf_home_fallback
from agent_uq.vendor import graphuq as graphuq_bootstrap

DEFAULT_ACTION_SIMILARITY_BATCH_SIZE = 6


@contextlib.contextmanager
def _disable_transformers_auto_device_map():

    from transformers.modeling_utils import PreTrainedModel

    original_attr = PreTrainedModel.__dict__["from_pretrained"]
    original_func = original_attr.__func__

    def patched_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        if kwargs.get("device_map") == "auto":
            kwargs = dict(kwargs)
            kwargs.pop("device_map", None)
        return original_func(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(patched_from_pretrained)
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = original_attr


@contextlib.contextmanager
def _enable_hf_home_fallback_for_transformers_loaders():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    original_tokenizer = AutoTokenizer.from_pretrained
    original_model = AutoModelForSequenceClassification.from_pretrained

    def tokenizer_from_pretrained(model_ref, *args, **kwargs):
        return load_with_hf_home_fallback(
            original_tokenizer,
            model_ref,
            *args,
            **kwargs,
        )

    def model_from_pretrained(model_ref, *args, **kwargs):
        return load_with_hf_home_fallback(
            original_model,
            model_ref,
            *args,
            **kwargs,
        )

    AutoTokenizer.from_pretrained = staticmethod(tokenizer_from_pretrained)
    AutoModelForSequenceClassification.from_pretrained = staticmethod(model_from_pretrained)
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = original_tokenizer
        AutoModelForSequenceClassification.from_pretrained = original_model


_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(name: str) -> Any:
    if name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[name] = load_with_hf_home_fallback(
            AutoTokenizer.from_pretrained,
            name,
            trust_remote_code=True,
        )
    return _TOKENIZER_CACHE[name]


class _GraphConstructorSingleton:


    _instance: Any = None
    _device: str | None = None

    @classmethod
    def get(cls, *, nli_device: str | None = None) -> Any:
        if cls._instance is not None and (nli_device is None or nli_device == cls._device):
            return cls._instance

        graphuq_bootstrap.ensure()

        from src.stat_calculators.graph_construct import GraphConstructor
        from src.utils.nli import NLIScorer


        device = resolve_graphuq_nli_device(nli_device)


        with _enable_hf_home_fallback_for_transformers_loaders():
            gc = GraphConstructor()
            try:
                gc.scorer = NLIScorer(device=device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device == "cuda":

                    import torch

                    torch.cuda.empty_cache()
                    gc.scorer = NLIScorer(device="cpu")
                    device = "cpu"
                else:
                    raise

        cls._instance = gc
        cls._device = device
        return cls._instance


class _ActionSimilarityScorer:


    def __init__(self, *, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_graphuq_action_similarity_device(device)
        with _disable_transformers_auto_device_map():
            self.model = load_with_hf_home_fallback(
                SentenceTransformer,
                model_name,
                cache_kwarg="cache_folder",
                trust_remote_code=True,
                device=self.device,
            )

    def predict(
        self,
        existing_texts: list[str],
        new_texts: list[str],
        *,
        threshold: float,
        batch_size: int,
    ) -> list[list[bool]]:
        rows = self.score(
            existing_texts,
            new_texts,
            batch_size=batch_size,
        )
        return [[float(score) >= float(threshold) for score in row] for row in rows]

    def score(
        self,
        existing_texts: list[str],
        new_texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        if not existing_texts:
            return [[] for _ in new_texts]
        if not new_texts:
            return []

        existing_embeddings = self.model.encode(
            existing_texts,
            batch_size=int(batch_size),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        new_embeddings = self.model.encode(
            new_texts,
            batch_size=int(batch_size),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = new_embeddings @ existing_embeddings.T
        if hasattr(scores, "detach"):
            rows = scores.detach().cpu().tolist()
        else:
            rows = scores.tolist()
        return [[float(score) for score in row] for row in rows]


class _ActionSimilarityScorerSingleton:
    _instance: Any = None
    _model_name: str | None = None
    _device: str | None = None

    @classmethod
    def get(
        cls,
        *,
        model_name: str,
        action_similarity_device: str | None = None,
    ) -> Any:
        resolved_device = resolve_graphuq_action_similarity_device(action_similarity_device)
        if (
            cls._instance is not None
            and cls._model_name == model_name
            and cls._device == resolved_device
        ):
            return cls._instance

        scorer = _ActionSimilarityScorer(
            model_name=model_name,
            device=resolved_device,
        )
        cls._instance = scorer
        cls._model_name = model_name
        cls._device = resolved_device
        return cls._instance


def _release_graphuq_gpu_state() -> None:

    _GraphConstructorSingleton._instance = None
    _GraphConstructorSingleton._device = None
    _ActionSimilarityScorerSingleton._instance = None
    _ActionSimilarityScorerSingleton._model_name = None
    _ActionSimilarityScorerSingleton._device = None


def _resolve_nli_device() -> str:

    import os

    override = os.environ.get("AGENT_UQ_NLI_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 1024**3:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_graphuq_nli_device(nli_device: str | None = None) -> str:

    return nli_device or _resolve_nli_device()


def _resolve_action_similarity_device() -> str:

    import os

    override = os.environ.get("AGENT_UQ_ACTION_SIMILARITY_DEVICE")
    if override:
        return override
    return _resolve_nli_device()


def resolve_graphuq_action_similarity_device(
    action_similarity_device: str | None = None,
) -> str:
    return action_similarity_device or _resolve_action_similarity_device()
