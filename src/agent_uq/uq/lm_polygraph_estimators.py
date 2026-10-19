from __future__ import annotations

from agent_uq.uq.lm_polygraph_estimators_auxiliary import (
    PTrueEstimator,
    Verbalized2SEstimator,
)
from agent_uq.uq.lm_polygraph_estimators_delegated import (
    PPLEstimator,
    PredictiveEntropy,
    PredictiveEntropyEstimator,
    Perplexity,
)
from agent_uq.uq.lm_polygraph_estimators_graph import (
    DegMatEstimator,
    EccentricityEstimator,
    EigValLaplacianEstimator,
    SemanticEntropyEstimator,
)
from agent_uq.uq.lm_polygraph_estimators_sar import (
    SemanticDensityEstimator,
    SentenceSAREstimator,
    TokenSAREstimator,
    _anchor_text_without_one_token,
    _token_similarity_scores,
)

_ESTIMATOR_CLASSES = (
    DegMatEstimator,
    EccentricityEstimator,
    EigValLaplacianEstimator,
    SemanticEntropyEstimator,
    PPLEstimator,
    PredictiveEntropyEstimator,
    SentenceSAREstimator,
    TokenSAREstimator,
    SemanticDensityEstimator,
    PTrueEstimator,
    Verbalized2SEstimator,
)

for _cls in _ESTIMATOR_CLASSES:
    _cls.__module__ = __name__

_anchor_text_without_one_token.__module__ = __name__
_token_similarity_scores.__module__ = __name__

del _cls, _ESTIMATOR_CLASSES

__all__ = [
    "DegMatEstimator",
    "EccentricityEstimator",
    "EigValLaplacianEstimator",
    "PPLEstimator",
    "PTrueEstimator",
    "PredictiveEntropy",
    "PredictiveEntropyEstimator",
    "Perplexity",
    "SemanticDensityEstimator",
    "SemanticEntropyEstimator",
    "SentenceSAREstimator",
    "TokenSAREstimator",
    "Verbalized2SEstimator",
    "_anchor_text_without_one_token",
    "_token_similarity_scores",
]
