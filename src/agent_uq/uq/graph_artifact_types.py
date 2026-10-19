from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ARTIFACT_SCHEMA_VERSION = 2

DEFAULT_ACTION_SIMILARITY_MODEL_NAME = "Salesforce/SFR-Embedding-Code-2B_R"
DEFAULT_ACTION_SIMILARITY_THRESHOLD = 0.7

_REQUIRED_TOP_LEVEL_FIELDS = (
    "record_id",
    "dataset_key",
    "tokenizer_name",
    "artifact_schema_version",
    "turns",
)
_REQUIRED_ARTIFACT_FIELDS = (
    "build_key",
    "build_params",
    "valid_candidate_indices",
    "responses",
    "graph",
    "answers_id",
    "n_graph_nodes",
    "n_graph_edges",
    "n_answer_nodes",
)
_REQUIRED_BUILD_PARAM_FIELDS = ("merge_way", "sim_way")

GRAPH_CACHE_PROFILE_METADATA_FILENAME = ".graph_cache_profile.json"
GRAPH_CACHE_PROFILE_KEYS = (
    "raw_profile_id",
    "artifact_schema_version",
    "nli_execution_mode",
    "merge_way",
    "sim_way",
    "action_similarity_model_name",
    "action_similarity_threshold",
)
GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE = "read_write"
GRAPH_ARTIFACT_CACHE_MODE_READ_ONLY = "read_only"
GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT = "reject_unprofiled"
GRAPH_ARTIFACT_CACHE_ADOPTION_ADOPT = "adopt_unprofiled"


class GraphArtifactError(Exception):
    pass


class GraphArtifactDecodeError(GraphArtifactError):
    pass


class GraphArtifactSchemaError(GraphArtifactError):
    pass


class GraphArtifactPathAmbiguityError(GraphArtifactError):
    pass


class GraphArtifactWriteError(GraphArtifactError):
    pass


class GraphArtifactProfileError(GraphArtifactError):
    pass


class GraphArtifactCacheMissError(GraphArtifactError):
    pass


@dataclass
class GraphArtifactEntry:


    build_key: str
    build_params: dict[str, Any]
    valid_candidate_indices: list[int]
    responses: list[dict[str, Any]]
    graph: dict[str, Any]
    answers_id: list[int]
    n_graph_nodes: int
    n_graph_edges: int
    n_answer_nodes: int


@dataclass
class GraphArtifactTurn:


    turn_idx: int
    artifacts: dict[str, GraphArtifactEntry] = field(default_factory=dict)


@dataclass
class GraphArtifactRecord:


    record_id: str
    dataset_key: str
    tokenizer_name: str
    artifact_schema_version: int
    turns: dict[int, GraphArtifactTurn] = field(default_factory=dict)
