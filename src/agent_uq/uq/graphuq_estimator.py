from __future__ import annotations

import sys
from typing import Any, Sequence

from agent_uq.records import CandidateRecord
from agent_uq.release.protocol import GUTQ_REVISION
from agent_uq.uq.base import (
    UNSCORABLE_KIND_NO_VALID_CANDIDATES,
    TurnContext,
    UQResult,
    with_unscorable_kind,
)
from agent_uq.uq.graph_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
    GraphArtifactEntry,
    GraphArtifactError,
    derive_build_key,
)
from agent_uq.uq.graphuq_graph_builder import (
    _build_graph,
)
from agent_uq.uq.graphuq_response_adapter import (
    _accepted_span_candidate_to_graphuq_response,
)
from agent_uq.uq.graphuq_runtime import (
    _get_tokenizer as _runtime_get_tokenizer,
    resolve_graphuq_nli_device,
)
from agent_uq.uq.pairwise_nli_artifact import derive_nli_execution_mode
from agent_uq.uq.raw_turn_adapter import (
    recover_accepted_span_candidate,
    with_adapter_contract,
)
from agent_uq.vendor import graphuq as graphuq_bootstrap


def _facade_override(name: str, original: Any) -> Any | None:
    facade = sys.modules.get("agent_uq.uq.graphuq_adapter")
    override = getattr(facade, name, None)
    if callable(override) and override is not original:
        return override
    return None


def _facade_get_tokenizer(name: str) -> Any:
    override = _facade_override("_get_tokenizer", _runtime_get_tokenizer)
    if override is not None:
        return override(name)
    return _runtime_get_tokenizer(name)


def _facade_response_from_candidate(candidate: Any, *, tokenizer: Any) -> dict[str, Any]:
    override = _facade_override(
        "_accepted_span_candidate_to_graphuq_response",
        _accepted_span_candidate_to_graphuq_response,
    )
    if override is not None:
        return override(candidate, tokenizer=tokenizer)
    return _accepted_span_candidate_to_graphuq_response(candidate, tokenizer=tokenizer)


def _facade_build_graph(responses: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, list[int]]:
    override = _facade_override("_build_graph", _build_graph)
    if override is not None:
        return override(responses, **kwargs)
    return _build_graph(responses, **kwargs)


def _facade_resolve_graphuq_nli_device(nli_device: str | None) -> str:
    override = _facade_override(
        "resolve_graphuq_nli_device",
        resolve_graphuq_nli_device,
    )
    if override is not None:
        return override(nli_device)
    return resolve_graphuq_nli_device(nli_device)


class GraphUQEstimator:
    name = "gut_up"

    def __init__(
        self,
        *,
        graph_type: str = "gut_up",
        merge_way: str = "aov",
        sim_way: str = "2entail",
        weight: str = "prob",
        token_level: str = "avg neg logprob",
        step_aggregation: str = "bottom-3%",
        beta: float = 0.0,
        window_size: int = 6,
        aggregation: str = "sum",
        alpha: float = 0.2,
        nli_batch_size: int = 128,
        nli_device: str | None = None,
        action_similarity_model_name: str = DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
        action_similarity_threshold: float = 0.7,
        action_similarity_batch_size: int = 6,
        action_similarity_device: str | None = None,
    ) -> None:
        if graph_type != "gut_up":
            raise ValueError("the public release supports graph_type='gut_up'")
        self.graph_type = graph_type
        self.merge_way = merge_way
        self.sim_way = sim_way
        self.weight = weight
        self.token_level = token_level
        self.step_aggregation = step_aggregation
        self.beta = beta
        self.window_size = window_size
        self.aggregation = aggregation
        self.alpha = alpha
        self.nli_batch_size = int(nli_batch_size)
        self.nli_device = nli_device
        self.action_similarity_model_name = action_similarity_model_name
        self.action_similarity_threshold = float(action_similarity_threshold)
        self.action_similarity_batch_size = int(action_similarity_batch_size)
        self.action_similarity_device = action_similarity_device

        graphuq_bootstrap.ensure()
        from src.estimators.gut_q_up import GUTQUP

        common_kwargs = dict(
            merge_way=merge_way,
            sim_way=sim_way,
            weight=weight,
            token_level=token_level,
            step_aggregation=step_aggregation,
            beta=beta,
            window_size=window_size,
        )

        self._inner = GUTQUP(aggregation=aggregation, alpha=alpha, **common_kwargs)

    def _effective_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "gutq_revision": GUTQ_REVISION,
            "graph_type": self.graph_type,
            "merge_way": self.merge_way,
            "sim_way": self.sim_way,
            "weight": self.weight,
            "token_level": self.token_level,
            "step_aggregation": self.step_aggregation,
            "beta": float(self.beta),
            "window_size": int(self.window_size),
            "aggregation": self.aggregation,
            "action_similarity_model_name": self.action_similarity_model_name,
            "action_similarity_threshold": float(self.action_similarity_threshold),
            "nli_execution_mode": derive_nli_execution_mode(
                _facade_resolve_graphuq_nli_device(self.nli_device)
            ),
        }
        if self.graph_type == "gut_up":
            params["alpha"] = float(self.alpha)
        return with_adapter_contract(
            params,
            field_name="graphuq_adapter_contract",
        )

    def score_graph_artifact_entry(
        self,
        *,
        record_id: str,
        turn_idx: int,
        entry: GraphArtifactEntry,
    ) -> UQResult:
        return self.score_graph_artifact_entries(
            entries=[(record_id, turn_idx, entry)],
        )[(record_id, int(turn_idx))]

    def score_graph_artifact_entries(
        self,
        *,
        entries: Sequence[tuple[str, int, GraphArtifactEntry]],
    ) -> dict[tuple[str, int], UQResult]:


        g_data = []
        by_id: dict[str, tuple[str, int, GraphArtifactEntry]] = {}
        for record_id, turn_idx, entry in entries:
            item_id = f"{record_id}_turn{int(turn_idx)}"
            by_id[item_id] = (record_id, int(turn_idx), entry)
            g_data.append(
                {
                    "id": item_id,
                    "G": entry.graph,
                    "answers_id": list(entry.answers_id),
                }
            )
        try:
            u_dict = self._inner(g_data)
        except Exception as exc:
            raise RuntimeError(
                f"GraphUQEstimator: inner estimator {self.graph_type} failed for "
                f"{len(g_data)} graph artifact entries: {exc}"
            ) from exc

        results: dict[tuple[str, int], UQResult] = {}
        for item_id, (record_id, turn_idx, entry) in by_id.items():
            scalar = float(u_dict[item_id])
            results[(record_id, turn_idx)] = self._graph_artifact_result(
                scalar=scalar,
                entry=entry,
            )
        return results

    def _graph_artifact_result(
        self,
        *,
        scalar: float,
        entry: GraphArtifactEntry,
    ) -> UQResult:
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                "graph_type": self.graph_type,
                "aggregation": self.aggregation,
                "n_valid_candidates": len(entry.valid_candidate_indices),
                "excluded_indices": [],
                "adapter_invalid_indices": [],
                "adapter_invalid_reasons": {},
                "valid_candidate_indices": list(entry.valid_candidate_indices),
                "n_graph_nodes": int(entry.n_graph_nodes),
                "n_graph_edges": int(entry.n_graph_edges),
                "n_answer_nodes": int(entry.n_answer_nodes),
                "action_similarity_model_name": self.action_similarity_model_name,
                "action_similarity_threshold": self.action_similarity_threshold,
                "graph_artifact_build_key": entry.build_key,
                "graph_artifact_cache_status": "hit",
            },
            params=self._effective_params(),
        )

    def score(
        self,
        candidates: Sequence[CandidateRecord],
        context: TurnContext,
    ) -> UQResult:


        excluded: list[int] = []
        adapter_invalid_indices: list[int] = []
        adapter_invalid_reasons: dict[int, str] = {}
        for i, c in enumerate(candidates):
            if not (c.compliant and c.parsed_action is not None):
                excluded.append(i)
                continue

        tokenizer = None
        responses: list[dict[str, Any]] = []
        valid_indices = []
        for raw_idx, cand in enumerate(candidates):
            if not (cand.compliant and cand.parsed_action is not None):
                continue
            if tokenizer is None:
                tokenizer = _facade_get_tokenizer(context.tokenizer_name)
            adapted, reason = recover_accepted_span_candidate(
                cand,
                raw_index=raw_idx,
                tokenizer=tokenizer,
            )
            if adapted is None:
                excluded.append(raw_idx)
                adapter_invalid_indices.append(raw_idx)
                adapter_invalid_reasons[raw_idx] = str(reason or "adapter_invalid")
                continue
            valid_indices.append(raw_idx)
            try:
                resp = _facade_response_from_candidate(
                    adapted,
                    tokenizer=tokenizer,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GraphUQEstimator: failed to convert candidate "
                    f"turn={context.turn_idx} idx={raw_idx} (record_id={context.record_id}): {exc}"
                ) from exc
            responses.append(resp)

        if len(responses) == 0:
            return UQResult(
                turn_uncertainty=float("nan"),
                extras=with_unscorable_kind(
                    {
                        "excluded_indices": excluded,
                        "adapter_invalid_indices": adapter_invalid_indices,
                        "adapter_invalid_reasons": adapter_invalid_reasons,
                        "n_valid_candidates": 0,
                        "reason": "no valid candidates to score",
                    },
                    kind=UNSCORABLE_KIND_NO_VALID_CANDIDATES,
                ),
                params=self._effective_params(),
            )


        session = context.graph_artifact_session
        build_key = derive_build_key(
            responses=responses,
            merge_way=self.merge_way,
            sim_way=self.sim_way,
            action_similarity_model_name=self.action_similarity_model_name,
            action_similarity_threshold=self.action_similarity_threshold,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        )

        graph_payload: dict[str, Any] | None = None
        n_graph_nodes: int
        n_graph_edges: int
        answers_id: list[int]
        cache_status: str

        if session is not None:
            try:
                cached = session.lookup(context.turn_idx, build_key)
            except GraphArtifactError:


                raise
            if cached is not None:
                graph_payload = cached.graph
                answers_id = list(cached.answers_id)
                n_graph_nodes = int(cached.n_graph_nodes)
                n_graph_edges = int(cached.n_graph_edges)
                cache_status = "hit"
            else:
                G, built_answers_id = self._build_graph_with_context(
                    responses=responses, context=context
                )
                from networkx.readwrite import json_graph

                graph_payload = json_graph.node_link_data(G)
                answers_id = list(built_answers_id)
                n_graph_nodes = G.number_of_nodes()
                n_graph_edges = G.number_of_edges()
                entry = GraphArtifactEntry(
                    build_key=build_key,
                    build_params={
                        "merge_way": self.merge_way,
                        "sim_way": self.sim_way,
                        "action_similarity_model_name": self.action_similarity_model_name,
                        "action_similarity_threshold": self.action_similarity_threshold,
                    },
                    valid_candidate_indices=valid_indices,
                    responses=responses,
                    graph=graph_payload,
                    answers_id=answers_id,
                    n_graph_nodes=n_graph_nodes,
                    n_graph_edges=n_graph_edges,
                    n_answer_nodes=len(answers_id),
                )
                try:
                    session.stage(context.turn_idx, entry)
                except GraphArtifactError:
                    raise
                cache_status = "miss"
        else:

            G, built_answers_id = self._build_graph_with_context(
                responses=responses, context=context
            )
            from networkx.readwrite import json_graph

            graph_payload = json_graph.node_link_data(G)
            answers_id = list(built_answers_id)
            n_graph_nodes = G.number_of_nodes()
            n_graph_edges = G.number_of_edges()
            cache_status = "transient"

        g_data = [
            {
                "id": f"{context.record_id}_turn{context.turn_idx}",
                "G": graph_payload,
                "answers_id": answers_id,
            }
        ]
        try:
            u_dict = self._inner(g_data)
        except Exception as exc:
            raise RuntimeError(
                f"GraphUQEstimator: inner estimator {self.graph_type} failed for "
                f"turn={context.turn_idx} (record_id={context.record_id}): {exc}"
            ) from exc

        scalar = float(next(iter(u_dict.values())))
        return UQResult(
            turn_uncertainty=scalar,
            extras={
                "graph_type": self.graph_type,
                "aggregation": self.aggregation,
                "n_valid_candidates": len(valid_indices),
                "excluded_indices": excluded,
                "adapter_invalid_indices": adapter_invalid_indices,
                "adapter_invalid_reasons": adapter_invalid_reasons,
                "n_graph_nodes": n_graph_nodes,
                "n_graph_edges": n_graph_edges,
                "n_answer_nodes": len(answers_id),
                "action_similarity_model_name": self.action_similarity_model_name,
                "action_similarity_threshold": self.action_similarity_threshold,
                "graph_artifact_build_key": build_key,
                "graph_artifact_cache_status": cache_status,
            },
            params=self._effective_params(),
        )

    def _build_graph_with_context(
        self,
        *,
        responses: list[dict[str, Any]],
        context: TurnContext,
    ) -> tuple[Any, list[int]]:

        try:
            return _facade_build_graph(
                responses,
                merge_way=self.merge_way,
                sim_way=self.sim_way,
                nli_batch_size=self.nli_batch_size,
                nli_device=self.nli_device,
                action_similarity_model_name=self.action_similarity_model_name,
                action_similarity_threshold=self.action_similarity_threshold,
                action_similarity_batch_size=self.action_similarity_batch_size,
                action_similarity_device=self.action_similarity_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"GraphUQEstimator: graph construction failed for "
                f"turn={context.turn_idx} (record_id={context.record_id}): {exc}"
            ) from exc
