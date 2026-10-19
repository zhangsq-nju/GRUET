from __future__ import annotations

import sys
from typing import Any

from agent_uq.uq.graph_artifact import (
    DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
    DEFAULT_ACTION_SIMILARITY_THRESHOLD,
)
from agent_uq.uq.graphuq_runtime import (
    DEFAULT_ACTION_SIMILARITY_BATCH_SIZE,
    _ActionSimilarityScorerSingleton as _RuntimeActionSimilarityScorerSingleton,
    _GraphConstructorSingleton as _RuntimeGraphConstructorSingleton,
)


def _graph_constructor_singleton() -> Any:
    facade = sys.modules.get("agent_uq.uq.graphuq_adapter")
    override = getattr(facade, "_GraphConstructorSingleton", None)
    if override is not None and override is not _RuntimeGraphConstructorSingleton:
        return override
    return _RuntimeGraphConstructorSingleton


def _action_similarity_scorer_singleton() -> Any:
    facade = sys.modules.get("agent_uq.uq.graphuq_adapter")
    override = getattr(facade, "_ActionSimilarityScorerSingleton", None)
    if override is not None and override is not _RuntimeActionSimilarityScorerSingleton:
        return override
    return _RuntimeActionSimilarityScorerSingleton


def _predict_text_equivalence(
    scorer: Any,
    existing_texts: list[str],
    new_texts: list[str],
    *,
    sim_way: str,
    nli_batch_size: int,
) -> list[list[bool]]:

    if not existing_texts:
        return [[] for _ in new_texts]
    if not new_texts:
        return []
    flags = scorer.predict(
        existing_texts * len(new_texts),
        sum([[text] * len(existing_texts) for text in new_texts], []),
        judge=sim_way,
        batch_size=nli_batch_size,
    )
    width = len(existing_texts)
    return [
        [bool(x) for x in flags[i * width : (i + 1) * width]]
        for i in range(len(new_texts))
    ]


def _predict_action_similarity(
    scorer: Any,
    existing_texts: list[str],
    new_texts: list[str],
    *,
    threshold: float,
    batch_size: int,
) -> list[list[bool]]:

    return scorer.predict(
        existing_texts,
        new_texts,
        threshold=threshold,
        batch_size=batch_size,
    )


def _build_graph(
    responses: list[dict[str, Any]],
    *,
    merge_way: str,
    sim_way: str,
    nli_batch_size: int,
    nli_device: str | None = None,
    action_similarity_model_name: str = DEFAULT_ACTION_SIMILARITY_MODEL_NAME,
    action_similarity_threshold: float = DEFAULT_ACTION_SIMILARITY_THRESHOLD,
    action_similarity_batch_size: int = DEFAULT_ACTION_SIMILARITY_BATCH_SIZE,
    action_similarity_device: str | None = None,
) -> tuple[Any, list[int]]:

    import networkx as nx

    constructor = _graph_constructor_singleton().get(nli_device=nli_device)
    scorer = constructor.scorer

    G = nx.DiGraph()
    G.add_node(0, num_nodes=1)
    answer_node_ids: list[int] = []
    node_id = 1

    for chain_idx, response in enumerate(responses):
        cot = response["step-wise info"]
        nodeset = set(G.nodes) - set(answer_node_ids) - {0}
        prev_id = 0

        nodelist = list(nodeset)
        text_list = [G.nodes[u]["text"] for u in nodelist]
        newtext_list = [step["text"] for step in cot]
        flags = _predict_text_equivalence(
            scorer,
            text_list,
            newtext_list,
            sim_way=sim_way,
            nli_batch_size=nli_batch_size,
        )
        step_u_list = [
            [nodelist[j] for j in range(len(nodelist)) if flags[i][j]]
            for i in range(len(cot))
        ]

        if merge_way == "aov":
            for h, step in enumerate(cot):
                u = next((u for u in step_u_list[h] if u in nodeset), None)
                if u is not None:
                    nodeset -= nx.ancestors(G, u)
                    nodeset -= {u}
                    if prev_id != u:
                        G.add_edge(prev_id, u, chain=chain_idx)
                        G.nodes[u]["prob_info_all"].append(step["prob info"])
                        G.nodes[u]["num_nodes"] += 1
                    prev_id = u
                else:
                    G.add_node(
                        node_id,
                        text=step["text"],
                        num_nodes=1,
                        prob_info_all=[step["prob info"]],
                    )
                    G.add_edge(prev_id, node_id, chain=chain_idx)
                    prev_id = node_id
                node_id += 1
        elif merge_way == "allroads":
            for h, step in enumerate(cot):
                G.add_node(
                    node_id,
                    text=step["text"],
                    num_nodes=1,
                    prob_info_all=[step["prob info"]],
                )
                G.add_edge(prev_id, node_id, chain=chain_idx)
                prev_id = node_id
                u_list = step_u_list[h]
                for u in u_list:
                    if u not in G.nodes:
                        continue
                    nx.contracted_nodes(G, u, prev_id, self_loops=False, copy=False)
                    G.nodes[u]["prob_info_all"] += G.nodes[u]["contraction"][prev_id]["prob_info_all"]
                    G.nodes[u]["num_nodes"] += G.nodes[u]["contraction"][prev_id]["num_nodes"]
                    G.nodes[u].pop("contraction", None)
                    prev_id = u
                if len(u_list) == 0:
                    node_id += 1
            for _, _, data in G.edges(data=True):
                data.pop("contraction", None)
        else:
            raise ValueError(f"Unknown merge way: {merge_way!r}")

        answer = response["answer info"]
        existing_answers = list(answer_node_ids)
        answer_texts = [G.nodes[u]["text"] for u in existing_answers]
        if answer_texts:
            action_scorer = _action_similarity_scorer_singleton().get(
                model_name=action_similarity_model_name,
                action_similarity_device=action_similarity_device,
            )
            answer_flags = _predict_action_similarity(
                action_scorer,
                answer_texts,
                [answer["text"]],
                threshold=action_similarity_threshold,
                batch_size=action_similarity_batch_size,
            )
        else:
            answer_flags = [[]]
        matched_answer = next(
            (
                existing_answers[j]
                for j in range(len(existing_answers))
                if answer_flags and answer_flags[0][j]
            ),
            None,
        )
        if matched_answer is not None:
            G.add_edge(prev_id, matched_answer, chain=chain_idx)
            G.nodes[matched_answer]["prob_info_all"].append(answer["prob info"])
            G.nodes[matched_answer]["num_nodes"] += 1
        else:
            G.add_node(
                node_id,
                text=answer["text"],
                num_nodes=1,
                prob_info_all=[answer["prob info"]],
            )
            answer_node_ids.append(node_id)
            G.add_edge(prev_id, node_id, chain=chain_idx)
        node_id += 1

    return G, answer_node_ids
