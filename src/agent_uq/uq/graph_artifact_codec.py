from __future__ import annotations

from typing import Any, Mapping

from agent_uq.uq.graph_artifact_types import (
    _REQUIRED_ARTIFACT_FIELDS,
    _REQUIRED_BUILD_PARAM_FIELDS,
    _REQUIRED_TOP_LEVEL_FIELDS,
    GraphArtifactEntry,
    GraphArtifactRecord,
    GraphArtifactSchemaError,
    GraphArtifactTurn,
)


def _ensure_dict(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: expected dict at {where}, got {type(value).__name__}"
        )
    return value


def _ensure_list(value: Any, *, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: expected list at {where}, got {type(value).__name__}"
        )
    return value


def _entry_to_jsonable(entry: GraphArtifactEntry) -> dict[str, Any]:
    return {
        "build_key": entry.build_key,
        "build_params": dict(entry.build_params),
        "valid_candidate_indices": list(entry.valid_candidate_indices),
        "responses": entry.responses,
        "graph": entry.graph,
        "answers_id": list(entry.answers_id),
        "n_graph_nodes": int(entry.n_graph_nodes),
        "n_graph_edges": int(entry.n_graph_edges),
        "n_answer_nodes": int(entry.n_answer_nodes),
    }


def record_to_jsonable(record: GraphArtifactRecord) -> dict[str, Any]:


    turns_out: list[dict[str, Any]] = []
    for turn_idx in sorted(record.turns):
        turn = record.turns[turn_idx]
        artifacts_out: dict[str, dict[str, Any]] = {}
        for build_key in sorted(turn.artifacts):
            artifacts_out[build_key] = _entry_to_jsonable(turn.artifacts[build_key])
        turns_out.append({"turn_idx": int(turn_idx), "artifacts": artifacts_out})

    return {
        "record_id": record.record_id,
        "dataset_key": record.dataset_key,
        "tokenizer_name": record.tokenizer_name,
        "artifact_schema_version": int(record.artifact_schema_version),
        "turns": turns_out,
    }


def record_from_jsonable(payload: Any, *, source: str = "<memory>") -> GraphArtifactRecord:


    data = _ensure_dict(payload, where=f"{source} (top level)")
    for key in _REQUIRED_TOP_LEVEL_FIELDS:
        if key not in data:
            raise GraphArtifactSchemaError(
                f"graph-artifact schema: {source} is missing required top-level "
                f"field {key!r}"
            )

    record_id = data["record_id"]
    dataset_key = data["dataset_key"]
    tokenizer_name = data["tokenizer_name"]
    artifact_schema_version = data["artifact_schema_version"]
    if not isinstance(record_id, str) or not record_id:
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.record_id must be a non-empty string"
        )
    if not isinstance(dataset_key, str):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.dataset_key must be a string"
        )
    if not isinstance(tokenizer_name, str):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.tokenizer_name must be a string"
        )
    if not isinstance(artifact_schema_version, int):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.artifact_schema_version must be an int"
        )

    turns_payload = data["turns"]
    turns_out: dict[int, GraphArtifactTurn] = {}

    if isinstance(turns_payload, list):
        for i, raw_turn in enumerate(turns_payload):
            turn_dict = _ensure_dict(raw_turn, where=f"{source}.turns[{i}]")
            if "turn_idx" not in turn_dict:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.turns[{i}] missing 'turn_idx'"
                )
            if "artifacts" not in turn_dict:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.turns[{i}] missing 'artifacts'"
                )
            turn = _read_turn(turn_dict, source=f"{source}.turns[{i}]")
            turns_out[turn.turn_idx] = turn
    elif isinstance(turns_payload, dict):
        for k, raw_turn in turns_payload.items():
            try:
                idx = int(k)
            except (TypeError, ValueError) as exc:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.turns has non-int key {k!r}"
                ) from exc
            turn_dict = _ensure_dict(raw_turn, where=f"{source}.turns[{k}]")
            if "artifacts" not in turn_dict:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.turns[{k}] missing 'artifacts'"
                )
            turn_dict.setdefault("turn_idx", idx)
            turn = _read_turn(turn_dict, source=f"{source}.turns[{k}]")
            turns_out[turn.turn_idx] = turn
    else:
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.turns must be a list or dict, "
            f"got {type(turns_payload).__name__}"
        )

    return GraphArtifactRecord(
        record_id=record_id,
        dataset_key=dataset_key,
        tokenizer_name=tokenizer_name,
        artifact_schema_version=artifact_schema_version,
        turns=turns_out,
    )


def _read_turn(turn_dict: Mapping[str, Any], *, source: str) -> GraphArtifactTurn:
    raw_idx = turn_dict["turn_idx"]
    if not isinstance(raw_idx, int):
        raise GraphArtifactSchemaError(
            f"graph-artifact schema: {source}.turn_idx must be int, "
            f"got {type(raw_idx).__name__}"
        )
    artifacts_payload = turn_dict["artifacts"]
    artifacts_dict = _ensure_dict(artifacts_payload, where=f"{source}.artifacts")
    artifacts_out: dict[str, GraphArtifactEntry] = {}
    for build_key, raw_entry in artifacts_dict.items():
        if not isinstance(build_key, str) or not build_key:
            raise GraphArtifactSchemaError(
                f"graph-artifact schema: {source}.artifacts has non-string build_key {build_key!r}"
            )
        entry_dict = _ensure_dict(raw_entry, where=f"{source}.artifacts[{build_key}]")
        for required in _REQUIRED_ARTIFACT_FIELDS:
            if required not in entry_dict:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.artifacts[{build_key}] missing "
                    f"required field {required!r}"
                )
        if entry_dict["build_key"] != build_key:
            raise GraphArtifactSchemaError(
                f"graph-artifact schema: {source}.artifacts[{build_key}].build_key "
                f"({entry_dict['build_key']!r}) does not match its own key"
            )
        build_params = _ensure_dict(
            entry_dict["build_params"],
            where=f"{source}.artifacts[{build_key}].build_params",
        )
        for required in _REQUIRED_BUILD_PARAM_FIELDS:
            if required not in build_params:
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.artifacts[{build_key}].build_params "
                    f"missing required field {required!r}"
                )
        valid_indices = _ensure_list(
            entry_dict["valid_candidate_indices"],
            where=f"{source}.artifacts[{build_key}].valid_candidate_indices",
        )
        if not all(isinstance(i, int) for i in valid_indices):
            raise GraphArtifactSchemaError(
                f"graph-artifact schema: {source}.artifacts[{build_key}].valid_candidate_indices "
                f"must contain ints"
            )
        responses = _ensure_list(
            entry_dict["responses"],
            where=f"{source}.artifacts[{build_key}].responses",
        )
        graph_payload = _ensure_dict(
            entry_dict["graph"],
            where=f"{source}.artifacts[{build_key}].graph",
        )
        answers_id = _ensure_list(
            entry_dict["answers_id"],
            where=f"{source}.artifacts[{build_key}].answers_id",
        )
        if not all(isinstance(i, int) for i in answers_id):
            raise GraphArtifactSchemaError(
                f"graph-artifact schema: {source}.artifacts[{build_key}].answers_id "
                f"must contain ints"
            )
        for count_field in ("n_graph_nodes", "n_graph_edges", "n_answer_nodes"):
            if not isinstance(entry_dict[count_field], int):
                raise GraphArtifactSchemaError(
                    f"graph-artifact schema: {source}.artifacts[{build_key}].{count_field} "
                    f"must be int"
                )

        artifacts_out[build_key] = GraphArtifactEntry(
            build_key=build_key,
            build_params=dict(build_params),
            valid_candidate_indices=[int(i) for i in valid_indices],
            responses=list(responses),
            graph=dict(graph_payload),
            answers_id=[int(i) for i in answers_id],
            n_graph_nodes=int(entry_dict["n_graph_nodes"]),
            n_graph_edges=int(entry_dict["n_graph_edges"]),
            n_answer_nodes=int(entry_dict["n_answer_nodes"]),
        )

    return GraphArtifactTurn(turn_idx=int(raw_idx), artifacts=artifacts_out)
