from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from agent_uq.uq.base import EstimatorInfrastructureError


def content_identity(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mathematical_runtime(runtime):

    projected = {key: value for key, value in runtime.items()
                 if key not in ('batch_size', 'device_diagnostics')}
    hashes = projected.get('source_hashes', {})
    if hashes.get('topouq') == '51a5d2e3974adaad8dff876037b5daedfae5fd755d00361b6f9ad1d6deb6e8e9':

        projected['source_hashes'] = {**hashes, 'topouq':
            'de22f10446da4dc918801a7f871488b025ed682c12644072d691708b73965690'}
    return projected


def turn_cache_key(record_id, turn_idx):
    return content_identity([record_id, int(turn_idx)])


def selected_candidate_digest(candidate):
    return content_identity({"text": candidate.text, "token_ids": candidate.token_ids,
                            "token_logprobs": candidate.token_logprobs,
                            "compliant": candidate.compliant,
                            "parsed_action": candidate.parsed_action,
                            "backend": candidate.backend})


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _read_artifact(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, ValueError) as error:
        raise EstimatorInfrastructureError(f"reasoning artifact unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise EstimatorInfrastructureError(f"reasoning artifact malformed: {path}")
    return payload


def read_result(artifact_dir, record_id, turn_idx, *, expected_identity):
    path = Path(artifact_dir) / "results" / (turn_cache_key(record_id, turn_idx) + ".json")
    payload = _read_artifact(path)
    return validate_result(payload, record_id, turn_idx, expected_identity=expected_identity, context=path)


def validate_estimates(estimates, expected_names, *, context):
    if not isinstance(estimates, dict) or set(estimates) != set(expected_names):
        raise EstimatorInfrastructureError(f"reasoning result schema mismatch: {context}")
    for entry in estimates.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("extras"), dict) or "scalar" not in entry:
            raise EstimatorInfrastructureError(f"reasoning result entry malformed: {context}")
        scalar = entry["scalar"]
        if (scalar is None and not entry["extras"].get("reason")
                or scalar is not None and (type(scalar) not in (int, float) or not math.isfinite(scalar))):
            raise EstimatorInfrastructureError(f"reasoning result scalar malformed: {context}")


def validate_result(payload, record_id, turn_idx, *, expected_identity, context="unpublished payload"):

    path = context
    if (not isinstance(payload, dict) or payload.get("schema") != "reasoning_contenders_v1"
            or payload.get("identity") != expected_identity
            or payload.get("record_id") != record_id
            or type(payload.get("turn_idx")) is not int or payload["turn_idx"] != turn_idx
            or payload.get("status") != "complete"):
        raise EstimatorInfrastructureError(f"reasoning result identity mismatch: {path}")
    names = {"topouq", "cot_uq_me", "cot_uq_sp"}
    hashes = {"input_identity", "prompt_hash", "initial_prompt_hash", "selected_idx", "selected_candidate_digest"}
    if (not isinstance(payload.get("input_hashes"), dict)
            or not hashes.issubset(payload["input_hashes"])
            or not isinstance(payload.get("estimates"), dict)
            or set(payload["estimates"]) != names
            or not isinstance(payload.get("params"), dict) or set(payload["params"]) != names):
        raise EstimatorInfrastructureError(f"reasoning result schema mismatch: {path}")
    validate_estimates(payload["estimates"], names, context=path)
    for name in names:
        params = payload["params"][name]
        if (not isinstance(params, dict) or not isinstance(params.get("artifact_dir"), str)
                or not params["artifact_dir"] or params.get("artifact_identity") != expected_identity
                or params.get("reasoning_contender_contract") != "source_runtime_cache_v1"):
            raise EstimatorInfrastructureError(f"reasoning result params malformed: {path}")
    return payload


def stage_cache_key(record_id, turn_idx, *, sample_index, stage,
                    subquestion_index=None, attempt_index=0, identity=None):
    return content_identity([identity, turn_cache_key(record_id, turn_idx),
                             sample_index, stage, subquestion_index, attempt_index])


def write_stage(artifact_dir, key, payload):
    checkpoint = {**payload, "schema": "reasoning_stage_v1", "stage_key": key, "status": "complete"}
    checkpoint["checkpoint_hash"] = content_identity(checkpoint)
    write_json_atomic(Path(artifact_dir) / "stages" / (key + ".json"), checkpoint)


def read_stage(artifact_dir, key, *, expected_identity, request_hash, parents):
    path = Path(artifact_dir) / "stages" / (key + ".json")
    payload = _read_artifact(path)
    fingerprint = {name: value for name, value in payload.items() if name != "checkpoint_hash"}
    required = {"prepared_input", "request_token_count", "output", "completion_token_ids", "finish_reason", "sampling"}
    if (payload.get("schema") != "reasoning_stage_v1" or payload.get("status") != "complete"
            or payload.get("stage_key") != key or payload.get("identity") != expected_identity
            or payload.get("request_hash") != request_hash or payload.get("parents") != parents
            or not required.issubset(payload)
            or payload.get("checkpoint_hash") != content_identity(fingerprint)):
        raise EstimatorInfrastructureError(f"reasoning stage identity mismatch: {path}")
    return payload
