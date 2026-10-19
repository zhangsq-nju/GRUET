from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path

from agent_uq.records import _from_jsonable_raw, build_scored_shell
from agent_uq.uq.cotuq_source import stored_thought_steps
from agent_uq.uq.raw_turn_adapter import CURRENT_TURN_ADAPTER_CONTRACT, recover_accepted_span_candidate
from agent_uq.uq.reasoning_contender_cache import content_identity, selected_candidate_digest, write_json_atomic


STRUCTURED_INPUT_CONTRACT = "original_task_selected_observation_v1"


def task_question_fields(dataset_key, initial_prompt):

    question = None
    source = None
    if dataset_key in ("nl2bash_fs1", "python_mbpp", "sql_spider"):
        if initial_prompt.endswith("\n\nThought 1:"):
            body = initial_prompt[:-len("\n\nThought 1:")]
            reset_start = body.rfind("\n\nObservation 0:")
            if reset_start > body.rfind("\n\nThought 1:"):
                body = body[:reset_start]
            marker = "\n\nQuestion: "
            start = body.rfind(marker)
            if start >= 0:
                question = body[start + len(marker):]
            elif body.startswith("Question: "):
                question = body[len("Question: "):]
            if question is not None:
                question = question.split("\n\nObservation 0:", 1)[0]
            source = "native_intercode_initial_task_segment"
    elif "swe" in dataset_key:
        task = initial_prompt.rsplit("--- END OF DEMONSTRATION ---", 1)[-1]
        marker = "We're currently solving the following issue within our repository. Here's the issue text:\nISSUE:\n"
        delimiter = "\n\n\nINSTRUCTIONS:\nNow, you're going to solve this issue on your own."
        if task.count(marker) == 1 and delimiter in task:
            question = task.split(marker, 1)[1].split(delimiter, 1)[0]
            source = "native_swe_current_issue_template"
    return {"task_question": question or "", "task_question_reason": None if question else "native_task_question_not_found",
            "task_question_source": source if question else None}


def _observation_fields(selected_idx, observation):
    if selected_idx is None:
        return {"observation_text": "", "observation_reason": "selected_idx_missing"}
    if observation is None:
        return {"observation_text": "", "observation_reason": "selected_observation_missing"}
    if not isinstance(observation, str):
        raise ValueError("stored selected observation must be a string or null")
    return {"observation_text": observation, "observation_reason": None}


def _read_selected_observations(raw_path):

    result = {}
    current = None
    observations = []
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rb") as stream:
        for line in stream:
            if line.startswith(b'      "turn_idx": '):
                current = json.loads(line.split(b": ", 1)[1].rstrip(b",\r\n"))
                observations = []
            elif current not in result and line.startswith(b"        {"):
                observations.append(None)
            elif current not in result and line.startswith(b'          "observation": '):
                if not observations:
                    raise ValueError("observation outside native candidate layout")
                observations[-1] = json.loads(line.split(b": ", 1)[1].rstrip(b",\r\n"))
            elif line.startswith(b'      "selected_idx": '):
                index = json.loads(line.split(b": ", 1)[1].rstrip(b",\r\n"))
                if current is None or current in result:
                    raise ValueError("invalid native raw turn layout")
                if index is not None and not 0 <= index < len(observations):
                    raise ValueError("stored selected_idx out of bounds or unsupported raw layout")
                result[current] = (index, None if index is None else observations[index])
    return result


def enrich_prepared_record(prepared):

    import copy
    path = Path(prepared["raw_path"])
    if _stat_fingerprint(path.stat()) != prepared["raw_stat"]:
        raise ValueError(f"raw source changed before enrichment: {path}")
    question_fields = [task_question_fields(t["dataset_key"], t["initial_prompt"]) for t in prepared["turns"]]
    if any(fields["task_question_reason"] for fields in question_fields):
        raise ValueError("original task question unavailable from native initial prompt")
    observations = _read_selected_observations(path)
    if _stat_fingerprint(path.stat()) != prepared["raw_stat"]:
        raise ValueError(f"raw source changed during enrichment: {path}")
    enriched = copy.deepcopy(prepared)
    enriched.pop("prepared_identity", None)
    enriched["structured_input_provenance"] = {
        "contract": STRUCTURED_INPUT_CONTRACT, "raw_path": str(path),
        "raw_sha256": prepared["raw_sha256"], "raw_stat": prepared["raw_stat"],
        "fingerprint_verification": "saved_sha256_and_unchanged_size_mtime_inode",
        "observation_source": "raw.turns[turn_idx].candidates[selected_idx].observation",
    }
    for turn in enriched["turns"]:
        if turn["turn_idx"] not in observations:
            raise ValueError("requested turn missing from native raw layout")
        index, observation = observations[turn["turn_idx"]]
        if index != turn["selected_idx"]:
            raise ValueError("prepared and raw selected_idx disagree")
        turn.update(task_question_fields(turn["dataset_key"], turn["initial_prompt"]))
        turn.update(_observation_fields(index, observation))
        turn["structured_input_contract"] = STRUCTURED_INPUT_CONTRACT
        turn.pop("input_identity", None)
        turn["input_identity"] = content_identity(turn)
    return enriched


class PreparedStepTokenizer:


    def __init__(self, step_tokenization):
        self.by_text = step_tokenization
        self.by_tokens = {tuple(value["tokens"]): list(value["token_ids"])
                          for value in step_tokenization.values()}

    def tokenize(self, text):
        return list(self.by_text[text]["tokens"])

    def convert_tokens_to_ids(self, tokens):
        return list(self.by_tokens[tuple(tokens)])


def _source_tokenizer(name, *, model_name="", source_environment=None):
    expected = source_environment or (
        "gruet-gemma" if "gemma" in model_name.lower() else "gruet")
    if os.environ.get("CONDA_DEFAULT_ENV") != expected:
        raise ValueError(f"source tokenizer {name} requires Conda environment {expected}")
    from agent_uq.uq.graphuq_runtime import _get_tokenizer
    return _get_tokenizer(name)


class _LazySourceTokenizer:
    def __init__(self, resolver, source_environment=None):
        self.resolver = resolver
        self.source_environment = source_environment
        self.tokenizer = None

    def bind(self, name, model_name):
        self.tokenizer = (self.resolver(name) if self.resolver is not None else
                          _source_tokenizer(name, model_name=model_name,
                                            source_environment=self.source_environment))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def _tokenizer_revision(tokenizer, explicit):
    if explicit:
        return explicit
    revision = (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if revision:
        return str(revision)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return "tokenizer_sha256:" + content_identity({
            "backend": backend.to_str(),
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        })
    raise ValueError("source tokenizer revision unavailable; supply tokenizer_revision")


def _stat_fingerprint(stat):
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def prepare_record(raw_path, *, run_id, tokenizer, turn_indices=None, tokenizer_revision=None):
    raw_path = Path(raw_path).resolve()
    before = raw_path.stat()
    encoded = raw_path.read_bytes()
    after = raw_path.stat()
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ValueError(f"raw source changed during preparation: {raw_path}")
    document = gzip.decompress(encoded) if raw_path.suffix == ".gz" else encoded
    raw = _from_jsonable_raw(json.loads(document))
    if isinstance(tokenizer, _LazySourceTokenizer):
        tokenizer.bind(raw.tokenizer_name, raw.model)
    revision = _tokenizer_revision(tokenizer, tokenizer_revision)
    indices = [turn.turn_idx for turn in raw.turns]
    wanted = indices if turn_indices is None else list(turn_indices)
    if len(indices) != len(set(indices)) or len(wanted) != len(set(wanted)) or not set(wanted).issubset(indices):
        raise ValueError(f"invalid or missing requested turn indices: {raw_path}")
    metadata = dataclasses.asdict(build_scored_shell(raw, raw_path=raw_path))
    for field in ("raw_path", "estimators", "turns"):
        metadata.pop(field)
    prepared = {"schema": "reasoning_prepared_v1", "run_id": run_id,
                "record_id": raw.record_id, "raw_path": str(raw_path),
                "raw_sha256": hashlib.sha256(encoded).hexdigest(),
                "raw_stat": _stat_fingerprint(before), "scored_metadata": metadata, "turns": []}
    first_prompt = raw.turns[0].prompt if raw.turns else ""
    for turn in raw.turns:
        if turn.turn_idx not in wanted:
            continue
        entry = {"turn_idx": turn.turn_idx, "prompt": turn.prompt, "initial_prompt": first_prompt,
                 "selected_idx": turn.selected_idx, "tokenizer_name": raw.tokenizer_name,
                 "tokenizer_revision": revision, "model": raw.model, "model_name": raw.model,
                 "dataset_key": raw.dataset_key, "accepted_span_contract": CURRENT_TURN_ADAPTER_CONTRACT,
                 "accepted_source_format": None, "selected_candidate_digest": None,
                 "accepted_text": "", "thought_text": "", "action_text": "", "token_ids": [],
                 "token_logprobs": [], "preparation_reason": None,
                 "step_tokenization": {}, "cot_preparation_reason": None}
        entry.update(task_question_fields(raw.dataset_key, first_prompt))
        entry.update(_observation_fields(turn.selected_idx, None if turn.selected_idx is None or not 0 <= turn.selected_idx < len(turn.candidates) else turn.candidates[turn.selected_idx].observation))
        entry["structured_input_contract"] = STRUCTURED_INPUT_CONTRACT
        if turn.selected_idx is None:
            entry["preparation_reason"] = "selected_idx_missing"
        elif not 0 <= turn.selected_idx < len(turn.candidates):
            raise ValueError("stored selected_idx out of bounds")
        else:
            selected = turn.candidates[turn.selected_idx]
            entry["selected_candidate_digest"] = selected_candidate_digest(selected)
            entry["accepted_source_format"] = (selected.backend or {}).get("source_format")
            if selected.compliant is not True or selected.parsed_action is None:
                entry["preparation_reason"] = "selected_candidate_not_compliant"
            else:
                accepted, reason = recover_accepted_span_candidate(selected, raw_index=turn.selected_idx, tokenizer=tokenizer)
                if accepted is None:
                    entry["preparation_reason"] = reason
                else:
                    entry.update(accepted_text=accepted.text, thought_text=accepted.thought_text,
                                 action_text=accepted.action_text, token_ids=accepted.token_ids,
                                 token_logprobs=accepted.token_logprobs)
                    try:
                        steps = stored_thought_steps(accepted.thought_text)
                    except ValueError as error:
                        entry["cot_preparation_reason"] = str(error)
                    else:
                        for text in steps.values():
                            tokens = tokenizer.tokenize(text)
                            entry["step_tokenization"][text] = {"tokens": list(tokens),
                                "token_ids": list(tokenizer.convert_tokens_to_ids(tokens))}
        entry["input_identity"] = content_identity(entry)
        prepared["turns"].append(entry)
    return prepared


def _adopt_prepared(record, previous, *, run_id, inventory_identity, tokenizer_revision):
    if (not isinstance(previous, dict) or previous.get("inventory_identity") != inventory_identity
            or previous.get("record_id") != record["record_id"] or previous.get("run_id") != run_id):
        return None
    try:
        if previous["raw_path"] != str(Path(record["raw_path"]).resolve()):
            return None
        if previous["raw_stat"] != _stat_fingerprint(Path(record["raw_path"]).stat()):
            return None
        prepared = json.loads(Path(previous["prepared_path"]).read_text(encoding="utf-8"))
        identity_payload = {key: value for key, value in prepared.items() if key != "prepared_identity"}
        if (any(turn.get("structured_input_contract") != STRUCTURED_INPUT_CONTRACT for turn in prepared["turns"])
                or prepared["prepared_identity"] != previous["prepared_identity"]
                or content_identity(identity_payload) != previous["prepared_identity"]
                or prepared["record_id"] != record["record_id"]
                or prepared["run_id"] != run_id
                or prepared["raw_path"] != previous["raw_path"]
                or prepared["raw_sha256"] != previous["raw_sha256"]
                or prepared["raw_stat"] != previous["raw_stat"]
                or prepared["scored_path"] != str(Path(record["scored_path"]).resolve())
                or previous["scored_path"] != prepared["scored_path"]
                or [turn["turn_idx"] for turn in prepared["turns"]] != record["turn_indices"]
                or previous["turn_indices"] != record["turn_indices"]
                or previous["input_identities"] != {str(turn["turn_idx"]): turn["input_identity"] for turn in prepared["turns"]}
                or tokenizer_revision is not None and any(turn["tokenizer_revision"] != tokenizer_revision for turn in prepared["turns"])):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return previous


def prepare_inventory(inventory, artifact_root, *, tokenizer_resolver=None, run_id=None, tokenizer_revision=None,
                      source_environment=None):

    if isinstance(inventory, (str, Path)):
        inventory = json.loads(Path(inventory).read_text(encoding="utf-8"))
    runs = inventory["runs"]
    totals = {"runs": len(runs), "records": sum(len(run["records"]) for run in runs),
              "turns": sum(len(rec["turn_indices"]) for run in runs for rec in run["records"])}
    if inventory["totals"] != totals or len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("invalid canonical inventory counts or duplicate run IDs")
    if run_id is not None and run_id not in {run["run_id"] for run in runs}:
        raise ValueError(f"unknown inventory run_id: {run_id}")
    artifact_root = Path(artifact_root).resolve()
    inventory_identity = content_identity(inventory)
    manifest_path = artifact_root / "preparation_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    previous_records = {(rec["run_id"], rec["record_id"]): rec
                        for rec in previous.get("records", [])}
    records = []
    for run in runs:
        seen = set()
        for record in run["records"]:
            record_id = record["record_id"]
            if record_id in seen:
                raise ValueError(f"duplicate inventory record ID: {run['run_id']}/{record_id}")
            seen.add(record_id)
            progress_path = (artifact_root / "preparations" / inventory_identity / "progress" /
                             run["run_id"] / (content_identity(record_id) + ".json"))
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                progress = previous_records.get((run["run_id"], record_id))
            adopted = _adopt_prepared(record, progress,
                                      run_id=run["run_id"], inventory_identity=inventory_identity,
                                      tokenizer_revision=tokenizer_revision)
            if adopted is not None:
                records.append(adopted)
                continue
            if run_id is not None and run["run_id"] != run_id:
                continue
            prepared = prepare_record(record["raw_path"], run_id=run["run_id"],
                                      tokenizer=_LazySourceTokenizer(tokenizer_resolver, source_environment),
                                      turn_indices=record["turn_indices"], tokenizer_revision=tokenizer_revision)
            if prepared["record_id"] != record_id:
                raise ValueError(f"inventory record identity mismatch: {record['raw_path']}")
            prepared["scored_path"] = str(Path(record["scored_path"]).resolve())
            prepared["prepared_identity"] = content_identity(prepared)
            prepared_path = (artifact_root / "preparations" / prepared["prepared_identity"] /
                             "prepared" / run["run_id"] / (content_identity(record_id) + ".json"))
            write_json_atomic(prepared_path, prepared)
            reference = {"run_id": run["run_id"], "record_id": record_id,
                            "inventory_identity": inventory_identity,
                            "raw_path": prepared["raw_path"], "scored_path": prepared["scored_path"],
                            "raw_sha256": prepared["raw_sha256"], "raw_stat": prepared["raw_stat"],
                            "prepared_identity": prepared["prepared_identity"], "prepared_path": str(prepared_path),
                            "turn_indices": [turn["turn_idx"] for turn in prepared["turns"]],
                            "input_identities": {str(turn["turn_idx"]): turn["input_identity"] for turn in prepared["turns"]}}
            write_json_atomic(progress_path, reference)
            records.append(reference)
    records.sort(key=lambda record: (record["run_id"], record["record_id"]))
    prepared_totals = {"runs": len({record["run_id"] for record in records}), "records": len(records),
                       "turns": sum(len(record["turn_indices"]) for record in records)}
    complete = prepared_totals == totals
    full_identity = content_identity({"inventory_identity": inventory_identity,
        "records": [[record["run_id"], record["record_id"], record["prepared_identity"]] for record in records]}) if complete else None
    manifest = {"schema": "reasoning_preparation_manifest_v1", "status": "complete" if complete else "incomplete",
                "inventory": inventory, "inventory_identity": inventory_identity,
                "prepared_inventory_identity": full_identity, "totals": totals,
                "prepared_totals": prepared_totals, "records": records}
    if complete:
        write_json_atomic(artifact_root / "preparations" / full_identity / "manifest.json", manifest)
    write_json_atomic(artifact_root / "inventory.json", inventory)
    write_json_atomic(manifest_path, manifest)
    return manifest


def enrich_preparation_manifest(source_manifest, artifact_root, *, workers=1):

    if isinstance(source_manifest, (str, Path)):
        source_manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete" or not source_manifest.get("prepared_inventory_identity"):
        raise ValueError("enrichment requires a complete preparation manifest")
    root = Path(artifact_root).resolve()
    source_identity = source_manifest["prepared_inventory_identity"]
    records = []
    arguments = [(source_identity, root, reference) for reference in source_manifest["records"]]
    if workers == 1:
        records = [_enrich_manifest_reference(argument) for argument in arguments]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_enrich_manifest_reference, arguments))
    full_identity = content_identity({"inventory_identity": source_manifest["inventory_identity"],
        "records": [[r["run_id"], r["record_id"], r["prepared_identity"]] for r in records]})
    manifest = {**source_manifest, "prepared_inventory_identity": full_identity, "records": records,
                "source_prepared_inventory_identity": source_identity, "structured_input_contract": STRUCTURED_INPUT_CONTRACT}
    write_json_atomic(root / "preparations" / full_identity / "manifest.json", manifest)
    write_json_atomic(root / "inventory.json", manifest["inventory"])
    write_json_atomic(root / "preparation_manifest.json", manifest)
    return manifest


def _enrich_manifest_reference(arguments):
    source_identity, root, reference = arguments
    source_path = Path(reference["prepared_path"])
    if root == source_path.parent or root in source_path.parents:
        raise ValueError("enrichment artifact root must be separate from original artifacts")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload = {k: v for k, v in source.items() if k != "prepared_identity"}
    if (content_identity(source_payload) != reference["prepared_identity"]
            or source.get("prepared_identity") != reference["prepared_identity"]
            or source["record_id"] != reference["record_id"] or source["run_id"] != reference["run_id"]
            or source["raw_stat"] != reference["raw_stat"] or source["raw_sha256"] != reference["raw_sha256"]):
        raise ValueError("source prepared identity or provenance mismatch")
    progress_path = root / "enrichment_progress" / source_identity / reference["run_id"] / (content_identity(reference["record_id"]) + ".json")
    try:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        cached = json.loads(Path(saved["prepared_path"]).read_text(encoding="utf-8"))
        payload = {k: v for k, v in cached.items() if k != "prepared_identity"}
        valid = (saved.get("source_prepared_identity") == reference["prepared_identity"]
                 and content_identity(payload) == saved["prepared_identity"] == cached["prepared_identity"]
                 and cached.get("structured_input_provenance", {}).get("contract") == STRUCTURED_INPUT_CONTRACT
                 and _stat_fingerprint(Path(source["raw_path"]).stat()) == source["raw_stat"])
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if valid:
        return saved
    enriched = enrich_prepared_record(source)
    enriched["source_prepared_identity"] = reference["prepared_identity"]
    enriched["prepared_identity"] = content_identity(enriched)
    prepared_path = root / "preparations" / enriched["prepared_identity"] / "prepared" / reference["run_id"] / (content_identity(reference["record_id"]) + ".json")
    write_json_atomic(prepared_path, enriched)
    saved = {**reference, "prepared_identity": enriched["prepared_identity"], "prepared_path": str(prepared_path),
             "source_prepared_identity": reference["prepared_identity"],
             "input_identities": {str(t["turn_idx"]): t["input_identity"] for t in enriched["turns"]}}
    write_json_atomic(progress_path, saved)
    return saved
