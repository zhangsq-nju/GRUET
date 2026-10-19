from __future__ import annotations

import copy
import logging
import math
import pathlib
from typing import Any, Mapping

from agent_uq.records import (
    RawTrajectory,
    ScoredTrajectory,
    build_scored_shell,
    find_record_path,
    read_scored,
    write_scored,
)
from agent_uq.uq.base import EstimatorInfrastructureError, TurnContext
from agent_uq.uq.graph_artifact import (
    GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    GraphArtifactError,
    GraphArtifactRecordSession,
)
from agent_uq.uq.lm_polygraph import extract_task_context_text
from agent_uq.uq.pairwise_nli_artifact import (
    PairwiseNLIArtifactError,
    PairwiseNLIArtifactRecordSession,
)

log = logging.getLogger("agent_uq.scoring.worker")


def _score_one_record(
    raw: RawTrajectory,
    estimator,
    estimator_name: str,
    out_dir: pathlib.Path,
    *,
    gzip_records: bool,
    pairwise_nli_artifact_dir: pathlib.Path | None = None,
    graph_artifact_dir: pathlib.Path | None = None,
    graph_artifact_cache_mode: str = GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    raw_source_path: pathlib.Path | str | None = None,
    estimator_params: dict[str, Any] | None = None,
    estimator_runtime: Mapping[str, Any] | None = None,
) -> pathlib.Path:

    def _merge_existing_scored(
        shell: ScoredTrajectory,
        existing: ScoredTrajectory,
    ) -> ScoredTrajectory:
        shell.estimators = {}
        for name, meta in existing.estimators.items():
            shell.estimators[name] = copy.deepcopy(meta)
        turns_by_idx = {turn.turn_idx: turn for turn in shell.turns}
        for turn in existing.turns:
            if turn.turn_idx not in turns_by_idx:
                continue
            turns_by_idx[turn.turn_idx].uncertainty_estimates = {
                name: dict(entry)
                for name, entry in turn.uncertainty_estimates.items()
            }
        return shell

    scored = build_scored_shell(
        raw,
        raw_path=raw_source_path,
        leaf_root=out_dir.parent,
    )
    existing_scored_path = find_record_path(
        out_dir,
        raw.record_id,
        prefer_gzip=gzip_records,
        reject_duplicates=False,
    )
    if existing_scored_path is not None:
        scored = _merge_existing_scored(scored, read_scored(existing_scored_path))


    scored.estimators.pop(estimator_name, None)
    for turn in scored.turns:
        turn.uncertainty_estimates.pop(estimator_name, None)


    pairwise_session: PairwiseNLIArtifactRecordSession | None = None
    if pairwise_nli_artifact_dir is not None:
        pairwise_session = PairwiseNLIArtifactRecordSession(
            pairwise_nli_artifact_dir=pairwise_nli_artifact_dir,
            record_id=raw.record_id,
            dataset_key=raw.dataset_key,
            gzip_records=gzip_records,
        )

    session: GraphArtifactRecordSession | None = None
    if graph_artifact_dir is not None:
        session = GraphArtifactRecordSession(
            graph_artifact_dir=graph_artifact_dir,
            record_id=raw.record_id,
            dataset_key=raw.dataset_key,
            tokenizer_name=raw.tokenizer_name,
            gzip_records=gzip_records,
            access_mode=graph_artifact_cache_mode,
        )

    turns_by_idx = {turn.turn_idx: turn for turn in scored.turns}
    effective_params = dict(estimator_params or {})
    initial_prompt = raw.turns[0].prompt if raw.turns else ""
    for raw_turn in raw.turns:
        excluded_candidate_indices = [
            idx
            for idx, cand in enumerate(raw_turn.candidates)
            if not (cand.compliant and cand.parsed_action is not None)
        ]
        ctx = TurnContext(
            record_id=raw.record_id,
            dataset_key=raw.dataset_key,
            turn_idx=raw_turn.turn_idx,
            tokenizer_name=raw.tokenizer_name,
            model_name=raw.model,
            prompt=raw_turn.prompt,
            initial_prompt=initial_prompt,
            task_context_text=extract_task_context_text(raw_turn.prompt),
            selection_policy=raw_turn.selection_policy,
            selected_idx=raw_turn.selected_idx,
            excluded_candidate_indices=excluded_candidate_indices,
            pairwise_nli_artifact_dir=pairwise_nli_artifact_dir,
            pairwise_nli_artifact_gzip=gzip_records,
            pairwise_nli_artifact_session=pairwise_session,
            graph_artifact_dir=graph_artifact_dir,
            graph_artifact_gzip=gzip_records,
            graph_artifact_session=session,
        )
        try:
            result = estimator.score(raw_turn.candidates, ctx)
        except (PairwiseNLIArtifactError, GraphArtifactError):


            log.exception(
                "persistent-artifact integrity error on %s turn %d for estimator %s",
                raw.record_id, raw_turn.turn_idx, estimator_name,
            )
            raise
        except EstimatorInfrastructureError:
            log.exception(
                "fatal estimator infrastructure failure on %s turn %d for %s",
                raw.record_id, raw_turn.turn_idx, estimator_name,
            )
            raise
        except Exception:
            log.exception(
                "estimator %s failed on %s turn %d", estimator_name, raw.record_id, raw_turn.turn_idx
            )
            continue
        scalar = float(result.turn_uncertainty)
        if math.isnan(scalar):
            log.warning(
                "estimator %s returned NaN on %s turn %d",
                estimator_name, raw.record_id, raw_turn.turn_idx,
            )
        if not effective_params:
            effective_params = dict(result.params)
        target_turn = turns_by_idx[raw_turn.turn_idx]
        entry = {
            "scalar": scalar,
            "extras": result.extras,
        }
        target_turn.uncertainty_estimates[estimator_name] = entry

    estimator_meta: dict[str, Any] = {"params": dict(effective_params)}
    if estimator_runtime:
        estimator_meta["runtime"] = copy.deepcopy(dict(estimator_runtime))
    scored.estimators[estimator_name] = estimator_meta


    if pairwise_session is not None and pairwise_session.has_staged_changes:
        pairwise_session.flush()
    if session is not None and session.has_staged_changes:
        session.flush()

    return write_scored(
        scored,
        out_dir,
        gzip_records=gzip_records,
        raw_path=raw_source_path,
    )
