from __future__ import annotations

import logging
import pathlib
from typing import Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from agent_uq.records import read_raw
from agent_uq.scoring.artifact_dirs import (
    resolve_graph_artifact_dir,
    resolve_pairwise_nli_artifact_dir,
)
from agent_uq.scoring.execution import ScoreExecutionPlan, shard_raw_paths
from agent_uq.scoring.params import (
    apply_score_execution_to_param_views,
    derive_estimator_param_views,
)
from agent_uq.scoring.record_scorer import _score_one_record
from agent_uq.uq.base import get_estimator
from agent_uq.uq.graph_artifact import (
    GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT,
    GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE,
    normalize_graph_artifact_cache_adoption,
    normalize_graph_artifact_cache_mode,
    prepare_graph_artifact_cache_root,
)

log = logging.getLogger("agent_uq.scoring.worker")


def _run_score_worker(
    cfg: DictConfig,
    *,
    plan: ScoreExecutionPlan,
    raw_files: Sequence[pathlib.Path],
) -> None:
    out_dir = pathlib.Path(cfg.output_dir)
    scored_dir = out_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    pairwise_nli_artifact_dir = resolve_pairwise_nli_artifact_dir(
        out_dir,
        use_pairwise_nli_artifact_cache=bool(
            cfg.get("use_pairwise_nli_artifact_cache", True)
        ),
    )
    graph_artifact_dir = resolve_graph_artifact_dir(
        out_dir,
        use_graph_artifact_cache=bool(cfg.get("use_graph_artifact_cache", True)),
        configured_graph_artifact_dir=cfg.get("graph_artifact_dir"),
    )
    graph_artifact_cache_mode = normalize_graph_artifact_cache_mode(
        cfg.get("graph_artifact_cache_mode", GRAPH_ARTIFACT_CACHE_MODE_READ_WRITE)
    )
    graph_artifact_cache_adoption = normalize_graph_artifact_cache_adoption(
        cfg.get("graph_artifact_cache_adoption", GRAPH_ARTIFACT_CACHE_ADOPTION_REJECT)
    )
    graph_artifact_cache_profile = cfg.get("graph_artifact_cache_profile")
    graph_artifact_cache_profile_plain = (
        OmegaConf.to_container(graph_artifact_cache_profile, resolve=True)
        if graph_artifact_cache_profile is not None
        else None
    )
    if graph_artifact_cache_profile_plain is not None and not isinstance(
        graph_artifact_cache_profile_plain,
        Mapping,
    ):
        raise TypeError(
            "graph_artifact_cache_profile must resolve to a mapping when provided"
        )
    if graph_artifact_dir is not None:
        prepare_graph_artifact_cache_root(
            graph_artifact_dir,
            expected_profile=graph_artifact_cache_profile_plain,
            access_mode=graph_artifact_cache_mode,
            adoption=graph_artifact_cache_adoption,
        )

    estimator_name: str = cfg.estimator.name
    param_views = derive_estimator_param_views(
        cfg.estimator,
        sampled_model_name=str(cfg.model.name),
    )
    runtime_views = apply_score_execution_to_param_views(
        estimator_name,
        param_views,
        plan=plan,
    )
    estimator = get_estimator(estimator_name, **runtime_views.constructor_kwargs)
    log.info(
        "scoring with estimator %s (constructor_kwargs=%s, persisted_params=%s, runtime=%s)",
        estimator_name,
        runtime_views.constructor_kwargs,
        runtime_views.persisted_params,
        runtime_views.runtime_metadata,
    )

    worker_raw_files = list(raw_files)
    if plan.worker_index is not None:
        worker_raw_files = shard_raw_paths(
            worker_raw_files,
            num_workers=plan.num_workers,
            worker_index=plan.worker_index,
        )
        log.info(
            "worker %d/%d assigned %d raw records",
            plan.worker_index,
            plan.num_workers,
            len(worker_raw_files),
        )

    for index, raw_path in enumerate(worker_raw_files):
        try:
            raw = read_raw(raw_path)
        except Exception:
            log.exception("failed to read raw record %s", raw_path)
            continue
        log.info("[%d/%d] %s", index + 1, len(worker_raw_files), raw.record_id)
        out = _score_one_record(
            raw,
            estimator,
            estimator_name,
            scored_dir,
            gzip_records=bool(cfg.gzip_records),
            pairwise_nli_artifact_dir=pairwise_nli_artifact_dir,
            graph_artifact_dir=graph_artifact_dir,
            graph_artifact_cache_mode=graph_artifact_cache_mode,
            raw_source_path=raw_path,
            estimator_params=runtime_views.persisted_params,
            estimator_runtime=runtime_views.runtime_metadata,
        )
        log.info("  -> %s", out.name)

    log.info(
        "scoring complete: %d records scored under %s",
        len(worker_raw_files),
        scored_dir,
    )
