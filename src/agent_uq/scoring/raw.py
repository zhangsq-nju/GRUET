from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence

from omegaconf import DictConfig, OmegaConf

from agent_uq.scoring.execution import ScoreExecutionPlan, resolve_score_execution_plan

log = logging.getLogger("agent_uq.scoring.raw")

INTERNAL_SCORE_CFG_PATH_ENV = "AGENT_UQ_INTERNAL_SCORE_CFG_PATH"
INTERNAL_SCORE_RAW_FILES_PATH_ENV = "AGENT_UQ_INTERNAL_SCORE_RAW_FILES_PATH"


def load_internal_score_cfg_override() -> DictConfig | None:
    cfg_path = os.environ.get(INTERNAL_SCORE_CFG_PATH_ENV)
    if not cfg_path:
        return None
    loaded = OmegaConf.load(cfg_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            "internal score cfg override must resolve to a DictConfig; "
            f"got {type(loaded).__name__}"
        )
    return loaded


def load_internal_score_raw_files() -> list[pathlib.Path] | None:
    manifest_path = os.environ.get(INTERNAL_SCORE_RAW_FILES_PATH_ENV)
    if not manifest_path:
        return None
    payload = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(path_str, str) or not path_str for path_str in payload
    ):
        raise ValueError(
            "internal explicit-raw score manifest must be a JSON list of non-empty strings"
        )
    return [pathlib.Path(path_str) for path_str in payload]


def make_internal_score_worker_cfg(
    cfg: DictConfig,
    *,
    worker_index: int,
) -> DictConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            "score worker cfg must resolve to a mapping before internal fan-out; "
            f"got {type(raw).__name__}"
        )
    worker_cfg = OmegaConf.create(raw)
    worker_cfg.score_execution.worker_index = worker_index
    return worker_cfg


def run_explicit_raw_score_parent(
    cfg: DictConfig,
    *,
    plan: ScoreExecutionPlan,
    raw_files: Sequence[pathlib.Path],
    popen_factory: Callable[..., subprocess.Popen[Any]] | None = None,
) -> None:
    if popen_factory is None:
        popen_factory = subprocess.Popen

    with tempfile.TemporaryDirectory(prefix="agent_uq_score_raw_subset_") as tmp_dir_str:
        tmp_dir = pathlib.Path(tmp_dir_str)
        raw_manifest_path = tmp_dir / "raw_files.json"
        raw_manifest_path.write_text(
            json.dumps([str(path.resolve()) for path in raw_files], indent=2),
            encoding="utf-8",
        )

        worker_processes: list[tuple[int, subprocess.Popen[Any]]] = []
        for worker_index in range(plan.num_workers):
            worker_cfg_path = tmp_dir / f"worker_{worker_index}.yaml"
            OmegaConf.save(
                make_internal_score_worker_cfg(cfg, worker_index=worker_index),
                worker_cfg_path,
            )

            env = dict(os.environ)
            env[INTERNAL_SCORE_CFG_PATH_ENV] = str(worker_cfg_path)
            env[INTERNAL_SCORE_RAW_FILES_PATH_ENV] = str(raw_manifest_path)
            if plan.devices:
                env["AGENT_UQ_SCORE_WORKER_DEVICE"] = plan.devices[worker_index]

            hydra_run_dir = (
                pathlib.Path(cfg.output_dir) / f"hydra_score_worker_{worker_index}"
            )
            cmd = [
                sys.executable,
                "-m",
                "agent_uq.cli.score",
                f"output_dir={cfg.output_dir}",
                f"hydra.run.dir={hydra_run_dir}",
            ]
            log.info(
                "launching explicit-raw score worker %d/%d: %s",
                worker_index,
                plan.num_workers,
                " ".join(str(part) for part in cmd),
            )
            worker_processes.append((worker_index, popen_factory(cmd, env=env)))

        failures: list[tuple[int, int]] = []
        for worker_index, process in worker_processes:
            returncode = process.wait()
            if returncode != 0:
                failures.append((worker_index, returncode))
        if failures:
            raise RuntimeError(
                "one or more explicit-raw score workers failed: "
                + ", ".join(
                    f"worker {worker_index} exited {returncode}"
                    for worker_index, returncode in failures
                )
            )


def run_score_with_raw_files(
    cfg: DictConfig,
    *,
    raw_files: Sequence[pathlib.Path],
    worker: Callable[..., None] | None = None,
    popen_factory: Callable[..., subprocess.Popen[Any]] | None = None,
) -> None:


    if worker is None:
        from agent_uq.scoring.worker import _run_score_worker as worker

    plan = resolve_score_execution_plan(cfg.get("score_execution"))
    if plan.worker_index is not None or plan.num_workers == 1:
        worker(cfg, plan=plan, raw_files=raw_files)
        return

    log.info(
        "internal explicit-raw scoring executes %d worker shard(s) via subprocess fan-out over %d raw record(s)",
        plan.num_workers,
        len(raw_files),
    )
    run_explicit_raw_score_parent(
        cfg,
        plan=plan,
        raw_files=raw_files,
        popen_factory=popen_factory,
    )
