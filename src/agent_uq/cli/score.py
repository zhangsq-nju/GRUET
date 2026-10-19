from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
from typing import Any, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from agent_uq.records import list_record_paths
from agent_uq.scoring.execution import (
    ScoreExecutionPlan as _ScoreExecutionPlan,
    build_worker_score_cli_args as _build_worker_score_cli_args,
    resolve_score_execution_plan as _resolve_score_execution_plan,
)
from agent_uq.scoring.raw import (
    load_internal_score_cfg_override as _load_internal_score_cfg_override,
    load_internal_score_raw_files as _load_internal_score_raw_files,
)
import agent_uq.scoring.worker as scoring_worker

log = logging.getLogger("agent_uq.cli.score")


def _run_score_parent(
    cfg: DictConfig,
    *,
    plan: _ScoreExecutionPlan,
    raw_files: Sequence[pathlib.Path],
    argv: Sequence[str],
) -> None:
    if plan.num_workers == 1:
        scoring_worker._run_score_worker(cfg, plan=plan, raw_files=raw_files)
        return

    worker_processes: list[tuple[int, subprocess.Popen[Any]]] = []
    for worker_index in range(plan.num_workers):
        worker_args = _build_worker_score_cli_args(
            argv,
            worker_index=worker_index,
            hydra_run_dir=pathlib.Path(cfg.output_dir) / f"hydra_score_worker_{worker_index}",
        )
        env = dict(os.environ)
        if plan.devices:
            env["AGENT_UQ_SCORE_WORKER_DEVICE"] = plan.devices[worker_index]
        cmd = [sys.executable, "-m", "agent_uq.cli.score", *worker_args]
        log.info("launching score worker %d/%d: %s", worker_index, plan.num_workers, " ".join(cmd))
        worker_processes.append(
            (worker_index, subprocess.Popen(cmd, env=env))
        )

    failures: list[tuple[int, int]] = []
    for worker_index, process in worker_processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((worker_index, returncode))
    if failures:
        raise RuntimeError(
            "one or more score workers failed: "
            + ", ".join(
                f"worker {worker_index} exited {returncode}"
                for worker_index, returncode in failures
            )
        )


@hydra.main(version_base=None, config_path="../conf", config_name="agent_uq")
def main(cfg: DictConfig) -> None:
    cfg_override = _load_internal_score_cfg_override()
    if cfg_override is not None:
        cfg = cfg_override
    log.info("agent_uq score config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = pathlib.Path(cfg.output_dir)
    raw_dir = out_dir / "raw"
    raw_files = _load_internal_score_raw_files()
    if raw_files is None and not raw_dir.is_dir():
        raise FileNotFoundError(
            f"agent_uq.cli.score: no raw directory at {raw_dir}. Run agent_uq.cli.run first."
        )
    if raw_files is None:
        raw_files = list_record_paths(raw_dir, reject_duplicates=True)
        log.info("found %d raw records under %s", len(raw_files), raw_dir)
    else:
        log.info(
            "found %d explicit raw records from internal subset manifest",
            len(raw_files),
        )
    plan = _resolve_score_execution_plan(cfg.get("score_execution"))
    if plan.worker_index is None:
        _run_score_parent(
            cfg,
            plan=plan,
            raw_files=raw_files,
            argv=sys.argv[1:],
        )
    else:
        scoring_worker._run_score_worker(
            cfg,
            plan=plan,
            raw_files=raw_files,
        )


if __name__ == "__main__":
    main()
