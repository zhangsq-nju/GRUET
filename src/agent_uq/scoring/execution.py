from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from agent_uq.uq.pairwise_nli_artifact import derive_nli_execution_mode


@dataclass(frozen=True)
class ScoreExecutionPlan:


    num_workers: int
    devices: tuple[str, ...]
    worker_index: int | None
    assigned_device: str | None
    execution_mode: str | None


def _score_execution_to_container(
    cfg_score_execution: DictConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if cfg_score_execution is None:
        return {}
    if isinstance(cfg_score_execution, DictConfig):
        raw = OmegaConf.to_container(cfg_score_execution, resolve=True)
    else:
        raw = dict(cfg_score_execution)
    if not isinstance(raw, dict):
        raise TypeError(
            "score_execution config must resolve to a mapping; "
            f"got {type(raw).__name__}"
        )
    return raw


def resolve_score_execution_plan(
    cfg_score_execution: DictConfig | Mapping[str, Any] | None,
) -> ScoreExecutionPlan:

    raw = _score_execution_to_container(cfg_score_execution)
    num_workers = raw.get("num_workers", 1)
    devices_raw = raw.get("devices", []) or []
    worker_index = raw.get("worker_index")

    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise ValueError(
            "score_execution.num_workers must be a positive integer; "
            f"got {type(num_workers).__name__} {num_workers!r}"
        )
    if num_workers <= 0:
        raise ValueError(
            "score_execution.num_workers must be >= 1; "
            f"got {num_workers}"
        )
    if worker_index is not None and (
        isinstance(worker_index, bool) or not isinstance(worker_index, int)
    ):
        raise ValueError(
            "score_execution.worker_index must be null or an integer; "
            f"got {type(worker_index).__name__} {worker_index!r}"
        )
    if worker_index is not None and not 0 <= worker_index < num_workers:
        raise ValueError(
            "score_execution.worker_index must be within "
            f"[0, {num_workers}); got {worker_index}"
        )
    if not isinstance(devices_raw, list):
        raise ValueError(
            "score_execution.devices must be a list of device strings; "
            f"got {type(devices_raw).__name__}"
        )
    devices = tuple(str(device) for device in devices_raw)
    if len(set(devices)) != len(devices):
        raise ValueError(
            "score_execution.devices must not contain duplicates; "
            f"got {list(devices)!r}"
        )
    if devices and len(devices) != num_workers:
        raise ValueError(
            "score_execution.devices length must match score_execution.num_workers; "
            f"got {len(devices)} devices for {num_workers} workers"
        )
    if num_workers > 1 and not devices:
        raise ValueError(
            "score_execution.devices must list one device per worker when "
            "score_execution.num_workers > 1"
        )

    execution_mode: str | None = None
    if devices:
        execution_modes = {derive_nli_execution_mode(device) for device in devices}
        if len(execution_modes) != 1:
            raise ValueError(
                "score_execution.devices must resolve to one execution mode; "
                f"got {sorted(execution_modes)!r} from {list(devices)!r}"
            )
        execution_mode = next(iter(execution_modes))

    assigned_device: str | None = None
    if worker_index is not None and devices:
        assigned_device = devices[worker_index]
    elif worker_index is None and num_workers == 1 and len(devices) == 1:
        assigned_device = devices[0]

    return ScoreExecutionPlan(
        num_workers=int(num_workers),
        devices=devices,
        worker_index=worker_index,
        assigned_device=assigned_device,
        execution_mode=execution_mode,
    )


def shard_raw_paths(
    raw_files: Sequence[pathlib.Path],
    *,
    num_workers: int,
    worker_index: int,
) -> list[pathlib.Path]:

    if num_workers <= 0:
        raise ValueError(f"num_workers must be >= 1; got {num_workers}")
    if not 0 <= worker_index < num_workers:
        raise ValueError(
            f"worker_index must be within [0, {num_workers}); got {worker_index}"
        )
    return [
        raw_path
        for index, raw_path in enumerate(raw_files)
        if index % num_workers == worker_index
    ]


def build_worker_score_cli_args(
    base_args: Sequence[str],
    *,
    worker_index: int,
    hydra_run_dir: pathlib.Path,
) -> list[str]:
    args = [
        arg
        for arg in base_args
        if not arg.startswith("score_execution.worker_index=")
        and not arg.startswith("hydra.run.dir=")
    ]
    args.append(f"score_execution.worker_index={worker_index}")
    args.append(f"hydra.run.dir={hydra_run_dir}")
    return args
