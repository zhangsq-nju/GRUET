from __future__ import annotations

import datetime
import logging
import pathlib
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from agent_uq.agent_loop import AgentLoop, AgentLoopConfig
from agent_uq.envs.nl2bash_docker import NL2BashDockerEnv, NL2BashDockerEnvConfig
from agent_uq.envs.python_docker import PythonDockerEnv, PythonDockerEnvConfig
from agent_uq.envs.sql_docker import SqlDockerEnv, SqlDockerEnvConfig
from agent_uq.policies.base import get_policy
import agent_uq.policies.builtin
from agent_uq.records import RawTrajectory, list_record_paths, read_raw, write_raw
import agent_uq.sampling.config as sampling_config
from agent_uq.cache.leaf_manifest import (
    LeafDiagnostics,
    aggregate_diagnostics_from_raws,
    lightweight_sampling_cache_validation_mismatches,
    read_leaf_manifest,
    write_leaf_manifest,
)
import agent_uq.vllm_runtime as vllm_runtime

log = logging.getLogger("agent_uq.cli.run")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _validate_raw_coverage(raw_dir: pathlib.Path, expected_ids: list[str]) -> None:
    present = [read_raw(path).record_id
               for path in list_record_paths(raw_dir, reject_duplicates=True)]
    if len(present) != len(expected_ids) or set(present) != set(expected_ids):
        raise ValueError("sampling cache raw/completed target coverage differs; use a new output_dir")


def _build_env(cfg_env: DictConfig):

    name = cfg_env.adapter_name
    backend = str(cfg_env.get("backend", "docker"))
    if backend != "docker":
        raise ValueError(
            f"unsupported env backend {backend!r} for {name!r}; "
            "GRUET InterCode adapters are Docker-only"
        )
    if name == "python":
        return PythonDockerEnv(
            PythonDockerEnvConfig(
                dataset_key=cfg_env.dataset_key,
                dataset_path=cfg_env.dataset_path,
                target_type=cfg_env.target_type,
                terminal_action=cfg_env.terminal_action,
                image=str(cfg_env.get("image", "intercode-python")),
                step_timeout_s=float(
                    cfg_env.get("step_timeout_s", cfg_env.get("exec_timeout_s", 5.0))
                ),
                score_timeout_s=float(cfg_env.get("score_timeout_s", 15.0)),
                per_case_timeout_s=float(cfg_env.get("per_case_timeout_s", 4.0)),
            )
        )
    if name == "sql":
        return SqlDockerEnv(
            SqlDockerEnvConfig(
                dataset_key=cfg_env.dataset_key,
                dataset_path=cfg_env.dataset_path,
                schema_sql_path=cfg_env.schema_sql_path,
                target_type=cfg_env.target_type,
                image=str(cfg_env.get("image", "mysql:latest")),
                root_password=str(cfg_env.get("root_password", "password")),
                step_timeout_s=float(cfg_env.get("step_timeout_s", 5.0)),
                score_timeout_s=float(cfg_env.get("score_timeout_s", 10.0)),
                startup_timeout_s=float(cfg_env.get("startup_timeout_s", 120.0)),
                query_row_limit=int(cfg_env.get("query_row_limit", 200)),
                lower_case_table_names=int(cfg_env.get("lower_case_table_names", 1)),
            )
        )
    if name == "nl2bash":
        return NL2BashDockerEnv(
            NL2BashDockerEnvConfig(
                dataset_key=cfg_env.dataset_key,
                target_type=cfg_env.target_type,
                dataset_path=cfg_env.get("dataset_path", None),
                dataset_splits=cfg_env.get("dataset_splits", None),
                image=str(cfg_env.get("image", "intercode-nl2bash")),
                shell=str(cfg_env.get("shell", "/bin/bash")),
                step_timeout_s=float(
                    cfg_env.get("step_timeout_s", cfg_env.get("exec_timeout_s", 5.0))
                ),
                score_timeout_s=float(
                    cfg_env.get(
                        "score_timeout_s",
                        cfg_env.get("step_timeout_s", cfg_env.get("exec_timeout_s", 5.0)),
                    )
                ),
            )
        )
    raise ValueError(f"unknown env adapter_name: {name!r}")


def _close_env_best_effort(env: Any) -> None:

    primary_exc = sys.exc_info()[1]
    try:
        env.close()
    except Exception:
        if primary_exc is None:
            log.exception("environment cleanup failed during cli.run shutdown")
            return
        log.exception(
            "environment cleanup failed during cli.run shutdown while handling %s",
            type(primary_exc).__name__,
        )


@hydra.main(version_base=None, config_path="../conf", config_name="agent_uq")
def main(cfg: DictConfig) -> None:
    sampling_config.apply_rejection_defaults_and_validate(cfg)
    tensor_parallel_size = int(cfg.model.get("tensor_parallel_size", 1))
    vllm_runtime.configure_vllm_worker_multiproc_method(tensor_parallel_size)
    log.info("agent_uq run config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = pathlib.Path(cfg.output_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)


    env = _build_env(cfg.env)
    try:
        set_leaf_output_dir = getattr(env, "set_leaf_output_dir", None)
        if callable(set_leaf_output_dir):
            set_leaf_output_dir(out_dir)
        records = list(env.load_records())
        mr = cfg.get("max_records", None)
        n_target = int(mr) if mr is not None else len(records)
        records = records[:n_target]
        dataset_source = cfg.env.get(
            "dataset_path",
            cfg.env.get("dataset_key", "<unknown>"),
        )
        log.info("loaded %d records from %s", len(records), dataset_source)


        target_record_ids = [r.record_id for r in records]

        sampling_subobject, runtime_block = (
            sampling_config.extract_effective_sampling_manifest_views(cfg)
        )
        existing_manifest = read_leaf_manifest(out_dir)
        completed_record_ids: list[str] = []
        completed_record_id_set: set[str] = set()
        manifest_diagnostics = LeafDiagnostics(0.0, 0.0, 0)

        if existing_manifest is not None:
            if existing_manifest.target_record_ids != target_record_ids:
                raise ValueError("sampling cache target records differ; use a new output_dir")
            _validate_raw_coverage(raw_dir, existing_manifest.completed_record_ids)
            mismatches = lightweight_sampling_cache_validation_mismatches(
                existing_manifest.sampling,
                sampling_subobject,
                existing_leaf=existing_manifest.leaf,
                effective_leaf={"model": cfg.model.name, "env": cfg.env.dataset_key},
            )
            if mismatches:
                details = "; ".join(mismatches[:8])
                if len(mismatches) > 8:
                    details += f"; ... and {len(mismatches) - 8} more"
                raise ValueError(
                    "sampling cache validation failed for "
                    f"{out_dir}: {details}. Reusing this cache would mix results "
                    "from different sampling settings. Use a new output_dir or "
                    "manually clear/replace the existing sampling cache."
                )
            manifest_diagnostics = existing_manifest.diagnostics
            for record_id in existing_manifest.completed_record_ids:
                if record_id in target_record_ids and record_id not in completed_record_id_set:
                    completed_record_ids.append(record_id)
                    completed_record_id_set.add(record_id)
            if existing_manifest.status == "complete":
                _validate_raw_coverage(raw_dir, target_record_ids)
                log.info(
                    "sampling cache complete: reusing %d raw records under %s",
                    len(completed_record_ids),
                    raw_dir,
                )
                return

        write_leaf_manifest(
            out_dir,
            leaf={"model": cfg.model.name, "env": cfg.env.dataset_key},
            sampling=sampling_subobject,
            runtime=runtime_block,
            target_record_ids=target_record_ids,
            completed_record_ids=completed_record_ids,
            raw_count=len(completed_record_ids),
            diagnostics=manifest_diagnostics,
            completed_at="",
            status="running",
        )

        records = [
            record
            for record in records
            if record.record_id not in completed_record_id_set
        ]
        if completed_record_ids:
            log.info(
                "resuming sampling: %d/%d raw records already completed",
                len(completed_record_ids),
                len(target_record_ids),
            )
        if not records:
            _validate_raw_coverage(raw_dir, target_record_ids)
            write_leaf_manifest(
                out_dir,
                leaf={"model": cfg.model.name, "env": cfg.env.dataset_key},
                sampling=sampling_subobject,
                runtime=runtime_block,
                target_record_ids=target_record_ids,
                completed_record_ids=completed_record_ids,
                raw_count=len(completed_record_ids),
                diagnostics=manifest_diagnostics,
                completed_at=_utc_now(),
                status="complete",
            )
            log.info(
                "run complete: reused %d/%d raw records under %s",
                len(completed_record_ids),
                len(target_record_ids),
                raw_dir,
            )
            return


        from agent_uq.vllm_sampler import VLLMSampler

        sampler = VLLMSampler(
            model_path=cfg.model.model_path,
            tokenizer_path=cfg.model.tokenizer_path,
            dtype=cfg.model.dtype,
            gpu_memory_utilization=float(cfg.model.gpu_memory_utilization),
            trust_remote_code=bool(cfg.model.trust_remote_code),
            tensor_parallel_size=vllm_runtime.validate_tensor_parallel_size(
                tensor_parallel_size
            ),
        )

        policy = get_policy(cfg.policy.name, seed=int(cfg.policy.get("seed", 42)))

        sampling_params = {
            "temperature": float(cfg.sampling.temperature),
            "top_p": float(cfg.sampling.top_p),
            "max_new_tokens": int(cfg.sampling.max_new_tokens),
        }
        stop_sequences = cfg.sampling.get("stop_sequences", None)
        if stop_sequences:
            sampling_params["stop"] = [str(seq) for seq in stop_sequences]
        decoding_top_k = cfg.sampling.get("top_k", None)
        if decoding_top_k is not None:
            sampling_params["decode_top_k"] = int(decoding_top_k)

        loop_cfg = AgentLoopConfig(
            K=int(cfg.sampling.top_k_logprobs),
            M=int(cfg.sampling.num_samples),
            N=int(cfg.sampling.max_turns),
            num_shots=int(cfg.sampling.get("num_shots", 3)),
            seed=int(cfg.sampling.seed),
            model_name=cfg.model.name,
            tokenizer_name=cfg.model.tokenizer_path,
            system_prompt=cfg.get(
                "system_prompt",
                "",
            ),
            sampling_params=sampling_params,
            rejection_enabled=bool(cfg.sampling.rejection.enabled),
            retry_budget_multiplier_per_turn=(
                sampling_config._resolve_retry_budget_multiplier_per_turn(
                    cfg.sampling.rejection
                )
            ),
            max_retry_candidates_per_turn=int(cfg.sampling.rejection.max_retry_candidates_per_turn),
            max_retry_candidates_per_episode=int(cfg.sampling.rejection.max_retry_candidates_per_episode),
        )
        loop = AgentLoop(sampler=sampler, env=env, policy=policy, cfg=loop_cfg)

        n_solved = 0
        successful_raws: list[RawTrajectory] = []
        finalization: dict[str, Any] | None = None
        for i, record in enumerate(records):
            log.info("[%d/%d] %s", i + 1, len(records), record.record_id)
            raw = loop.run(record)
            out_path = write_raw(raw, raw_dir, gzip_records=bool(cfg.gzip_records))
            successful_raws.append(raw)
            if raw.record_id not in completed_record_id_set:
                completed_record_ids.append(raw.record_id)
                completed_record_id_set.add(raw.record_id)
            write_leaf_manifest(
                out_dir,
                leaf={"model": cfg.model.name, "env": cfg.env.dataset_key},
                sampling=sampling_subobject,
                runtime=runtime_block,
                target_record_ids=target_record_ids,
                completed_record_ids=completed_record_ids,
                raw_count=len(completed_record_ids),
                diagnostics=aggregate_diagnostics_from_raws(successful_raws),
                completed_at="",
                status="running",
            )
            log.info(
                "  -> %s  solved=%s reward=%.3f turns=%d done=%s",
                out_path.name, raw.solved, raw.final_reward, len(raw.turns), raw.done_reason,
            )
            if raw.solved:
                n_solved += 1
        log.info(
            "run complete: %d/%d solved, raw records under %s",
            n_solved, len(records), raw_dir,
        )
        finalize_leaf = getattr(env, "finalize_leaf", None)
        if callable(finalize_leaf):
            try:
                finalization = dict(finalize_leaf(successful_raws) or {})
            except Exception as exc:
                log.exception("SWE leaf finalization failed")
                finalization = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }


        diagnostics = (
            aggregate_diagnostics_from_raws(successful_raws)
            if successful_raws
            else manifest_diagnostics
        )
        _validate_raw_coverage(raw_dir, target_record_ids)
        write_leaf_manifest(
            out_dir,
            leaf={"model": cfg.model.name, "env": cfg.env.dataset_key},
            sampling=sampling_subobject,
            runtime=runtime_block,
            target_record_ids=target_record_ids,
            completed_record_ids=completed_record_ids,
            raw_count=len(completed_record_ids),
            diagnostics=diagnostics,
            completed_at=_utc_now(),
            status="complete",
            finalization=finalization,
        )
        log.info(
            "leaf manifest written: %s (raw_count=%d/%d, mean_reject_rate=%.3f, "
            "mean_attempts_used=%.2f, episodes_exhausted=%d)",
            out_dir / "manifest.json",
            len(completed_record_ids),
            len(target_record_ids),
            diagnostics.mean_reject_rate,
            diagnostics.mean_attempts_used,
            diagnostics.episodes_exhausted_retry_budget,
        )
    finally:
        _close_env_best_effort(env)


if __name__ == "__main__":
    main()
