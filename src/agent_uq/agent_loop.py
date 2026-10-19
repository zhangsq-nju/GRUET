from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol, Sequence

import logging
import re

from agent_uq.actions import (
    action_payload,
    action_tool_name,
    normalize_action,
    render_action,
)
from agent_uq.envs.base import EnvAdapter, StepResult, TaskRecord
from agent_uq.policies.base import SelectableCandidate, SelectionPolicy
from agent_uq.prompt_assembly import HistoryTurn, TaskRecord as PromptTaskRecord, build
from agent_uq.prompts.parser import PARSE_OK, is_compliant_candidate, parse_completion
from agent_uq.records import (
    CandidateRecord,
    RawTrajectory,
    RejectionConfigRec,
    TopKEntryRec,
    TrajectoryConfig,
    Turn,
)
from agent_uq.vllm_sampler import Candidate as VLLMCandidate


log = logging.getLogger("agent_uq.agent_loop")


class Sampler(Protocol):
    def sample(
        self,
        prompt: str,
        *,
        n: int,
        top_k: int,
        **sampling_params: Any,
    ) -> list[VLLMCandidate]: ...


@dataclass
class AgentLoopConfig:
    K: int
    M: int
    N: int
    num_shots: int
    seed: int
    model_name: str
    tokenizer_name: str
    system_prompt: str
    sampling_params: dict[str, Any]


    rejection_enabled: bool = True
    retry_budget_multiplier_per_turn: int = 3
    max_retry_candidates_per_turn: int = 0
    max_retry_candidates_per_episode: int = 0


@dataclass
class AgentLoop:
    sampler: Sampler
    env: EnvAdapter
    policy: SelectionPolicy
    cfg: AgentLoopConfig

    def run(self, task: TaskRecord) -> RawTrajectory:


        per_turn_budget = self.cfg.max_retry_candidates_per_turn or (
            int(self.cfg.retry_budget_multiplier_per_turn) * self.cfg.M
        )
        per_episode_budget = self.cfg.max_retry_candidates_per_episode or (
            per_turn_budget * self.cfg.N
        )
        episode_retry_budget_remaining = per_episode_budget


        episode_exhausted_retry_budget = False


        reset_result = _coerce_reset_result(self.env.reset(task))
        initial_obs = reset_result.observation
        initial_state = (
            None if reset_result.initial_state is None else dict(reset_result.initial_state)
        )
        history: list[HistoryTurn] = []


        if initial_obs:
            history.append(
                HistoryTurn(thought="", action="", observation=initial_obs)
            )

        prompt_task = PromptTaskRecord(
            record_id=task.record_id,
            dataset_key=task.dataset_key,
            adapter_name=task.adapter_name,
            target_type=task.target_type,
            question=task.question,
        )

        turns: list[Turn] = []
        done_reason = "max_turns"
        env_done = False
        env_history_payload: list[dict[str, Any]] = []
        latest_state = (
            None if initial_state is None else dict(initial_state)
        )

        for turn_idx in range(self.cfg.N):
            prompt_context = None
            build_prompt_context = getattr(self.env, "build_prompt_context", None)
            if callable(build_prompt_context):
                prompt_context = build_prompt_context(
                    task,
                    initial_state=initial_state,
                    latest_state=latest_state,
                )
            prompt = build(
                self.cfg.system_prompt,
                prompt_task,
                history,
                prompt_context=prompt_context,
                num_shots=self.cfg.num_shots,
            )
            sampler_extra: dict[str, Any] = {}
            build_sampler_kwargs = getattr(self.env, "build_sampler_kwargs", None)
            if callable(build_sampler_kwargs):
                sampler_extra = dict(build_sampler_kwargs(task) or {})


            cand_records: list[CandidateRecord] = []
            valid_indices: list[int] = []
            attempts_used = 0

            use_rejection = (
                self.cfg.rejection_enabled and episode_retry_budget_remaining > 0
            )
            turn_mode = "retry" if use_rejection else "oneshot"

            if use_rejection:
                turn_budget = min(per_turn_budget, episode_retry_budget_remaining)
                sampler_call_idx = 0
                while len(valid_indices) < self.cfg.M and attempts_used < turn_budget:
                    batch_n = min(self.cfg.M - len(valid_indices), turn_budget - attempts_used)
                    seed_i = (
                        self.cfg.seed
                        + turn_idx * per_turn_budget
                        + sampler_call_idx
                    )
                    raw_candidates = self.sampler.sample(
                        prompt,
                        n=batch_n,
                        top_k=self.cfg.K,
                        seed=seed_i,
                        **sampler_extra,
                        **self.cfg.sampling_params,
                    )
                    sampler_call_idx += 1
                    for vc in raw_candidates:
                        rec = _parse_and_classify(vc)
                        cand_records.append(rec)
                        if rec.compliant:
                            valid_indices.append(len(cand_records) - 1)
                    attempts_used += batch_n

                episode_retry_budget_remaining = max(
                    0, episode_retry_budget_remaining - attempts_used
                )
            else:


                if self.cfg.rejection_enabled and not episode_exhausted_retry_budget:
                    episode_exhausted_retry_budget = True
                seed_i = self.cfg.seed + turn_idx
                raw_candidates = self.sampler.sample(
                    prompt,
                    n=self.cfg.M,
                    top_k=self.cfg.K,
                    seed=seed_i,
                    **sampler_extra,
                    **self.cfg.sampling_params,
                )
                for vc in raw_candidates:
                    rec = _parse_and_classify(vc)
                    cand_records.append(rec)
                    if rec.compliant:
                        valid_indices.append(len(cand_records) - 1)
                attempts_used = len(cand_records)

            compliant_count = len(valid_indices)
            reject_count = attempts_used - compliant_count
            reject_rate = (reject_count / attempts_used) if attempts_used else 0.0
            log.debug(
                "turn=%d mode=%s attempts=%d compliant=%d reject=%d reject_rate=%.2f",
                turn_idx,
                turn_mode,
                attempts_used,
                compliant_count,
                reject_count,
                reject_rate,
            )


            if not valid_indices:
                turns.append(
                    Turn(
                        turn_idx=turn_idx,
                        prompt=prompt,
                        candidates=cand_records,
                        selected_idx=None,
                        selection_policy=self.policy.name,
                        attempts_used=attempts_used,
                        reject_count=reject_count,
                        compliant_count=compliant_count,
                    )
                )
                done_reason = "no_valid_candidates"
                break


            selectable = [
                SelectableCandidate(raw_index=i, candidate=cand_records[i])
                for i in valid_indices
            ]
            chosen_raw_idx = self.policy.choose(selectable)
            assert chosen_raw_idx in valid_indices, (
                f"policy {self.policy.name} returned raw index {chosen_raw_idx} "
                f"not in compliant set {valid_indices}"
            )


            chosen_rec = cand_records[chosen_raw_idx]
            assert chosen_rec.parsed_action is not None
            chosen_action = normalize_action(chosen_rec.parsed_action)
            assert chosen_action is not None
            verb = action_tool_name(chosen_action)
            payload = action_payload(chosen_action)
            try:
                step_result = _step_env(self.env, chosen_action, verb, payload)
                obs_text = step_result.observation
                env_done = bool(step_result.done)
            except Exception as exc:

                obs_text = f"<EnvAdapterError: {type(exc).__name__}: {exc}>"
                step_result = StepResult(
                    observation=obs_text,
                    reward=0.0,
                    done=False,
                    info={"error": True},
                )
                env_done = False

            chosen_rec.observation = obs_text

            turns.append(
                Turn(
                    turn_idx=turn_idx,
                    prompt=prompt,
                    candidates=cand_records,
                    selected_idx=chosen_raw_idx,
                        selection_policy=self.policy.name,
                        attempts_used=attempts_used,
                        reject_count=reject_count,
                        compliant_count=compliant_count,
                        state=(
                            None
                            if step_result.state is None
                            else dict(step_result.state)
                        ),
                        tool_call_ids=(
                            None
                            if step_result.tool_call_ids is None
                            else list(step_result.tool_call_ids)
                        ),
                        step_info=dict(step_result.info or {}),
                    )
            )
            env_history_payload.append(
                {
                    "action": chosen_action,
                    "verb": verb,
                    "payload": payload,
                    "observation": obs_text,
                    "step_reward": step_result.reward,
                    "state": (
                        None if step_result.state is None else dict(step_result.state)
                    ),
                    "tool_call_ids": (
                        None
                        if step_result.tool_call_ids is None
                        else list(step_result.tool_call_ids)
                    ),
                    "info": dict(step_result.info or {}),
                }
            )


            history = list(history) + [
                HistoryTurn(
                    thought=chosen_rec.parsed_thought or "",
                    action=render_action(chosen_action),
                    observation=obs_text,
                )
            ]
            if step_result.state is not None:
                latest_state = dict(step_result.state)

            if env_done:
                done_reason = "solved" if step_result.reward > 0 else "env_done"
                break


        scoring_history_payload = env_history_payload
        if (
            task.dataset_key == "python_mbpp"
            and done_reason == "max_turns"
            and (
                not env_history_payload
                or env_history_payload[-1].get("verb") != "submit"
            )
        ):
            scoring_history_payload = list(env_history_payload) + [
                {
                    "action": None,
                    "verb": "submit",
                    "payload": "__forced_submit_on_max_turns__",
                    "observation": "<forced submit on max_turns>",
                    "step_reward": 0.0,
                    "state": None,
                    "tool_call_ids": None,
                    "info": {"forced_submit_on_max_turns": True},
                }
            ]
            done_reason = "max_turns_forced_submit"
        try:
            final_reward, solved = self.env.score(task, scoring_history_payload)
        except Exception as exc:
            final_reward = 0.0
            solved = False
            done_reason = f"score_error: {type(exc).__name__}"

        return RawTrajectory(
            record_id=task.record_id,
            dataset_key=task.dataset_key,
            target_type=task.target_type,
            env=task.adapter_name,
            model=self.cfg.model_name,
            config=TrajectoryConfig(
                K=self.cfg.K,
                M=self.cfg.M,
                N=self.cfg.N,
                policy=self.policy.name,
                seed=self.cfg.seed,
                rejection=RejectionConfigRec(
                    enabled=bool(self.cfg.rejection_enabled),
                    max_retry_candidates_per_turn=int(per_turn_budget),
                    max_retry_candidates_per_episode=int(per_episode_budget),
                ),
            ),
            tokenizer_name=self.cfg.tokenizer_name,
            initial_state=initial_state,
            turns=turns,
            final_reward=float(final_reward),
            solved=bool(solved),
            done_reason=done_reason,
            episode_exhausted_retry_budget=episode_exhausted_retry_budget,
            env_backend=task.env_backend,
            env_provenance=dict(task.env_provenance or {}),
        )


def _parse_and_classify(vc: VLLMCandidate) -> CandidateRecord:

    backend_action = _extract_backend_action(vc)
    if backend_action is not None:
        thought = (vc.text or "").strip() or None
        compliant = _has_step_marker(vc.text)
        return _vllm_to_record(
            vc,
            parse_status=PARSE_OK,
            parse_error=None,
            parsed_thought=thought,
            parsed_action=backend_action,
            compliant=compliant,
            non_compliant_reason=(None if compliant else "missing_step_marker"),
        )

    parsed = parse_completion(vc.text)
    compliant, reason = is_compliant_candidate(parsed, vc.text)
    return _vllm_to_record(
        vc,
        parse_status=parsed.parse_status,
        parse_error=parsed.parse_error,
        parsed_thought=parsed.parsed_thought,
        parsed_action=(
            normalize_action(
                {
                    "verb": parsed.parsed_action.verb,
                    "payload": parsed.parsed_action.payload,
                    "parser_type": "legacy_text",
                }
            )
            if parsed.parsed_action is not None
            else None
        ),
        compliant=compliant,
        non_compliant_reason=reason,
    )

def _vllm_to_record(
    vc: VLLMCandidate,
    *,
    parse_status: str,
    parse_error: str | None,
    parsed_thought: str | None,
    parsed_action: dict[str, Any] | None,
    compliant: bool,
    non_compliant_reason: str | None,
) -> CandidateRecord:
    return CandidateRecord(
        text=vc.text,
        token_ids=list(vc.token_ids),
        token_logprobs=list(vc.token_logprobs),
        top_k=[
            [TopKEntryRec(token_id=int(e.token_id), logprob=float(e.logprob)) for e in step]
            for step in vc.top_k
        ],
        finish_reason=vc.finish_reason,
        parse_status=parse_status,
        parse_error=parse_error,
        parsed_thought=parsed_thought,
        parsed_action=parsed_action,
        observation=None,
        backend=dict(vc.backend or {}),
        compliant=compliant,
        non_compliant_reason=non_compliant_reason,
    )


def _coerce_reset_result(reset_result: Any):
    if hasattr(reset_result, "observation") and hasattr(reset_result, "initial_state"):
        return reset_result
    return type(
        "_ResetCompat",
        (),
        {"observation": str(reset_result or ""), "initial_state": None},
    )()


def _extract_backend_action(vc: VLLMCandidate) -> dict[str, Any] | None:
    backend = dict(vc.backend or {})
    tool_calls = backend.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        tool_call = dict(tool_calls[0] or {})
    elif isinstance(backend.get("tool_call"), dict):
        tool_call = dict(backend.get("tool_call") or {})
    else:
        return None

    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        tool_name = str(function_payload.get("name") or tool_call.get("name") or "")
        raw_args = function_payload.get("arguments")
    else:
        tool_name = str(tool_call.get("name") or "")
        raw_args = tool_call.get("arguments")
    if not tool_name:
        return None

    arguments: dict[str, Any]
    if isinstance(raw_args, str):
        try:
            parsed_args = {} if not raw_args else json.loads(raw_args)
        except Exception:
            parsed_args = {"payload": raw_args}
        arguments = dict(parsed_args if isinstance(parsed_args, dict) else {"payload": parsed_args})
    elif isinstance(raw_args, dict):
        arguments = dict(raw_args)
    else:
        arguments = {}

    return normalize_action(
        {
            "kind": "tool",
            "tool_name": tool_name,
            "arguments": arguments,
            "text": backend.get("canonical_action_text"),
            "parser_type": "function_calling",
            "tool_call_id": tool_call.get("id"),
            "raw_tool_call": tool_call,
            "backend_source": "tool_call",
        }
    )


_STEP_MARKER_RE = re.compile(r"(?m)^[ \t]*Step\s+\d+\s*:")


def _has_step_marker(text: str | None) -> bool:
    if not text:
        return False
    return _STEP_MARKER_RE.search(text) is not None


def _step_env(
    env: EnvAdapter,
    action: dict[str, Any],
    verb: str,
    payload: str,
) -> StepResult:
    try:
        return env.step(action)
    except TypeError:
        return env.step(verb, payload)
