from __future__ import annotations

from pathlib import Path
import re

from agent_uq.uq.base import EstimatorInfrastructureError, UQResult, register_estimator
from agent_uq.uq.reasoning_contender_cache import (
    content_identity, read_result, selected_candidate_digest,
)


class CachedReasoningContenderEstimator:
    def __init__(self, *, name, artifact_dir, artifact_identity):
        if artifact_dir is None or not str(artifact_dir) or str(artifact_dir) == '???':
            raise ValueError('artifact_dir is required')
        if not isinstance(artifact_identity, str) or not artifact_identity or artifact_identity == '???':
            raise ValueError('artifact_identity is required')
        self.name = name
        self.artifact_dir = Path(artifact_dir).resolve()
        self.artifact_identity = artifact_identity

    def _effective_params(self):
        return {'artifact_dir': str(self.artifact_dir),
                'artifact_identity': self.artifact_identity,
                'reasoning_contender_contract': 'source_runtime_cache_v1'}

    def score(self, candidates, context):
        artifact = read_result(self.artifact_dir, context.record_id, context.turn_idx,
                               expected_identity=self.artifact_identity)
        hashes = artifact['input_hashes']
        fields = {'input_identity', 'prompt_hash', 'initial_prompt_hash', 'selected_idx',
                  'selected_candidate_digest', 'model_name', 'tokenizer_name', 'dataset_key'}
        if (set(hashes) != fields or not isinstance(hashes['input_identity'], str)
                or re.fullmatch(r'[0-9a-f]{64}', hashes['input_identity']) is None):
            raise EstimatorInfrastructureError('cached input provenance malformed')
        for field in ('model_name', 'tokenizer_name', 'dataset_key'):
            if hashes[field] != getattr(context, field):
                raise EstimatorInfrastructureError(f'cached {field} mismatch')
        if hashes['prompt_hash'] != content_identity(context.prompt):
            raise EstimatorInfrastructureError('cached current prompt mismatch')
        if hashes['initial_prompt_hash'] != content_identity(context.initial_prompt):
            raise EstimatorInfrastructureError('cached initial prompt mismatch')

        selected_idx = hashes['selected_idx']
        if ((selected_idx is not None and type(selected_idx) is not int)
                or (context.selected_idx is not None and type(context.selected_idx) is not int)
                or selected_idx != context.selected_idx):
            raise EstimatorInfrastructureError('cached selected index mismatch')
        if selected_idx is None:
            if hashes['selected_candidate_digest'] is not None:
                raise EstimatorInfrastructureError('cached selected input mismatch')
        else:
            if not 0 <= selected_idx < len(candidates):
                raise EstimatorInfrastructureError('cached selected index out of bounds')
            try:
                digest = selected_candidate_digest(candidates[selected_idx])
            except (TypeError, ValueError) as error:
                raise EstimatorInfrastructureError('cached selected input malformed') from error
            if digest != hashes['selected_candidate_digest']:
                raise EstimatorInfrastructureError('cached selected input mismatch')
        if artifact['params'][self.name] != self._effective_params():
            raise EstimatorInfrastructureError('cached estimator params mismatch')
        entry = artifact['estimates'][self.name]
        value = entry['scalar']
        extras = dict(entry['extras'])
        if value is None:
            extras['unscorable_kind'] = extras['reason']
        return UQResult(float('nan') if value is None else float(value),
                        extras=extras, params=self._effective_params())


@register_estimator('topouq')
def make_topouq(*, artifact_dir, artifact_identity):
    return CachedReasoningContenderEstimator(name='topouq', artifact_dir=artifact_dir,
                                             artifact_identity=artifact_identity)


@register_estimator('cot_uq_me')
def make_cot_uq_me(*, artifact_dir, artifact_identity):
    return CachedReasoningContenderEstimator(name='cot_uq_me', artifact_dir=artifact_dir,
                                             artifact_identity=artifact_identity)


@register_estimator('cot_uq_sp')
def make_cot_uq_sp(*, artifact_dir, artifact_identity):
    return CachedReasoningContenderEstimator(name='cot_uq_sp', artifact_dir=artifact_dir,
                                             artifact_identity=artifact_identity)
