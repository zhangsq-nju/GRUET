from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.metadata
import inspect
from itertools import islice
import json
import math
from pathlib import Path
import re
import time
import uuid

import numpy as np

from agent_uq.uq import cotuq_source as cot, topouq_source as topo
from agent_uq.uq import reasoning_contender_formats as formats
from agent_uq.uq.base import EstimatorInfrastructureError
from agent_uq.uq.reasoning_contender_cache import (
    content_identity, mathematical_runtime, read_stage, stage_cache_key, turn_cache_key,
    validate_estimates, validate_result, write_json_atomic, write_stage,
)
from agent_uq.uq.reasoning_contender_inputs import PreparedStepTokenizer

SAMPLING = {'topo': dict(temperature=.7, max_tokens=2048, top_p=1., seed=None),
            'cot': dict(temperature=1., max_tokens=256, top_p=1., seed=None)}


def _json_value(value):
    import torch
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_value(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if hasattr(value, 'model_dump'):
        return _json_value(value.model_dump(mode='json'))
    if hasattr(value, 'to_dict'):
        return _json_value(value.to_dict())
    raise ValueError(f'cannot persist runtime setting of type {type(value).__name__}')


def _load(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _manifest(root, *, prepared_only=False, require_structured=False):
    path = Path(root) / 'preparation_manifest.json'
    if not path.is_file():
        raise ValueError('complete preparation manifest is required')
    manifest = _load(path)
    if (manifest.get('schema') != 'reasoning_preparation_manifest_v1'
            or manifest.get('status') != 'complete'
            or not manifest.get('prepared_inventory_identity')
            or manifest.get('totals') != manifest.get('prepared_totals')):
        raise ValueError('preparation manifest is incomplete')
    references = manifest['records']
    counts = dict(runs=len({r['run_id'] for r in references}), records=len(references),
                  turns=sum(len(r['turn_indices']) for r in references))
    identity = content_identity(dict(inventory_identity=manifest['inventory_identity'],
        records=[[r['run_id'], r['record_id'], r['prepared_identity']]
                 for r in sorted(references, key=lambda r: (r['run_id'], r['record_id']))]))
    if (counts != manifest['totals'] or identity != manifest['prepared_inventory_identity']
            or content_identity(manifest['inventory']) != manifest['inventory_identity']
            or len({(r['run_id'], r['record_id']) for r in references}) != len(references)):
        raise ValueError('preparation manifest identity/count mismatch')


    for _ in _records(manifest, prepared_root=root, prepared_only=prepared_only,
                      require_structured=require_structured):
        pass
    return manifest


def _validate_raw_source(record, reference, inventory_record, *, prepared_only=False):
    raw_path = inventory_record.get('raw_path')
    fingerprint = record.get('raw_stat')
    if (not isinstance(raw_path, str) or not Path(raw_path).is_absolute()
            or record.get('raw_path') != raw_path or reference.get('raw_path') != raw_path
            or not isinstance(record.get('raw_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', record['raw_sha256'])
            or record.get('raw_sha256') != reference.get('raw_sha256')
            or not isinstance(fingerprint, dict) or set(fingerprint) != {'size', 'mtime_ns', 'inode'}
            or any(type(value) is not int or value < 0 for value in fingerprint.values())
            or content_identity(fingerprint) != content_identity(reference.get('raw_stat'))
            or any(key in inventory_record and content_identity(inventory_record[key]) != content_identity(record[key])
                   for key in ('raw_sha256', 'raw_stat'))):
        raise EstimatorInfrastructureError(f'raw source path/fingerprint mismatch: {raw_path}')
    if prepared_only:
        return
    try:
        stat = Path(raw_path).stat()
    except (OSError, KeyError, TypeError) as error:
        raise EstimatorInfrastructureError('raw source is missing or has an invalid inventory reference') from error
    current = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    if (record.get('raw_path') != raw_path or reference.get('raw_path') != raw_path
            or record.get('raw_stat') != reference.get('raw_stat') or record.get('raw_stat') != current
            or record.get('raw_sha256') != reference.get('raw_sha256')):
        raise EstimatorInfrastructureError(f'raw source path/fingerprint mismatch: {raw_path}')


def _prepared_path(reference, prepared_root):
    recorded = Path(reference['prepared_path'])
    if prepared_root is None:
        return recorded
    root = Path(prepared_root).resolve()
    if recorded.is_absolute():
        try:
            relative = recorded.relative_to(root)
        except ValueError:
            relative = (Path('preparations') / reference['prepared_identity'] / 'prepared' /
                        reference['run_id'] / (content_identity(reference['record_id']) + '.json'))
            if recorded.parts[-len(relative.parts):] != relative.parts:
                raise ValueError('prepared reference does not match preparation layout')
    else:
        relative = recorded
    selected = (root / relative).resolve()
    if not selected.is_relative_to(root):
        raise ValueError('prepared reference escapes selected root')
    return selected


def _records(manifest, *, prepared_root=None, prepared_only=False, require_structured=False):
    inventory_records = {(run['run_id'], record['record_id']): record
                         for run in manifest['inventory']['runs'] for record in run['records']}
    if set(inventory_records) != {(r['run_id'], r['record_id']) for r in manifest['records']}:
        raise EstimatorInfrastructureError('raw source inventory/manifest records mismatch')
    for reference in manifest['records']:
        record = _load(_prepared_path(reference, prepared_root))
        digest = content_identity({k: v for k, v in record.items() if k != 'prepared_identity'})
        if (record.get('schema') != 'reasoning_prepared_v1'
                or digest != reference['prepared_identity'] or digest != record.get('prepared_identity')
                or record['run_id'] != reference['run_id'] or record['record_id'] != reference['record_id']
                or [t['turn_idx'] for t in record['turns']] != reference['turn_indices']
                or {str(t['turn_idx']): t['input_identity'] for t in record['turns']} != reference['input_identities']):
            raise ValueError('prepared record identity mismatch')
        for turn in record['turns']:
            if require_structured:
                formats.require_structured_turn(turn)
            if content_identity({k: v for k, v in turn.items() if k != 'input_identity'}) != turn['input_identity']:
                raise ValueError('prepared turn identity mismatch')
        _validate_raw_source(record, reference, inventory_records[(record['run_id'], record['record_id'])],
                             prepared_only=prepared_only)
        yield reference, record


def _snapshot(path, model_fragment=None):
    path = Path(path).expanduser().absolute()
    if (not path.is_dir() or path.parent.name != 'snapshots'
            or not re.fullmatch('[0-9a-f]{40}', path.name)
            or (model_fragment is not None and model_fragment not in str(path.parent.parent))):
        raise ValueError(f'pinned local {model_fragment or "model"} snapshot required: {path}')
    index = path / 'model.safetensors.index.json'
    weights = ([path / name for name in set(_load(index)['weight_map'].values())] if index.is_file()
               else list(path.glob('*.safetensors')) + list(path.glob('pytorch_model.bin')))
    if not (path / 'config.json').is_file() or not weights or not all(weight.is_file() for weight in weights):
        raise ValueError(f'local model weights unavailable: {path}')
    return str(path)


def resolve_runtime(runtime, *, tokenizer=None, validate_assets=True):

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import (AttentionConfig, CompilationConfig, ProfilerConfig, StructuredOutputsConfig)
    if runtime.get('schema') != 'reasoning_runtime_v1' or runtime.get('enable_thinking') is not False:
        raise ValueError('runtime requires reasoning_runtime_v1 and enable_thinking=False')
    if type(runtime.get('batch_size')) is not int or runtime['batch_size'] < 1:
        raise ValueError('runtime batch_size must be positive')
    resolved = dict(runtime)
    resolved['topo_num_samples'] = runtime.get('topo_num_samples', 5)
    if type(resolved['topo_num_samples']) is not int or resolved['topo_num_samples'] < 2:
        raise ValueError('topo_num_samples must be an integer >= 2')
    resolved['adapter_revision'] = 'reasoning_contenders_v3_structured_six_clause_refined'
    resolved['phases'] = {}
    llm_defaults = {name: parameter.default for name, parameter in inspect.signature(LLM).parameters.items()
                    if parameter.default is not inspect.Parameter.empty}
    sampling_defaults = {}
    for phase in ('topo', 'cot'):
        supplied = runtime['phases'][phase]['llm_kwargs']
        dtype = supplied.get('dtype')
        allowed_dtypes = ('half', 'bfloat16') if phase == 'topo' else ('bfloat16',)
        if supplied.get('chat_template') is not None:
            raise ValueError('chat_template overrides are unsupported by the canonical renderer')
        if dtype not in allowed_dtypes or supplied.get('seed') != 0:
            raise ValueError(f'{phase} requires dtype in {allowed_dtypes} and engine seed=0')
        if supplied.get('max_model_len', 0) <= 0:
            raise ValueError('explicit positive max_model_len required')
        if validate_assets:
            _snapshot(supplied['model'], None)
            _snapshot(supplied.get('tokenizer', supplied['model']), None)
        kwargs = {**llm_defaults, **supplied}
        kwargs['tokenizer'] = kwargs['tokenizer'] or kwargs['model']

        engine_kwargs = {**kwargs}
        for name in ('chat_template',):
            engine_kwargs.pop(name, None)
        engine_kwargs.setdefault('disable_log_stats', True)
        engine_kwargs['hf_overrides'] = engine_kwargs['hf_overrides'] or {}
        engine_kwargs['offload_params'] = engine_kwargs['offload_params'] or set()
        for name, config_type in (('compilation_config', CompilationConfig),
                                  ('structured_outputs_config', StructuredOutputsConfig),
                                  ('profiler_config', ProfilerConfig), ('attention_config', AttentionConfig)):
            value = engine_kwargs[name]
            if value is None:
                engine_kwargs[name] = config_type()
            elif isinstance(value, dict):
                engine_kwargs[name] = config_type(**value)
            elif name == 'compilation_config' and isinstance(value, int):
                engine_kwargs[name] = config_type(mode=value)
        defaults = _json_value(EngineArgs(**engine_kwargs))
        resolved['phases'][phase] = dict(runtime['phases'][phase], llm_kwargs=_json_value(kwargs), engine_args=defaults,
                                        checkpoint_revision=Path(kwargs['model']).name, dtype=dtype)
        params = SamplingParams(**SAMPLING[phase])
        sampling_defaults[phase] = {name: _json_value(getattr(params, name))
                                   for name in params.__struct_fields__}
    if validate_assets:
        revision = runtime['bert_revision']
        bert_path = runtime.get('bert_snapshot', str(Path.home() / '.cache/huggingface/hub' /
                  'models--bert-base-uncased' / 'snapshots' / revision))
        resolved['bert_snapshot'] = _snapshot(bert_path, 'bert-base-uncased')
        if Path(resolved['bert_snapshot']).name != revision:
            raise ValueError('BERT revision mismatch')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolved['phases']['topo']['llm_kwargs']['tokenizer'],
                                                   local_files_only=True)
    resolved['sampling_defaults'] = sampling_defaults
    resolved['chat_template_hash'] = content_identity(tokenizer.chat_template)
    resolved['versions'] = {name: importlib.metadata.version(name)
                            for name in ('vllm', 'transformers', 'torch', 'numpy', 'scipy', 'networkx')}
    resolved['source_hashes'] = {module: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for module, path in (('topouq', topo.__file__), ('cotuq', cot.__file__), ('runtime', __file__),
                             ('formats', formats.__file__))}
    return resolved, tokenizer


def _context(tokenizer, runtime, phase, job):
    rendered = tokenizer.apply_chat_template(job['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
    if len(rendered) + job['sampling_params']['max_tokens'] > runtime['phases'][phase]['llm_kwargs']['max_model_len']:
        raise ValueError(f"full request exceeds context: {job['job_id']}")
    return len(rendered)


class QwenBatchClient:
    def __init__(self, runtime, phase):
        from vllm import LLM
        self.runtime, self.phase = runtime, phase
        self.llm = LLM(**runtime['phases'][phase]['llm_kwargs'])
        self.tokenizer = self.llm.get_tokenizer()
        if content_identity(self.tokenizer.chat_template) != runtime['chat_template_hash']:
            raise ValueError('loaded chat template differs from preflight')

    def chat_many(self, jobs):
        from vllm import SamplingParams
        counts = [_context(self.tokenizer, self.runtime, self.phase, job) for job in jobs]
        params = [SamplingParams(**job['sampling_params']) for job in jobs]
        outputs = self.llm.chat([job['messages'] for job in jobs], sampling_params=params,
                               chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
        if len(outputs) != len(jobs) or any(len(output.outputs) != 1 for output in outputs):
            raise RuntimeError('inference must return exactly one completion per job')

        processor = getattr(self.llm, 'input_processor', None)
        effective_params = []
        for supplied in params:
            actual = supplied.clone()
            if processor is not None:
                actual.update_from_generation_config(processor.generation_config_fields,
                                                       processor.renderer.get_eos_token_id())
                actual.update_from_tokenizer(self.tokenizer)
            effective_params.append({name: _json_value(getattr(actual, name)) for name in actual.__struct_fields__})
        return {job['job_id']: dict(text=output.outputs[0].text,
                    token_ids=list(output.outputs[0].token_ids), finish_reason=output.outputs[0].finish_reason,
                    request_token_count=count, sampling=actual)
                for job, output, count, actual in zip(jobs, outputs, counts, effective_params)}


def make_bert_embedder(snapshot):
    import torch
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = BertModel.from_pretrained(snapshot, local_files_only=True)
    model.eval()

    def embed(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embed


class _Recorder:
    def chat(self, system, user):
        self.messages = [dict(role='system', content=system), dict(role='user', content=user)]
        return ''


def _topo_messages(stage, prompt, value=None):
    recorder = _Recorder()
    if stage == 'topo_knowledge':
        topo.get_knowledge_points(recorder, prompt, topo.KNOWLEDGE_FORMAT_EXAMPLE)
    elif stage == 'topo_answer':
        subq = f'Given this context:"{prompt}" Only answer this sub question: {value}'
        topo.get_knowledge_answers(recorder, subq, topo.ANSWER_FORMAT_EXAMPLE)
    else:
        topo.get_reasoning_structure(recorder, value, prompt)
        recorder.messages[-1]['content'] += formats.graph_format_clause(len(value))
    return recorder.messages


def _annotation_messages(turn):
    formats.require_structured_turn(turn)
    formatted = cot.format_stored_cot(turn['thought_text'], turn['action_text'])
    return [dict(role='user', content=formats.ANNOTATION_TEMPLATE.replace('<QUESTION>', turn['task_question'])
                 .replace('<RESPONSE>', formatted))]


class _Stages:
    def __init__(self, artifact_dir, identity, reference, turn, runtime, phase, client, invocation_dir):
        self.directory, self.identity, self.reference = artifact_dir, identity, reference
        self.turn, self.runtime, self.phase, self.client = turn, runtime, phase, client
        self.invocation_dir = invocation_dir

    def job(self, stage, messages, *, sample=0, subquestion=None, attempt=0, parents=()):
        key = stage_cache_key(self.reference['record_id'], self.turn['turn_idx'], sample_index=sample,
                stage=stage, subquestion_index=subquestion, attempt_index=attempt, identity=self.identity)
        parent_refs = [dict(stage_key=p['stage_key'], checkpoint_hash=p['checkpoint_hash']) for p in parents]
        rendered = self.client.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        return dict(job_id=key, stage=stage, messages=messages, sampling_params=SAMPLING[self.phase],
                    sample_index=sample, subquestion_index=subquestion, attempt_index=attempt,
                    parents=parent_refs, request_hash=content_identity(rendered))

    def run(self, jobs):
        completed = []
        iterator = iter(jobs)
        while ready := list(islice(iterator, self.runtime['batch_size'])):
            results, batch = {}, []
            for job in ready:
                path = self.directory / 'stages' / (job['job_id'] + '.json')
                if path.exists():
                    results[job['job_id']] = read_stage(self.directory, job['job_id'],
                        expected_identity=self.identity, request_hash=job['request_hash'], parents=job['parents'])
                else:
                    batch.append(job)
            counts = {job['job_id']: _context(self.client.tokenizer, self.runtime, self.phase, job) for job in batch}
            responses = self.client.chat_many(batch) if batch else {}
            for job in batch:
                output = responses.get(job['job_id'])
                if (not isinstance(output, dict) or not isinstance(output.get('text'), str)
                        or output.get('finish_reason') not in ('stop', 'length')):
                    raise RuntimeError(f"incomplete inference response: {job['job_id']}")
                payload = dict(identity=self.identity, invocation_dir=self.invocation_dir,
                    request_hash=job['request_hash'], parents=job['parents'],
                    prepared_input=dict(prepared_path=self.reference['prepared_path'],
                        input_identity=self.turn['input_identity'], turn_idx=self.turn['turn_idx']),
                    stage=job['stage'], sample_index=job['sample_index'], subquestion_index=job['subquestion_index'],
                    attempt_index=job['attempt_index'], request_token_count=counts[job['job_id']],
                    output=output['text'], completion_token_ids=output['token_ids'], finish_reason=output['finish_reason'],
                    sampling=output.get('sampling', self.runtime['sampling_defaults'][self.phase]), enable_thinking=False,
                    dtype=self.runtime['phases'][self.phase]['dtype'])
                write_stage(self.directory, job['job_id'], payload)
                results[job['job_id']] = read_stage(self.directory, job['job_id'], expected_identity=self.identity,
                                                   request_hash=job['request_hash'], parents=job['parents'])
            completed.extend(results[job['job_id']] for job in ready)
        return completed


def _finite_estimate(scalar, extras):
    extras = dict(extras)
    if 'confidence' in extras and not math.isfinite(extras['confidence']):
        extras['confidence_nonfinite'] = str(extras['confidence'])
        extras['confidence'] = None
    for pair in extras.get('pairs', []):
        if not math.isfinite(pair['distance']):
            pair['distance'], pair['reason'] = None, 'source_nonfinite_distance'
    if not math.isfinite(scalar):
        extras.setdefault('reason', 'source_nonfinite_confidence')
        scalar = None
    return dict(scalar=scalar, extras=extras)


def _topology(stages, embed):
    prompt = formats.build_topo_input(stages.turn)
    knowledge = stages.run(stages.job('topo_knowledge', _topo_messages('topo_knowledge', prompt), sample=i)
                           for i in range(stages.runtime['topo_num_samples']))
    answer_jobs = (stages.job('topo_answer', _topo_messages('topo_answer', prompt, question),
                              sample=i, subquestion=j, parents=[parent])
                   for i, parent in enumerate(knowledge)
                   for j, question in enumerate(topo.extract_numbered_questions(parent['output'])))
    answers = stages.run(answer_jobs)
    pairs, graph_parents = [], []
    for i in range(stages.runtime['topo_num_samples']):
        qa, parents = {}, [knowledge[i]]
        for answer in answers:
            if answer['sample_index'] == i:
                topo.add_qa_from_block(qa, answer['output'])
                parents.append(answer)
        pairs.append(qa)
        graph_parents.append(parents)
    graphs = stages.run(stages.job('topo_graph', _topo_messages('topo_graph', prompt, qa),
                                   sample=i, parents=graph_parents[i]) for i, qa in enumerate(pairs))
    rows, embeddings = [], {}
    for i, graph in enumerate(graphs):
        row = formats.graph_row(graph['output'], pairs[i])
        rows.append(row)
        path = stages.directory / 'embeddings' / (graph['stage_key'] + '.json')
        if path.exists():
            cached = _load(path)
            if cached['parent'] != graph['checkpoint_hash'] or cached['identity'] != stages.identity:
                raise ValueError('embedding checkpoint identity mismatch')
        else:
            values = {}
            def save_embedding(text):
                if text not in values:
                    values[text] = np.asarray(embed(text)).tolist()
                return np.asarray(values[text])
            topo.source_embedding_record(row, save_embedding)
            cached = dict(identity=stages.identity, parent=graph['checkpoint_hash'], values=values)
            cached['checkpoint_hash'] = content_identity(cached)
            write_json_atomic(path, cached)
        if content_identity({k: v for k, v in cached.items() if k != 'checkpoint_hash'}) != cached['checkpoint_hash']:
            raise ValueError('embedding checkpoint integrity mismatch')
        embeddings.update(cached['values'])
    score, extras = topo.source_topouq(rows, lambda text: np.asarray(embeddings[text]))
    return {'topouq': _finite_estimate(score, extras)}


def _cot(stages):
    turn = stages.turn
    try:
        formatted = cot.format_stored_cot(turn['thought_text'], turn['action_text'])
        steps = cot.stored_thought_steps(turn['thought_text'])
    except ValueError as error:
        return {name: _finite_estimate(float('nan'), dict(reason=str(error))) for name in ('cot_uq_me', 'cot_uq_sp')}
    failures = []
    for attempt in range(20):
        annotation = stages.run([stages.job('cot_annotation', _annotation_messages(turn), attempt=attempt)])[0]
        try:
            parsed = formats.parse_keywords(annotation['output'])
            if not parsed or 'NO ANSWER' in annotation['output']:
                raise ValueError('source_annotation_invalid')
            probabilities, contributions, mapping = cot.keyword_probability_artifact(steps, parsed[0],
                turn['token_ids'], turn['token_logprobs'], PreparedStepTokenizer(turn['step_tokenization']))
            break
        except ValueError as error:
            failures.append(dict(attempt=attempt, reason=str(error)))
    else:
        extras = dict(reason='cot_annotation_retry_exhausted', annotation_attempts=20, failures=failures)
        return {name: _finite_estimate(float('nan'), extras) for name in ('cot_uq_me', 'cot_uq_sp')}
    common = dict(annotation_attempts=attempt + 1, annotation_stage=annotation['stage_key'],
                  keyword_probabilities=probabilities, keyword_contributions=contributions, mapping_diagnostics=mapping,
                  annotation_failures=failures)
    confidence = cot.mean_confidence(probabilities, contributions)
    result = {'cot_uq_me': _finite_estimate(1 - confidence, {**common, 'confidence': confidence})}
    prompt = cot.keystep_prompt(turn['task_question'], turn['action_text'], formatted, contributions)
    for attempt in range(20):
        probe = stages.run([stages.job('cot_sp', [dict(role='user', content=prompt)], attempt=attempt,
                                      parents=[annotation])])[0]
        confidence = formats.parse_confidence(probe['output'])
        if confidence is not None:
            result['cot_uq_sp'] = _finite_estimate(1 - confidence,
                {**common, 'confidence': confidence, 'sp_attempts': attempt + 1})
            break
    else:
        result['cot_uq_sp'] = _finite_estimate(float('nan'), {**common, 'sp_attempts': 20,
                                                            'reason': 'cot_sp_retry_exhausted'})
    return result


def _identity_manifest(manifest, runtime, artifact_root):
    mathematical = mathematical_runtime(runtime)
    identity = content_identity(dict(prepared_inventory_identity=manifest['prepared_inventory_identity'], runtime=mathematical))
    directories = {r['run_id']: str(Path(artifact_root).absolute() / r['run_id'] / identity) for r in manifest['records']}
    runtime_dir = Path(artifact_root).absolute() / 'runtime' / identity
    invocation_dir = runtime_dir / 'invocations' / uuid.uuid4().hex
    result = dict(schema='reasoning_runtime_identity_v1', identity=identity,
                  prepared_inventory_identity=manifest['prepared_inventory_identity'], artifact_dirs=directories,
                  runtime_identity=content_identity(mathematical), invocation_dir=str(invocation_dir),
                  invocation_runtime_identity=content_identity(runtime))
    write_json_atomic(invocation_dir / 'effective_runtime.json', runtime)
    write_json_atomic(invocation_dir / 'identity_manifest.json', result)
    write_json_atomic(runtime_dir / 'effective_runtime.json', mathematical)
    write_json_atomic(runtime_dir / 'identity_manifest.json', result)
    write_json_atomic(Path(artifact_root) / 'runtime_identity_manifest.json', result)
    return result


def _phase_path(directory, record_id, turn_idx, phase):
    return directory / 'phase_results' / phase / (turn_cache_key(record_id, turn_idx) + '.json')


def _read_phase(path, identity, turn, *, expected_phase):
    try:
        value = _load(path)
    except (OSError, ValueError) as error:
        raise EstimatorInfrastructureError(f'phase checkpoint unreadable: {path}') from error
    if (not isinstance(value, dict) or value.get('schema') != 'reasoning_phase_v1'
            or value.get('phase') != expected_phase
            or value.get('identity') != identity or value.get('input_identity') != turn['input_identity']
            or value.get('checkpoint_hash') != content_identity({k: v for k, v in value.items() if k != 'checkpoint_hash'})):
        raise EstimatorInfrastructureError(f'phase checkpoint schema/identity mismatch: {path}')
    names = ('topouq',) if expected_phase == 'topo' else ('cot_uq_me', 'cot_uq_sp')
    validate_estimates(value.get('estimates'), names, context=path)
    return value


def _publish(directory, record, turn, identity):
    paths = [_phase_path(directory, record['record_id'], turn['turn_idx'], phase) for phase in ('topo', 'cot')]
    if not all(path.exists() for path in paths):
        return False
    estimates = {}
    for phase, path in zip(('topo', 'cot'), paths):
        estimates.update(_read_phase(path, identity, turn, expected_phase=phase)['estimates'])
    params = dict(artifact_dir=str(directory), artifact_identity=identity,
                  reasoning_contender_contract='source_runtime_cache_v1')
    hashes = {key: turn[key] for key in ('input_identity', 'selected_idx', 'selected_candidate_digest',
                                        'model_name', 'tokenizer_name', 'dataset_key')}
    hashes.update(prompt_hash=content_identity(turn['prompt']), initial_prompt_hash=content_identity(turn['initial_prompt']))
    payload = dict(schema='reasoning_contenders_v1', status='complete', identity=identity,
                   record_id=record['record_id'], turn_idx=turn['turn_idx'], input_hashes=hashes,
                   estimates=estimates, params={name: params for name in estimates})
    validate_result(payload, record['record_id'], turn['turn_idx'], expected_identity=identity)
    write_json_atomic(directory / 'results' / (turn_cache_key(record['record_id'], turn['turn_idx']) + '.json'), payload)
    return True


def produce_phase(prepared_root, artifact_root, *, runtime, phase, client=None, embed=None, turn_filter=None,
                  prepared_only=False, worker_index=0, worker_count=1):

    if phase not in ('topo', 'cot'):
        raise ValueError('phase must be topo or cot')
    return _produce(prepared_root, artifact_root, runtime=runtime, phases=(phase,), client=client,
                    embed=embed, turn_filter=turn_filter, prepared_only=prepared_only,
                    worker_index=worker_index, worker_count=worker_count)


def produce_combined(prepared_root, artifact_root, *, runtime, client=None, embed=None, turn_filter=None,
                     prepared_only=False, worker_index=0, worker_count=1, on_ready=None, on_progress=None):

    return _produce(prepared_root, artifact_root, runtime=runtime, phases=('topo', 'cot'), client=client,
                    embed=embed, turn_filter=turn_filter, prepared_only=prepared_only,
                    worker_index=worker_index, worker_count=worker_count,
                    on_ready=on_ready, on_progress=on_progress)


def _produce(prepared_root, artifact_root, *, runtime, phases, client, embed, turn_filter,
             prepared_only, worker_index, worker_count, on_ready=None, on_progress=None):
    started = time.monotonic()
    if (type(worker_count) is not int or worker_count < 1 or type(worker_index) is not int
            or not 0 <= worker_index < worker_count):
        raise ValueError('worker_index must be in [0, worker_count), with positive worker_count')
    manifest = _manifest(prepared_root, prepared_only=prepared_only, require_structured=True)
    if turn_filter is not None:
        available = {(r['run_id'], r['record_id']): set(r['turn_indices']) for r in manifest['records']}
        if not isinstance(turn_filter, dict):
            raise ValueError('turn filter must be a run/record mapping')
        for run_id, records in turn_filter.items():
            if not isinstance(records, dict):
                raise ValueError('turn filter must be a run/record mapping')
            for record_id, indices in records.items():
                if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                        or (run_id, record_id) not in available
                        or not set(indices).issubset(available[(run_id, record_id)])):
                    raise ValueError('turn filter contains unknown original turns')
    effective, tokenizer = resolve_runtime(runtime, tokenizer=getattr(client, 'tokenizer', None),
                                           validate_assets=client is None)
    if len(phases) == 2:
        if (effective['phases']['topo']['llm_kwargs'] != effective['phases']['cot']['llm_kwargs']
                or effective['phases']['topo']['dtype'] != 'bfloat16'):
            raise ValueError('shared phases require equal bfloat16 engine profiles')
    effective['device_diagnostics'] = {**effective.get('device_diagnostics', {}),
                                     'worker_index': worker_index, 'worker_count': worker_count}
    identity_manifest = _identity_manifest(manifest, effective, artifact_root)
    identity = identity_manifest['identity']
    if client is None:
        client = QwenBatchClient(effective, phases[0])
        engine = client.llm.llm_engine.vllm_config
        observation = _json_value(engine)
        for phase in phases:
            write_json_atomic(Path(identity_manifest['invocation_dir']) / f'{phase}_engine_config.json', observation)
            write_json_atomic(Path(artifact_root) / 'runtime' / identity / f'{phase}_engine_config.json', observation)
    if 'topo' in phases and embed is None:
        embed = make_bert_embedder(effective['bert_snapshot'])
    if on_ready is not None:
        on_ready()
    report = {**identity_manifest, 'phase': phases[0] if len(phases) == 1 else 'combined',
              'turns': 0, 'completed_results': 0, 'topology_attempts': 0,
              'phase_terminal_counts': dict.fromkeys(phases, 0),
              'phase_completed_counts': dict.fromkeys(phases, 0),
              'phase_reused_counts': dict.fromkeys(phases, 0)}
    for reference, record in _records(manifest, prepared_root=prepared_root, prepared_only=prepared_only,
                                      require_structured=True):
        directory = Path(identity_manifest['artifact_dirs'][record['run_id']])
        for turn in record['turns']:
            key = content_identity([record['run_id'], record['record_id'], turn['turn_idx']])
            if int(key, 16) % worker_count != worker_index:
                continue
            if turn_filter is not None and turn['turn_idx'] not in turn_filter.get(record['run_id'], {}).get(record['record_id'], []):
                continue
            for phase in phases:
                reused = _produce_turn(directory, identity, reference, turn, effective, phase, client,
                                       embed, identity_manifest['invocation_dir'])
                report['phase_terminal_counts'][phase] += 1
                report['phase_reused_counts' if reused else 'phase_completed_counts'][phase] += 1
                if phase == 'topo':
                    report['topology_attempts'] += effective['topo_num_samples']
                report['elapsed_seconds'] = time.monotonic() - started
                if on_progress is not None:
                    on_progress(report)
            report['turns'] += 1
            report['completed_results'] += int(_publish(directory, record, turn, identity))
            report['elapsed_seconds'] = time.monotonic() - started
            if on_progress is not None:
                on_progress(report)
    report['elapsed_seconds'] = time.monotonic() - started
    return report


def _produce_turn(directory, identity, reference, turn, effective, phase, client, embed, invocation_dir):
    path = _phase_path(directory, reference['record_id'], turn['turn_idx'], phase)
    if path.exists():
        _read_phase(path, identity, turn, expected_phase=phase)
        return True
    stages = _Stages(directory, identity, reference, turn, effective, phase, client, invocation_dir)
    if phase == 'cot' and turn.get('preparation_reason'):
        names = ('cot_uq_me', 'cot_uq_sp')
        estimates = {name: _finite_estimate(float('nan'), dict(reason=turn['preparation_reason'])) for name in names}
    else:
        estimates = _topology(stages, embed) if phase == 'topo' else _cot(stages)
    checkpoint = dict(schema='reasoning_phase_v1', identity=identity, input_identity=turn['input_identity'],
                      phase=phase, estimates=estimates, invocation_dir=invocation_dir)
    checkpoint['checkpoint_hash'] = content_identity(checkpoint)
    write_json_atomic(path, checkpoint)
    return False


def preflight(prepared_root, artifact_root, *, runtime, tokenizer=None, prepared_only=False):

    manifest = _manifest(prepared_root, prepared_only=prepared_only, require_structured=True)
    effective, tokenizer = resolve_runtime(runtime, tokenizer=tokenizer)
    identity_manifest = _identity_manifest(manifest, effective, artifact_root)
    report = {**identity_manifest, 'counts': manifest['totals'], 'selected_span_turns': 0,
              'source_step_turns': 0, 'first_stage_requests': 0, 'longest_full_request': None,
              'first_stage_input_tokens': {'topo': {}, 'cot': {}},
              'local_models': {phase: effective['phases'][phase]['llm_kwargs']['model'] for phase in ('topo', 'cot')}}
    maximum = -1
    for reference, record in _records(manifest, prepared_root=prepared_root, prepared_only=prepared_only,
                                      require_structured=True):
        for turn in record['turns']:
            requests = [('topo', _topo_messages('topo_knowledge', formats.build_topo_input(turn)))]
            if not turn.get('preparation_reason'):
                report['selected_span_turns'] += 1
                try:
                    requests.append(('cot', _annotation_messages(turn)))
                    report['source_step_turns'] += 1
                except ValueError:
                    pass
            for phase, messages in requests:
                job = dict(job_id=f"{record['run_id']}:{record['record_id']}:{turn['turn_idx']}:{phase}",
                           messages=messages, sampling_params=SAMPLING[phase])
                count = _context(tokenizer, effective, phase, job)
                total = count + SAMPLING[phase]['max_tokens']
                report['first_stage_requests'] += 1
                counts = report['first_stage_input_tokens'][phase]
                counts[str(count)] = counts.get(str(count), 0) + 1
                if total > maximum:
                    maximum = total
                    report['longest_full_request'] = dict(job_id=job['job_id'], input_tokens=count,
                        max_tokens=SAMPLING[phase]['max_tokens'], full_tokens=total)
    write_json_atomic(Path(artifact_root) / 'runtime' / identity_manifest['identity'] / 'preflight.json', report)
    write_json_atomic(Path(identity_manifest['invocation_dir']) / 'preflight.json', report)
    write_json_atomic(Path(artifact_root) / 'preflight.json', report)
    return report
