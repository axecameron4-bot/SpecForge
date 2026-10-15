"""Generate with SGLang, then capture the exact generated token sequence.

Generation and capture are separate batched requests because the current
SpecForge capture sink exports prefill features, not decoding features.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from dataclasses import replace
from numbers import Integral

from .server_capture import ServerCaptureFailure, _flatten_list_wrappers


class SGLangTargetGenerationAdapter:
    def __init__(self, capture_adapter, *, max_new_tokens=768, temperature=0.0, top_p=1.0,
                 response_cache_dir=None, cache_namespace=None):
        self.capture_adapter = capture_adapter
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        if response_cache_dir and (not cache_namespace or temperature != 0):
            raise ValueError('Response caching requires pinned namespace and greedy generation')
        self.response_cache_dir = Path(response_cache_dir) if response_cache_dir else None
        self.cache_namespace = cache_namespace
        if self.response_cache_dir:
            self.response_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, task, params):
        if self.response_cache_dir is None:
            return None
        identity = {'namespace': self.cache_namespace,
                    'target': self.capture_adapter.target_model_version,
                    'input_ids': task.payload['input_ids'], 'sampling_params': params}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.response_cache_dir / f'{key}.json'

    def _generate(self, tasks, params):
        paths = [self._cache_path(t, p) for t, p in zip(tasks, params)]
        rows = [json.loads(p.read_text()) if p and p.exists() else None for p in paths]
        hits = [row is not None for row in rows]
        missing = [i for i, row in enumerate(rows) if row is None]
        if missing:
            adapter = self.capture_adapter
            generated = _flatten_list_wrappers(adapter.post_fn(
                f'{adapter.base_url}/generate',
                json_body={'input_ids': [list(tasks[i].payload['input_ids']) for i in missing],
                           'sampling_params': [params[i] for i in missing], 'stream': False},
                timeout=adapter.timeout_s,
            ))
            if len(generated) != len(missing):
                raise ValueError('SGLang generation response count does not match tasks')
            for index, row in zip(missing, generated):
                rows[index] = row
        return rows, paths, hits


    def health(self):
        return dict(self.capture_adapter.health(), target_generation=True)

    def produce_refs(self, tasks, *, capture):
        if not tasks:
            return []
        for task in tasks:
            if not task.metadata.get('target_generation_prompt'):
                raise ValueError('Target generation requires explicitly prepared prompt-only tasks')
            if any(task.payload['loss_mask']):
                raise ValueError('Dataset answer supervision must not enter target generation')
        params = []
        for task in tasks:
            remaining = task.max_length - len(task.payload['input_ids'])
            if remaining < 2:
                raise ValueError('Prompt leaves fewer than two generated-token slots')
            params.append({'temperature': self.temperature, 'top_p': self.top_p,
                           'max_new_tokens': min(self.max_new_tokens, remaining),
                           'min_new_tokens': 2})
        adapter = self.capture_adapter
        rows, cache_paths, cache_hits = self._generate(tasks, params)
        generated_tasks, failures, positions = [], {}, []
        for index, (task, row) in enumerate(zip(tasks, rows)):
            if not isinstance(row, dict):
                raise ValueError('SGLang generation returned a non-object row')
            output = row.get('output_ids')
            if not isinstance(output, list) or any(
                isinstance(t, bool) or not isinstance(t, Integral) or t < 0 for t in output
            ):
                raise ValueError('SGLang must return exact output_ids; no text re-tokenization fallback')
            if len(output) > params[index]['max_new_tokens']:
                raise ValueError('SGLang returned more generated tokens than requested')
            if len(output) < 2:
                failures[index] = ServerCaptureFailure(task.task_id, 'Target generated fewer than two tokens', retryable=False)
                continue
            if cache_paths[index] is not None and not cache_hits[index]:
                path = cache_paths[index]
                temporary = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
                try:
                    temporary.write_text(json.dumps(row))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            prompt_ids = list(task.payload['input_ids'])
            ids = prompt_ids + output
            metadata = dict(task.metadata, num_tokens=len(ids), prompt_tokens=len(prompt_ids),
                            generated_tokens=len(output), target_generated=True,
                            response_cache_hit=cache_hits[index],
                            generation_finish_reason=row.get('meta_info', {}).get('finish_reason'))
            generated_tasks.append(replace(task, payload={
                'input_ids': ids, 'loss_mask': [0]*len(prompt_ids)+[1]*len(output),
            }, metadata=metadata))
            positions.append(index)
        captured = adapter.produce_refs(generated_tasks, capture=capture) if generated_tasks else []
        if len(captured) != len(generated_tasks):
            raise ValueError('Capture response count does not match generated tasks')
        out = [failures.get(i) for i in range(len(tasks))]
        for position, task, ref in zip(positions, generated_tasks, captured):
            if not isinstance(ref, ServerCaptureFailure):
                ref = replace(ref, metadata=dict(ref.metadata, target_generated=True,
                    prompt_tokens=task.metadata['prompt_tokens'],
                    generated_tokens=task.metadata['generated_tokens'],
                    generation_finish_reason=task.metadata['generation_finish_reason'],
                    generation_sampling=params[position],
                    response_cache_hit=task.metadata['response_cache_hit']))
            out[position] = ref
        return out
