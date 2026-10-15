"""Benchmark a running SGLang target-only or DFLASH server on fixed prompts.

Run each mode separately, without concurrent training. Compare output IDs with
--reference to measure exact greedy-output agreement in addition to speed.
"""
import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:30000')
    p.add_argument('--label', required=True)
    p.add_argument('--tokenizer', default='cache/models/Qwen3.5-0.8B')
    p.add_argument('--data', default='cache/dataset/qwen35_08b_dflash/gsm8k.jsonl')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--reference', help='Prediction JSONL from the target-only run')
    a = p.parse_args()
    if a.size < 1 or a.concurrency < 1 or a.max_new_tokens < 1:
        p.error('size, concurrency and max-new-tokens must be positive')
    if Path(a.label).name != a.label:
        p.error('label must be a plain filename component')
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    data = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()][:a.size]
    prompts = [(row, tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['question']+'\nGive the final answer after ####.'}],
        tokenize=True, enable_thinking=False, add_generation_prompt=True,
    )) for row in data]
    reference = {r['id']: r for r in (
        [json.loads(l) for l in Path(a.reference).read_text().splitlines()] if a.reference else [])}
    if not prompts:
        p.error('benchmark dataset is empty')
    server_response = requests.get(a.url.rstrip('/')+'/get_server_info', timeout=30)
    server_response.raise_for_status()
    info = server_response.json()
    args = info.get('server_args', {})
    server_config = {k: args.get(k) for k in [
        'model_path', 'dtype', 'speculative_algorithm', 'speculative_draft_model_path',
        'attention_backend', 'tp_size', 'context_length', 'mem_fraction_static',
    ]}
    server_config['version'] = info.get('version')
    sampling = {'temperature': 0.0, 'max_new_tokens': a.max_new_tokens}
    local = threading.local()

    def request(item):
        row, ids = item
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        start = time.perf_counter()
        response = local.session.post(a.url.rstrip('/')+'/generate', json={
            'input_ids': ids, 'sampling_params': sampling, 'stream': False,
        }, timeout=600)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        output = result.get('output_ids')
        if not isinstance(output, list) or not output or any(type(t) is not int for t in output):
            raise ValueError('Server must return nonempty exact output_ids')
        record = {'id': row['id'], 'input_ids': ids, 'output_ids': output,
                  'sampling_params': sampling, 'concurrency': a.concurrency, 'seconds': time.perf_counter()-start,
                  'meta_info': result.get('meta_info', {})}
        if a.reference:
            previous = reference[row['id']]
            if previous['input_ids'] != ids or previous['sampling_params'] != sampling:
                raise ValueError('Reference prompt or generation budget does not match')
            record['matches_target_output'] = previous['output_ids'] == output
        return record

    # Warm-up is excluded from measurements, including JIT and CUDA graph work.
    request(prompts[0])
    records = []
    out = Path('outputs/qwen35_08b_dflash/sglang-benchmarks')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (out/f'{a.label}.jsonl').open('w', buffering=1) as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for record in pool.map(request, prompts):
                records.append(record)
                f.write(json.dumps(record)+'\n')
                print(json.dumps({'completed': len(records), 'tokens': len(record['output_ids']),
                                  'seconds': record['seconds']}), flush=True)
    elapsed = time.perf_counter()-start
    meta = [r['meta_info'] for r in records]
    verify = sum(m.get('spec_verify_ct', 0) for m in meta)
    proposed = sum(m.get('spec_num_proposed_drafts', 0) for m in meta)
    accepted = sum(m.get('spec_num_correct_drafts', 0) for m in meta)
    tokens = sum(len(r['output_ids']) for r in records)
    summary = {'label': a.label, 'url': a.url, 'server_config': server_config, 'examples': len(records), 'concurrency': a.concurrency,
               'max_new_tokens': a.max_new_tokens, 'wall_seconds': elapsed,
               'generated_tokens': tokens, 'tokens_per_second': tokens/elapsed,
               'latency_median_seconds': float(np.median([r['seconds'] for r in records])),
               'latency_p95_seconds': float(np.quantile([r['seconds'] for r in records], .95)),
               'length_limit_count': sum(len(r['output_ids']) == a.max_new_tokens for r in records),
               'native_spec_accept_length': tokens/verify if verify else None,
               'native_spec_accept_rate': accepted/proposed if proposed else None,
               'exact_target_output_agreement': sum(r['matches_target_output'] for r in records)/len(records) if a.reference else None,
               'note': 'Real SGLang serving timing; run modes separately on the same idle GPU.'}
    if a.reference:
        previous_tokens = sum(len(reference[r['id']]['output_ids']) for r in records)
        previous_seconds = sum(reference[r['id']]['seconds'] for r in records)
        if a.concurrency == 1 and all(reference[r['id']].get('concurrency') == 1 for r in records):
            summary['serial_speedup_vs_target'] = (tokens/elapsed)/(previous_tokens/previous_seconds)
    path = Path('experiments/qwen35_08b_dflash/results')/f'sglang-{a.label}.json'
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
