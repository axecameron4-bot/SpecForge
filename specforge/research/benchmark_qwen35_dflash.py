"""Measure DFlash on-policy block acceptance and target GSM8K accuracy.

This is a sampled-anchor acceptance benchmark, not an end-to-end speculative
serving speed benchmark. Hybrid target caches cannot safely use generic crop.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from specforge.research.qwen35_dflash import build_models, forward, read_jsonl
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel


class FullBlockBenchmarkModel(OnlineDFlashModel):
    """Sample only full blocks; preserve supervision of all completion tokens."""

    def _sample_anchor_positions(self, seq_len, loss_mask, device, max_valid_anchors=None):
        valid = loss_mask[:, :max(0, seq_len-self.block_size+1)] > 0.5
        # Every slot of a benchmark block must lie in the completion.
        for offset in range(1, self.block_size):
            valid = valid & (loss_mask[:, offset:offset+valid.shape[1]] > 0.5)
        counts = valid.sum(dim=1)
        width = min(self.num_anchors, int(counts.min()))
        if width == 0:
            raise ValueError('No full completion blocks')
        scores = torch.rand(valid.shape, device=device).masked_fill(~valid, 2.0)
        anchors = scores.argsort(dim=1)[:, :width].sort(dim=1).values
        return anchors, torch.ones_like(anchors, dtype=torch.bool)


def answer_number(text):
    text = text.replace(',', '')
    marked = re.findall(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if marked:
        return marked[-1]
    boxed = re.findall(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if boxed:
        return boxed[-1]
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='cache/models/Qwen3.5-0.8B')
    p.add_argument('--checkpoint', help='Omit for the seeded untrained draft control')
    p.add_argument('--trajectories', help='Reuse a previous benchmark prediction JSONL')
    p.add_argument('--layers', choices=['standard', 'all'], default='standard')
    p.add_argument('--anchors', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--run-id', default='baseline-standard-layers')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    target, tokenizer, draft, wrapper = build_models(a)
    draft.eval()
    wrapper = FullBlockBenchmarkModel(
        draft, target.get_output_embeddings(), target.get_input_embeddings(),
        mask_token_id=248070, block_size=16, attention_backend='sdpa',
        num_anchors=a.anchors, objective_chunk_blocks=1, teacher_metrics=False,
    )
    out = Path('outputs/qwen35_08b_dflash')/a.run_id
    out.mkdir(parents=True, exist_ok=True)
    cached = {r['id']: r for r in read_jsonl(a.trajectories)} if a.trajectories else {}
    totals = {}
    records = []
    with (out/'gsm8k_predictions.jsonl').open('w', buffering=1) as f, torch.no_grad():
        for row in read_jsonl('cache/dataset/qwen35_08b_dflash/gsm8k.jsonl')[:a.size]:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['question']+'\nGive the final answer after ####.'}],
                tokenize=True, enable_thinking=False, add_generation_prompt=True,
                return_tensors='pt', return_dict=True,
            ).to('cuda')
            n_prompt = prompt['input_ids'].shape[1]
            if row['id'] in cached:
                previous = cached[row['id']]
                if previous['prompt_tokens'] != n_prompt:
                    raise ValueError('Cached trajectory prompt does not match')
                ids = torch.tensor([previous['token_ids']], device='cuda')
                elapsed = previous['target_generation_seconds']
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                ids = target.generate(**prompt, max_new_tokens=a.max_new_tokens, do_sample=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
            text = tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)
            mask = torch.zeros_like(ids, dtype=torch.float32)
            mask[:, n_prompt:] = 1
            ratios = {}
            if int(mask.sum()) >= draft.block_size:
                torch.manual_seed(a.seed+len(records))
                _, _, metrics = forward(target, draft, wrapper, ids, mask, detailed=True)
                for key, (num, den) in metrics['ratio_metrics'].items():
                    num, den = float(num), float(den)
                    ratios[key] = num/max(den, 1e-12)
                    pair = totals.setdefault(key, [0.0, 0.0])
                    pair[0] += num
                    pair[1] += den
            record = {'id': row['id'], 'question': row['question'], 'reference': row['answer'],
                      'completion': text, 'predicted_answer': answer_number(text),
                      'correct': answer_number(text) == answer_number(row['answer']),
                      'new_tokens': ids.shape[1]-n_prompt, 'target_generation_seconds': elapsed,
                      'token_ids': ids[0].tolist(), 'prompt_tokens': n_prompt,
                      'hit_length_limit': ids.shape[1]-n_prompt == a.max_new_tokens,
                      'draft_ratios': ratios}
            records.append(record)
            f.write(json.dumps(record)+'\n')
            print(json.dumps({'example': len(records), 'correct': record['correct'],
                              'target_seconds': elapsed,
                              'accepted_length': ratios.get('dflash/hard_label/unary_greedy_accepted_length')}), flush=True)
    summary = {'checkpoint': a.checkpoint, 'reused_trajectories': a.trajectories, 'target_layer_ids': draft.target_layer_ids,
               'benchmark': 'seeded GSM8K test subset; zero-shot greedy; thinking disabled',
               'examples': len(records), 'max_new_tokens': a.max_new_tokens,
               'gsm8k_exact_match': sum(r['correct'] for r in records)/len(records),
               'length_limit_count': sum(r['hit_length_limit'] for r in records),
               'target_only_tokens_per_second': sum(r['new_tokens'] for r in records)/sum(r['target_generation_seconds'] for r in records),
               'draft_ratios': {k: n/max(d, 1e-12) for k, (n,d) in totals.items()},
               'limitations': ['Sampled on-policy anchors, not speculative serving throughput.',
                               'GSM8K score measures frozen target; token limit may truncate answers.',
                               'No end-to-end speedup claimed.']}
    path = Path('experiments/qwen35_08b_dflash/results')/f'{a.run_id}-gsm8k.json'
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
