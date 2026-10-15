"""Single-GPU live-target DFlash research runner (FP16/T4 compatible).

Reuses the production DFlash backbone and loss. Target features are recomputed
on each microbatch. Dataset answers supply hard labels (no rollout regeneration).
"""
import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3_5ForConditionalGeneration

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import (
    DFlashDraftModel, build_target_layer_ids, extract_context_feature,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encode(row, tokenizer, max_length, device="cuda"):
    messages = row['conversations']
    kwargs = dict(tokenize=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, **kwargs)
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if not text.startswith(prefix):
        raise ValueError('Assistant span does not share rendered prompt prefix')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=max_length)
    ids = encoded['input_ids']
    # BPE can merge across the prompt/answer boundary: use character offsets,
    # rather than assuming separately tokenized prompt IDs are a token prefix.
    supervised = [int(start >= len(prefix) and end > start)
                  for start, end in encoded['offset_mapping']]
    if sum(supervised) < 17:
        return None
    tokens = torch.tensor([ids], device=device)
    mask = torch.tensor([supervised], device=device, dtype=torch.float32)
    return tokens, mask


def build_models(args):
    target = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.target, dtype=torch.float16, attn_implementation='sdpa',
    ).to('cuda').eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    tc = target.config.text_config
    layer_ids = (list(range(tc.num_hidden_layers)) if args.layers == 'all' else
                 build_target_layer_ids(tc.num_hidden_layers, 5))
    config = Qwen3Config(
        hidden_size=tc.hidden_size, intermediate_size=tc.intermediate_size,
        num_attention_heads=tc.num_attention_heads, num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim, num_hidden_layers=5, vocab_size=tc.vocab_size,
        max_position_embeddings=tc.max_position_embeddings, rms_norm_eps=tc.rms_norm_eps,
        layer_types=['full_attention']*5, tie_word_embeddings=True,
        rope_theta=10000000.0,
    )
    config.num_target_layers = tc.num_hidden_layers
    config.block_size = 16
    config.dflash_config = {'target_layer_ids': layer_ids, 'mask_token_id': 248070}
    config._attn_implementation = 'sdpa'
    # FP32 master parameters with FP16 autocast; AdamW state stays FP32.
    draft = DFlashDraftModel(config).to('cuda')
    if args.checkpoint:
        draft.load_state_dict(torch.load(Path(args.checkpoint)/'draft.pt', map_location='cuda', weights_only=True))
    wrapper = OnlineDFlashModel(
        draft, target.get_output_embeddings(), target.get_input_embeddings(),
        mask_token_id=248070, attention_backend='sdpa', num_anchors=args.anchors,
        loss_decay_gamma=7, objective_chunk_blocks=1, teacher_metrics=False,
    )
    return target, tokenizer, draft, wrapper


def forward(target, draft, wrapper, tokens, mask, detailed=False):
    with torch.no_grad():
        # Avoid materializing sequence-length x 248320 logits.
        output = target(tokens, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        features = extract_context_feature(output.hidden_states, draft.target_layer_ids)
    with torch.autocast('cuda', dtype=torch.float16):
        return wrapper(tokens, features, mask, collect_detailed_metrics=detailed)


def validate(args, target, tokenizer, draft, wrapper):
    draft.eval()
    torch.manual_seed(args.seed + 1)
    records = []
    with torch.no_grad():
        for row in read_jsonl('cache/dataset/qwen35_08b_dflash/validation.jsonl'):
            encoded = encode(row, tokenizer, args.max_length)
            if encoded is None:
                continue
            loss, acc, _ = forward(target, draft, wrapper, *encoded)
            records.append({'id': row['id'], 'loss': loss.item(), 'accuracy': acc.item()})
            if len(records) >= args.validation_size:
                break
    if not records:
        raise ValueError('No valid validation examples')
    return {'examples': len(records), 'mean_example_loss': sum(r['loss'] for r in records)/len(records),
            'mean_example_token_accuracy': sum(r['accuracy'] for r in records)/len(records), 'records': records}


def save_checkpoint(out, draft, optimizer, scaler, step, args):
    folder = out / f'step-{step}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(draft.state_dict(), folder/'draft.pt')
    draft.config.save_pretrained(folder)
    torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'step': step,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'python_rng': random.getstate(), 'args': vars(args)}, folder/'training_state.pt')
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='cache/models/Qwen3.5-0.8B')
    p.add_argument('--layers', choices=['standard', 'all'], default='standard')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--anchors', type=int, default=16)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=6e-4)
    p.add_argument('--validation-size', type=int, default=32)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run-id', default='baseline-standard-layers')
    p.add_argument('--checkpoint')
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run outside the agent sandbox')
    out = Path('outputs/qwen35_08b_dflash')/a.run_id
    out.mkdir(parents=True, exist_ok=False)
    target, tokenizer, draft, wrapper = build_models(a)
    manifest = dict(vars(a), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    target_layer_ids=draft.target_layer_ids,
                    trainable_parameters=sum(p.numel() for p in draft.parameters()),
                    supervision='UltraChat hard labels; target features inferred live',
                    status='running')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = read_jsonl('cache/dataset/qwen35_08b_dflash/train.jsonl')
    random.shuffle(rows)
    stream = itertools.cycle(rows)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=a.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    processed = skipped = 0
    print(json.dumps(manifest), flush=True)
    with (out/'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(1, a.steps+1):
            draft.train()
            losses, accuracies = [], []
            for micro in range(a.accumulation):
                encoded = None
                for _ in range(len(rows)):
                    row = next(stream)
                    encoded = encode(row, tokenizer, a.max_length)
                    if encoded is not None:
                        break
                    skipped += 1
                if encoded is None:
                    raise ValueError('No valid training examples')
                loss, acc, _ = forward(target, draft, wrapper, *encoded)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                scaler.scale(loss/a.accumulation).backward()
                losses.append(loss.item())
                accuracies.append(acc.item())
                processed += 1
            warmup = max(1, round(a.steps*0.04))
            lr = a.learning_rate * (step/warmup if step <= warmup else
                  0.5*(1+math.cos(math.pi*(step-warmup)/max(1, a.steps-warmup))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            record = {'step': step, 'loss': sum(losses)/len(losses),
                      'accuracy': sum(accuracies)/len(accuracies), 'lr': lr,
                      'grad_norm': float(grad_norm) if torch.isfinite(grad_norm) else None, 'optimizer_skipped': scaler.get_scale() < old_scale,
                      'examples': processed, 'skipped_examples': skipped,
                      'seconds': time.perf_counter()-start,
                      'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9}
            log.write(json.dumps(record)+'\n')
            if step == 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)
            if step % a.save_every == 0 or step == a.steps:
                save_checkpoint(out, draft, optimizer, scaler, step, a)
        validation = validate(a, target, tokenizer, draft, wrapper)
    summary = dict(manifest, status='completed', completed_steps=a.steps,
                   elapsed_seconds=time.perf_counter()-start, validation=validation)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    results = Path('experiments/qwen35_08b_dflash/results')
    (results/f'{a.run_id}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'status': 'completed', 'validation_loss': validation['mean_example_loss'],
                      'validation_accuracy': validation['mean_example_token_accuracy']}), flush=True)


if __name__ == '__main__':
    main()
