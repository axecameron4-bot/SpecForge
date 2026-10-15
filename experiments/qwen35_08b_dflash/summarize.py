"""Create the final paired acceptance comparison and human-readable report."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('experiments/qwen35_08b_dflash/results')
OUTPUTS = Path('outputs/qwen35_08b_dflash')
KEY = 'dflash/hard_label/unary_greedy_accepted_length'


def load(path):
    return json.loads(path.read_text())


def records(run_id):
    return [json.loads(l) for l in (OUTPUTS/run_id/'gsm8k_predictions.jsonl').read_text().splitlines()]


def main():
    trained = load(ROOT/'baseline-standard-layers-gsm8k.json')
    untrained = load(ROOT/'untrained-standard-layers-gsm8k.json')
    validation = load(ROOT/'baseline-standard-layers.json')['validation']
    training = load(ROOT/'baseline-training-statistics.json')
    a, b = records('baseline-standard-layers'), records('untrained-standard-layers')
    if len(a) != len(b):
        raise ValueError('Control and trained runs have different sizes')
    differences = []
    for left, right in zip(a, b):
        if left['id'] != right['id'] or left['token_ids'] != right['token_ids']:
            raise ValueError('Unpaired target trajectories')
        if KEY in left['draft_ratios'] and KEY in right['draft_ratios']:
            differences.append(left['draft_ratios'][KEY]-right['draft_ratios'][KEY])
    rng = np.random.default_rng(42)
    values = np.array(differences)
    bootstrap = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    comparison = {'paired_questions': len(values), 'trained_accepted_length': trained['draft_ratios'][KEY],
                  'untrained_accepted_length': untrained['draft_ratios'][KEY],
                  'mean_question_paired_difference': float(values.mean()),
                  'paired_question_bootstrap_95_ci': np.quantile(bootstrap, [.025, .975]).tolist(),
                  'definition': 'Includes clean anchor; 1 is anchor-only, 16 is a full block.',
                  'not_measured': 'End-to-end speculative serving speedup'}
    (ROOT/'acceptance-comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    ci = comparison['paired_question_bootstrap_95_ci']
    report = f'''# Initial Qwen3.5-0.8B DFlash baseline

Frozen target: Qwen/Qwen3.5-0.8B. Five-layer DFlash draft (86,522,368
parameters), block size 16, standard target decoder layers [1,6,11,16,21].

## Training

UltraChat train_sft: {training['valid_examples']} valid examples from a seeded
8192-example subset, 300 optimizer steps, accumulation 4, length 512.
Target hidden states were computed live on every microbatch; labels were
UltraChat assistant answers, not regenerated target answers. Frozen target,
FP32 draft parameters with FP16 autocast, SDPA on Tesla T4.

- Successful optimizer updates: {training['successful_optimizer_updates']}/300.
- Training time: {training['training_seconds']:.2f} s.
- Peak allocated GPU memory: {training['peak_gpu_gb']:.3f} GB.
- Filtered examples (insufficient assistant tokens after truncation): {training['filtered_examples']}.
- Held-out teacher-forced validation: {validation['examples']} conversations,
  mean example loss {validation['mean_example_loss']:.5f},
  mean example token accuracy {validation['mean_example_token_accuracy']:.2%}.

![Training curve](baseline-training.png)

## Draft benchmark

{trained['examples']} seeded GSM8K test prompts; greedy target completions,
thinking disabled, maximum {trained['max_new_tokens']} new tokens.
Sample 16 full DFlash blocks per eligible target trajectory, using identical
trajectories and anchor RNG seeds for the trained and untrained drafts.

| Measurement | Untrained draft | 300-step draft |
| --- | ---: | ---: |
| Mean accepted length, including anchor | {untrained['draft_ratios'][KEY]:.6f} | {trained['draft_ratios'][KEY]:.6f} |
| Draft token accuracy on target continuations | {untrained['draft_ratios']['acc']:.2%} | {trained['draft_ratios']['acc']:.2%} |

Paired per-question mean acceptance improvement:
{comparison['mean_question_paired_difference']:.6f} tokens,
95% bootstrap interval [{ci[0]:.6f}, {ci[1]:.6f}] (seed 42, 10000 resamples).
Length 1 is the anchor-only floor; length 16 is a fully accepted block.

The frozen target's capped GSM8K exact match was {trained['gsm8k_exact_match']:.2%}.
{trained['length_limit_count']}/{trained['examples']} outputs reached the token
limit. This is a capped diagnostic, not a complete GSM8K task-quality estimate.

## Interpretation and next experiment

This short run establishes an initial, reproducible baseline. Acceptance near
the anchor-only floor is weak; falling loss does not establish useful decoding
acceleration or convergence. End-to-end speculative serving speed was not
measured. Hybrid Qwen3.5 caches require a compatible serving implementation;
generic KV cache cropping is unsafe.

Before drawing layer-importance conclusions, train to a defensible validation
plateau, then repeat with all 24 target decoder features under the same budget,
data, seed, draft depth and benchmark. The all-layer path passed a one-update
GPU smoke test (106,445,312 trainable parameters); it has not completed a matched
training experiment. Importance should be measured using paired layer ablations,
not inferred solely from projection weight norms.

## Artifacts

- Final weights and optimizer/RNG state:
  `outputs/qwen35_08b_dflash/baseline-standard-layers/step-300/`.
- Full target predictions and token trajectories:
  `outputs/qwen35_08b_dflash/baseline-standard-layers/gsm8k_predictions.jsonl`.
- Git tracks this report, metrics, manifests, summaries and the plot. Large model
  weights and full generated texts remain local under ignored cache/outputs.
- Reproduction commands: ../README.md. Persistent handover: ../PROGRESS.md.
'''
    (ROOT/'REPORT.md').write_text(report)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
