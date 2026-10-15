# Qwen3.5-0.8B DFlash layer study

Target: `Qwen/Qwen3.5-0.8B` (frozen). Train a DFlash draft using target
hidden states computed live, without an offline hidden-state dataset.

## Protocol

- Training: seeded subset of `HuggingFaceH4/ultrachat_200k`, `train_sft`.
- Validation: disjoint `test_sft`; retain prompt IDs and dataset revision.
- Benchmark: `openai/gsm8k`, `main`, test split, target greedy completions.
- Baseline: five draft decoder layers, block size 16, default evenly spaced
  target layers, computed by SpecForge `build_target_layer_ids(24, 5)`:
  `[1, 6, 11, 16, 21]` (zero-based decoder outputs, excludes embeddings).
- Follow-up: identical draft architecture and training budget, conditioning on
  all 24 target decoder outputs. Only the input projection grows. Target
  weights remain frozen; “all layers” means all target features are inputs,
  not fine-tuning the target or creating 24 draft decoder layers.
- Seed 42. Save revisions, hyperparameters, environment, step metrics,
  checkpoints, and validation/benchmark summaries.

Distinguish draft next-token accuracy and accepted-prefix length from task
accuracy. A lossless speculative decoder preserves target quality; its main
benefit is acceptance and speed. Teacher-forced acceptance on dataset answers
is a diagnostic, not serving throughput. On-policy acceptance must use target
completions. Never label a short pilot a converged baseline.

## Environment

Colab Tesla T4, 15 GiB. BF16 and Flex Attention kernels are unsuitable here.
The upstream main online runtime requires a separate SGLang/Mooncake stack.
A small research runner will reuse SpecForge's DFlash model, masks and loss,
with live Hugging Face target inference and SDPA/FP16 autocast on one GPU.
This is a separate research entry point, leaving the production runtime alone.

## Files

- `PROGRESS.md`: persistent handover and experiment state.
- `results/`: small tracked manifests, summaries and reports.
- `cache/`: ignored model weights and prepared datasets (workspace root).
- `outputs/`: ignored checkpoints and full step logs (workspace root).

Run commands and measured results are recorded as they become available.
