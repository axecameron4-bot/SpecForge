# Qwen3.5-0.8B DFlash layer study

**Active recipe: [SGLANG.md](SGLANG.md) and `sglang-online.yaml`.**
The HF run below is a legacy plumbing pilot trained on dataset answers.
The intended baseline trains on actual target-generated tokens.

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

## Reproduce the initial baseline

Run from the repository root with access to CUDA. Keep the existing Colab
PyTorch stack; the research entry points do not import the SGLang runtime.

```bash
export HF_HOME="$PWD/cache/huggingface"
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir cache/models/Qwen3.5-0.8B
python experiments/qwen35_08b_dflash/prepare_data.py
python -m specforge.research.qwen35_dflash \
  --steps 300 --accumulation 4 --validation-size 64 \
  --run-id baseline-standard-layers
python -m specforge.research.benchmark_qwen35_dflash \
  --checkpoint outputs/qwen35_08b_dflash/baseline-standard-layers/step-300 \
  --run-id baseline-standard-layers
python -m specforge.research.benchmark_qwen35_dflash \
  --trajectories outputs/qwen35_08b_dflash/baseline-standard-layers/gsm8k_predictions.jsonl \
  --run-id untrained-standard-layers
```

Training run IDs cannot overwrite an existing output directory. `--checkpoint`
loads draft weights as a warm start; it does not resume optimizer/data-stream
state. Checkpoints also retain optimizer/scaler/RNG state for later resume work.
For a matched all-layer run, use `--layers all` and a new run ID, holding every
other hyperparameter fixed. Final target hidden state follows Hugging Face's
output convention (final norm is applied to the last layer).

The 300-step initial budget processes 1200 valid examples from the seeded 8192
subset, batch size one with four-way accumulation. Maximum length is 512;
examples with fewer than 17 supervised tokens after truncation are skipped.
Use the same filtering and budget for the all-layer comparison.

Benchmark reports greedy accepted length INCLUDING the clean anchor token
(minimum 1, maximum 16), on sampled full blocks of target completions. It also
reports token accuracy, per-position survival and target GSM8K exact match.
Target-only generation timing is not a speculative serving speed measurement.

Data preparation reuses revisions from the tracked manifest by default.
Use `--refresh-revisions` only when deliberately creating a new data version.

After both benchmark runs, create the final comparison and report:

```bash
python experiments/qwen35_08b_dflash/summarize.py
```

`results/REPORT.md` is the human-readable report. The paired bootstrap resamples
questions (not individual correlated anchors within one question).
