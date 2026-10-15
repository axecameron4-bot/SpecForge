# Progress / handover

## 2026-10-15 — setup

- Fork: https://github.com/axecameron4-bot/SpecForge
- Base commit: d80daf31 (upstream main).
- Working directory: /content/dflash-research.
- GitHub credential helper reads a temporary token file; secrets never belong
  in repository files, command arguments, or logs.
- GPU is visible only to commands executed outside the sandbox: Tesla T4,
  15360 MiB. Initial sandbox-only GPU check was misleading.
- Installed: torch 2.11.0+cu130, transformers 5.18.0; sglang absent.
  Upstream pyproject pins torch 2.13.0 / transformers 5.12.1 / sglang 0.5.18.
  Do not silently replace the Colab torch stack.
- Downloaded target to cache/models/Qwen3.5-0.8B.
- Model revision: 2fc06364715b967f1860aea9cf38778875588b17.
- Target text config: hidden size 1024, 24 layers, 8 attention heads,
  2 KV heads, head dimension 256, vocabulary 248320; hybrid linear/full attention.
- Standard DFlash layer selection with five draft layers: [1,6,11,16,21].
- Next: prepare reproducible data, validate live target inference, implement
  T4 research runner and run smoke test before baseline training.
- No trained checkpoint or benchmark result exists yet.

## Live-target runner and baseline started

- Added `specforge.research.qwen35_dflash`, using the existing
  `DFlashDraftModel` and `OnlineDFlashModel` loss/masks, SDPA, FP16 autocast,
  FP32 trainable weights and AdamW state. No SGLang install is necessary.
- Target remains frozen; only the 86,522,368-parameter draft is optimized.
- Supports `--layers standard` and `--layers all`; the latter concatenates all
  24 decoder outputs into the learned input projection.
- Training supervision is UltraChat answers, with live target forward passes;
  it is NOT regenerated on-policy training. Benchmark is on-policy.
- Token boundary fix: use full-text character offsets because BPE may merge
  across separately rendered prompt and assistant tokens.
- Initial FP16 scale 65536 overflowed; scale 128 passed the three-step smoke
  with real updates (smoke-standard-v3). Validation loss after this smoke:
  9.60488; sample too small for a quality conclusion.
- Data: 8192 train / 256 validation / 64 GSM8K, seed 42; pinned revisions and
  hashes in results/data_manifest.json. Train/validation IDs do not overlap.
- Baseline command:
  `HF_HOME=$PWD/cache/huggingface python -m specforge.research.qwen35_dflash --steps 300 --accumulation 4 --validation-size 64 --run-id baseline-standard-layers`
- Run is in progress. 300 steps / 1200 valid examples is an initial budget,
  not an assertion of convergence. Checkpoints every 100 optimizer attempts;
  logs explicitly record any FP16-skipped updates.
- Test: all six selected layers in a tiny model receive nonzero projection
  gradients; target head and embeddings remain frozen. Real Qwen3.5 tokenizer
  boundary / truncated-prompt test passed.
- Added `specforge.research.benchmark_qwen35_dflash`: GSM8K greedy target task
  score plus DFlash acceptance on sampled full blocks from target trajectories.
  Generic speculative cache crop is unsafe for this hybrid model, so no
  end-to-end serving speedup is claimed. Full predictions stay in outputs/.

## Baseline training complete; benchmark running

- All 300/300 optimizer updates applied; no FP16 overflow skips.
- 1200 valid training examples; 126 examples filtered after token truncation.
- Training time: 376.91 s; peak allocated GPU memory: 3.546 GB.
- Validation: 64 valid held-out conversations, mean example loss 7.17156,
  mean example token accuracy 0.0631164. These are teacher-forced metrics.
- Checkpoints: outputs/qwen35_08b_dflash/baseline-standard-layers/step-{100,200,300}.
- Tracked: complete step metrics, summary, environment, and training curve.
- GSM8K benchmark launched on final checkpoint, 64 seeded test examples,
  max 256 new tokens, non-thinking greedy target generation. Await completion
  before reporting acceptance or task accuracy.
- Then run the untrained-draft control on the identical cached trajectories.

## Benchmark interpretation discovered during run

- At 16/64 GSM8K examples, 12 target completions reached the 256-token cap.
  The target task score is therefore a capped-generation diagnostic, not a
  complete GSM8K quality estimate. Keep length-limit counts in the final report.
- Acceptance still compares full 16-token draft blocks with actual generated
  target continuations. This remains the primary draft-quality measurement.
- At this early budget, acceptance is close to the anchor-only floor; do not
  claim meaningful acceleration or convergence from falling training loss.
