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

## All-layer path verified on GPU

- Command: `python -m specforge.research.qwen35_dflash --layers all --steps 1 --accumulation 1 --validation-size 2 --run-id smoke-all-layers`.
- Captured all layer IDs 0..23; 106,445,312 trainable draft parameters.
- Real optimizer update applied with finite gradient norm 54.33; peak allocated
  memory 3.882 GB. This is a smoke check, NOT a trained all-layer comparison.
- Added `summarize.py` for the final paired acceptance comparison (bootstrap
  over questions), using identical target trajectories and anchor seeds.
- Three research tests and the existing eager/SDPA mask parity test pass.

## Initial baseline and paired benchmark finished

- 64/64 GSM8K trajectories and the untrained control completed.
- Mean anchor-inclusive accepted length: untrained 1.0, trained 1.1328125.
- Paired question-bootstrap improvement CI: [0.1123047, 0.1533203] tokens.
- On-policy draft token accuracy: 0.0546875. Weak acceptance; no speedup claim.
- Frozen target capped exact match: 25/64; 40/64 completions hit 256 tokens.
- Final report: results/REPORT.md; paired results: results/acceptance-comparison.json.
- User requests a stronger, longer baseline next. Before launching, specify
  training budget and improve the recipe; current 300-step run is only a pilot.
- Current training dataset is UltraChat hard labels with live target features,
  not target-generated on-policy training. Distinguish these explicitly when
  choosing the stronger recipe. All processes from the pilot have finished.

## 2026-10-15 — corrected target-generated SGLang pipeline

User clarified that training labels must be actual target responses. The HF
pilot did not satisfy that requirement and is now explicitly labeled legacy.

- Added optional `data.target_generation` to the native SpecForge config.
- Raw prompt preparation tokenizes only system/user turns before the first
  assistant turn; dataset assistant answers are excluded completely.
- The native producer wraps SGLang capture with batched target generation.
  It uses exact output_ids, constructs generated-only supervision, then captures
  the complete prompt+response sequence through the existing prefill sink.
- Greedy response cache: pinned model namespace + target identity + exact prompt
  IDs + sampling params. Reuses target answers across epochs; hidden states
  still captured live. Cache hits travel in SampleRef provenance.
- Added configs/qwen3.5-0.8b-dflash.json and recipe
  `experiments/qwen35_08b_dflash/sglang-online.yaml`: 10k steps, effective batch8,
  max length2048, generation cap1536, anchors64, save every250, bounded backlog.
- Added SGLANG.md and bench_sglang.py (actual SGLang HTTP serving benchmark with
  native acceptance counters and exact target-output comparison).
- Downloaded SGLang0.5.18 / sglang-kernel0.4.6.post1 wheels for compatibility
  inspection only. No replacement of the installed Python/PyTorch stack.
- Existing spec-capture patch applied successfully to the unpacked0.5.18 wheel
  under cache/sglang-package (no live server execution).
- CUDA gate: cuobjdump confirms common_ops has no SM75 binary and no PTX.
  Current GPU Tesla T4/SM75 cannot run this Qwen3.5 CUDA path unmodified.
  Evidence: results/sglang-kernel-architecture-check.json; upstream issue24302.
- User was asked whether L4/A100 is available; no answer received yet.
- New tests verify exact generation->capture transport, loss masks, output
  lengths, exclusion of poisoned dataset answers, malformed responses,
  explicit config validation, and cache reuse with fresh capture.
- Real server generation/capture, training smoke, serving benchmark and long
  run are pending compatible GPU. No SGLang speedup or training result claimed.
