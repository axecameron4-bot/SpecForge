# Target-generated SGLang baseline

The earlier HF pilot trained on dataset assistant answers. It is retained only
as a plumbing pilot, not the intended target-generated baseline.

## Training data flow

1. Read only system/user messages before the first dataset assistant turn.
2. Apply the target tokenizer's generation template (thinking disabled).
3. Ask SGLang `/generate` for actual target `output_ids`, in batches.
4. Concatenate the original prompt IDs and the returned IDs exactly; supervise
   generated tokens only. Dataset assistant answers never enter these requests.
5. Send the complete sequence to the patched server for a full prefill, with
   `max_new_tokens=0`. The existing capture patch exports prefill features.
6. Transfer hidden states and token/mask tensors through Mooncake to the draft
   trainer. The trainer never loads or runs the target model.

The two-request approach avoids an invasive change to SGLang's decode-state
capture and preserves the existing producer/consumer ownership and cleanup.
Generation requests force a minimum of two new tokens, matching DFlash's
minimum supervised-token requirement. Generation caps depend on available
sequence budget; no produced sequence exceeds `data.max_length`.

## Hardware gate

Current Colab accelerator: Tesla T4, SM75. Downloaded SGLang 0.5.18 and its
required sglang-kernel 0.4.6.post1 for inspection; the existing Python environment
has not been replaced. `cuobjdump` on `sgl_kernel/sm90/common_ops.abi3.so` found
SM80, SM89, SM90, SM90a, SM100, SM120a binaries, **no SM75 and no PTX fallback**.
Qwen3.5's CUDA Gemma RMSNorm path calls this library. The standard package is
therefore not runnable on T4 for this model without porting kernels.

- Evidence: results/sglang-kernel-architecture-check.json.
- Upstream report: https://github.com/sgl-project/sglang/issues/24302.
- Use a supported GPU, e.g. L4 or A100, or an external SGLang server on one.
- SGLang 0.5.18 requires torch 2.13.0 / transformers 5.12.1. Install in a
  separate environment; do not overwrite the working Colab environment.
- Do not claim server or GPU smoke validation until a supported device is used.

## Recipe

`sglang-online.yaml` is validated against the SpecForge config schema:
10,000 optimizer steps; microbatch 1, accumulation 8 (effective batch 8),
sequence length 2048, up to 1536 generated tokens, 64 anchors, block size 16,
standard target layers [1,6,11,16,21], checkpoint every 250 steps.
Bound producer backlog to 16 refs and 512 MiB. These are initial memory limits;
profile a short run on the actual supported GPU before the full experiment.

Install the version-pinned SpecForge/SGLang dependencies and Mooncake in a
separate environment on supported hardware, then apply the existing patch:

```bash
scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
```

Launch the Mooncake master (in its own terminal):

```bash
mooncake_master --enable_http_metadata_server=true \
  --rpc_port=35551 --http_metadata_server_port=35880 \
  --metrics_port=35903 --enable_metric_reporting=false \
  --default_kv_lease_ttl=500
```

Launch capture/generation on the supported GPU (same GPU as the trainer is
permitted for this external topology; the memory split must be profiled):

```bash
export HF_HOME="$PWD/cache/huggingface"
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=2147483648
python -m sglang.launch_server \
  --model-path cache/models/Qwen3.5-0.8B \
  --dtype bfloat16 --skip-tokenizer-init \
  --tp-size 1 --context-length 4096 --mem-fraction-static 0.35 \
  --max-running-requests 8 --chunked-prefill-size -1 \
  --disable-radix-cache \
  --enable-spec-capture --spec-capture-method dflash \
  --spec-capture-aux-layer-ids 1 6 11 16 21 \
  --host 127.0.0.1 --port 30000
```

The token-only server returns exact output token IDs and accepts token IDs;
there is no dependency on decoding/re-encoding generated text. After a real
server smoke gate, run the trainer in another terminal:

```bash
python -m specforge.cli train -c experiments/qwen35_08b_dflash/sglang-online.yaml
```

For a smoke run, override `training.max_steps=3`, `training.accumulation_steps=1`,
`training.num_epochs=1`, and both `run_id` and `output_dir` to fresh values.
Never reuse an old online control directory for a new producer attempt.

## Validation status

Tests exercise generation plus the real capture adapter, feature specs and
Mooncake transport using an in-memory server/store test harness. They verify
exact output IDs, generated-only loss masks, feature/token length alignment,
answer exclusion, config validation and malformed-response rejection.
Real SGLang GPU integration and throughput are pending a compatible accelerator.
A serving benchmark must measure baseline target and DFLASH decoding separately
on fixed prompts, reporting latency, tokens/s, native acceptance counters and
output equality. The earlier HF sampled-anchor benchmark is not serving speed.

## Greedy response cache

The long recipe caches target-generated token IDs on disk, keyed by prompt IDs,
sampling parameters, target identity and an explicitly pinned model namespace.
Repeated epochs reuse these answers but still perform fresh SGLang hidden-state
capture. Caching is accepted only for temperature zero and a nonempty namespace.
Change the namespace when model weights, tokenizer/template or generation
semantics change; prompt IDs and sampling budgets already participate in the key.
Cache hits are recorded in SampleRef provenance. Dataset answers remain excluded.

## Real serving benchmark client

`bench_sglang.py` talks to a running SGLang server and measures actual latency,
tokens/s and native speculative acceptance counters. Use identical GPU/server
settings and stop training while benchmarking. Run a plain target server first:

```bash
python experiments/qwen35_08b_dflash/bench_sglang.py --label target-only
```

Then restart the server with the exported trained DFLASH draft and run:

```bash
python experiments/qwen35_08b_dflash/bench_sglang.py --label dflash \
  --reference outputs/qwen35_08b_dflash/sglang-benchmarks/target-only.jsonl
```

Both use 64 seeded GSM8K prompts, max 1024 new tokens, thinking disabled,
concurrency one by default. The client validates prompt/sampling agreement and
compares exact output token IDs. Full predictions remain in outputs/; small
summaries go into results/. Native server integration is not GPU-validated yet.
