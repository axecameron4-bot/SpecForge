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
