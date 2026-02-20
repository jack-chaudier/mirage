# Mirage-Aware Adapters

This directory contains released LoRA adapter weights used in mirage-aware training experiments.

## Included

- `mirage_aware_v1/adapter_config.json`
- `mirage_aware_v1/adapters.safetensors`

## Base Model Compatibility

- Base model family (released adapter): `mlx-community/gemma-2-2b-it-4bit`
- Canonical config: `mirage_aware_v1/adapter_config.json` (`rank=8`, `iters=600`)
- Intended usage: mirage-aware structured output protocol (`PIVOT_ID=...`, evidence assessment, prerequisite reporting)

## Loading

Use MLX loading for the released adapter, for example:

```bash
cd endogenous_context_theory
.venv/bin/python -m mlx_lm.generate \
  --model mlx-community/gemma-2-2b-it-4bit \
  --adapter-path release/adapters/mirage_aware_v1 \
  --prompt "PIVOT_ID=..."
```

The canonical Qwen/PEFT package used for the README mitigation table is at repo root as `mirage_aware_package.tar.gz` (extracts `mirage_aware_adapter_balanced/`, base `Qwen/Qwen2.5-7B-Instruct`, `r=8`). It is a separate lineage from `mirage_aware_v1`.
