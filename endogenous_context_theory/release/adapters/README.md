# Mirage-Aware Adapters

This directory contains released LoRA adapter weights used in mirage-aware training experiments.

## Included

- `mirage_aware_v1/adapter_config.json`
- `mirage_aware_v1/adapters.safetensors`

## Base Model Compatibility

- Base model family: Qwen 2.5 Instruct
- Intended usage: mirage-aware structured output protocol (`PIVOT_ID=...`, evidence assessment, prerequisite reporting)

## Loading

Use PEFT-compatible loading in your preferred runtime. Keep tokenizer/model IDs aligned with the base run metadata in the Colab notebooks under `release/notebooks/`.
