# Training Data Notes

This folder keeps curated datasets used by the public notebooks/scripts:
- `train_balanced.jsonl`
- `valid.jsonl`
- `data_stats.json`

Excluded from git:
- `train.jsonl` (large raw generation dump)

Regenerate data with:
- `endogenous_context_theory/scripts/generate_training_data.py`
- optional balancing via `endogenous_context_theory/scripts/make_balanced_train.py`
