# MirageBench Task Export (v01)

This folder contains the canonical 12-task export used in release experiments.

## Files

- `miragebench_v01_tasks.json`: full task objects (contexts, metadata, pivots, questions)
- `miragebench_v01_task_index.csv`: compact index for quick inspection

## Provenance

Generated from the canonical runtime path:

- notebook loader: `notebooks/miragebench_experiments_colab.ipynb`
- methodology patch: `_patch_runtime_with_methodology_fixes`

Regenerate with:

```bash
.venv/bin/python scripts/build_release_assets.py
```
