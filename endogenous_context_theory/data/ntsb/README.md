# NTSB Dataset Assets

Files in this directory:

- `ntsb_k_phase_cleanup_manifest.csv`:
  fixed six-row per-incident manifest (phase relabels and one k adjustment).
- `ntsb_event_graphs.cleaned.json`:
  cleaned benchmark dataset produced by applying the manifest to the source file.

Source file used for this benchmark:

- `ntsb_event_graphs.json` (local source file provided by user; not tracked in this repo)

Regenerate cleaned JSON:

```bash
cd /path/to/mirage
python endogenous_context_theory/scripts/apply_ntsb_manifest.py \
  --input-json /path/to/ntsb_event_graphs.json \
  --manifest-csv endogenous_context_theory/data/ntsb/ntsb_k_phase_cleanup_manifest.csv \
  --output-json endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json
```
