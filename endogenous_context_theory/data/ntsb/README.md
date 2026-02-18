# NTSB Dataset Assets

Files in this directory:

- `ntsb_k_phase_cleanup_manifest.csv`:
  fixed six-row per-incident manifest (phase relabels and one k adjustment).
- `ntsb_event_graphs.cleaned.json`:
  cleaned benchmark dataset produced by applying the manifest to the source file.

Source file used for this benchmark:

- `/Users/jackg/Downloads/ntsb_event_graphs.json`

Regenerate cleaned JSON:

```bash
/Users/jackg/mirage/.venv/bin/python /Users/jackg/mirage/endogenous_context_theory/scripts/apply_ntsb_manifest.py \
  --input-json /Users/jackg/Downloads/ntsb_event_graphs.json \
  --manifest-csv /Users/jackg/mirage/endogenous_context_theory/data/ntsb/ntsb_k_phase_cleanup_manifest.csv \
  --output-json /Users/jackg/mirage/endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json
```
