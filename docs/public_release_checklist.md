# Public Release Checklist (arXiv Day)

Use this checklist on the same day as the arXiv upload.

## 1. Final Repo Audit

- Verify `endogenous_context_theory/release/README.md` renders correctly.
- Verify release figures load in GitHub preview.
- Verify all required artifacts exist:
  - MirageBench task export (`release/miragebench_tasks/`)
  - blackbox notebook + CSVs (`release/notebooks/`, `release/results/blackbox_bf16_5model/`)
  - KV notebook + CSVs (`release/notebooks/`, `release/results/kv_cache_eviction_llama31_8b/`)
  - adapter weights (`release/adapters/mirage_aware_v1/`)
- Regenerate and commit `release/SHA256SUMS.txt`.

## 2. Reproducibility Smoke Test

- Create a clean virtual environment.
- Run:

```bash
cd endogenous_context_theory
python3 -m venv .venv_clean
source .venv_clean/bin/activate
pip install -r requirements.txt
python scripts/build_release_assets.py
```

- Confirm script exits successfully and tables/figures regenerate.

## 3. Security/Privacy Sweep

- Confirm no secrets in repo (`.env`, API keys, auth tokens).
- Confirm no private datasets accidentally included.
- Confirm large local scratch artifacts are excluded.

## 4. Paper Alignment

- Ensure section references in the manuscript point to release paths.
- Ensure metric values in manuscript tables match committed CSV outputs.

## 5. Publish

- Push final commit to `main`.
- Switch GitHub repo visibility to **Public**.
- Add topic tags and short repo description.
- Add arXiv link to README after posting.
