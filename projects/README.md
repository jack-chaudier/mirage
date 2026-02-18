# Projects Mirror

This directory contains curated mirrors of the predecessor projects that led to
the validity mirage work:

- **`lorien/`** — NarrativeField, the simulation-first storytelling engine. Multi-agent world simulation with grammar-constrained story extraction. The systematic extraction failures observed here (Paper 0) motivated the formal theory.
- **`rhun/`** — Domain-agnostic framework for studying greedy extraction failure under endogenous phase constraints. Formalizes the absorbing states (Paper 1) and streaming oscillation traps (Paper 2) discovered in NarrativeField.

These mirrors are imported with hygiene filters (no secrets, local virtualenvs, cache folders, or heavy runtime outputs).

## Refresh
Use:

```bash
scripts/sync_external_projects.sh
```

Default source paths:
- `../rhun`
- `../lorien`

Override with env vars if needed:

```bash
RHUN_SRC=/path/to/rhun LORIEN_SRC=/path/to/lorien scripts/sync_external_projects.sh
```
