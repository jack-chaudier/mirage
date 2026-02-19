# Narrator: Mirage-Guarded Interactive Fiction

Interactive narrative engine demonstrating validity mirage prevention.
A simulation engine maintains ground truth, an LLM generates prose,
and a guard layer catches pivot violations before the player sees them.

## Quick start

```bash
cd apps/narrator
pip install -e ../../projects/lorien/src/engine
pip install -e .
python -m narrator.server
```

Open http://localhost:8000

## Architecture

The runtime loop is: generate -> validate -> repair -> commit.

1. Generator proposes structured scene events plus prose.
2. Guard checks invariants against simulation-backed world state.
3. Invalid proposals are repaired (or regenerated) before commit.
4. Only committed prose reaches the player.

## Status

MVP scaffold. Currently running with mock scenes (no LLM required).
Set `LLM_MODE=live` and configure `.env` for LLM-backed generation.
