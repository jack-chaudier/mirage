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

## What it demonstrates

In a 14-turn session, the LLM maintained coherent dramatic escalation — from
polite dinner to full confrontation — while the guard caught 6 structural
violations: characters referencing unknown secrets, impossible locations,
causal inconsistencies. All were repaired transparently. The player
experienced a complete story arc without a single narrative contradiction.

See `examples/sample_session.json` for a full playthrough log with every
proposal, rejection, and repair.

## Exporting a session

```bash
# While the server is running:
./examples/export_session.sh <session-id> > examples/sample_session.json
```

The session ID is returned by `POST /api/game/new` and printed in the
browser console.

## Status

Live mode with LLM-backed generation. Set `LLM_MODE=live` and configure
`.env` with API keys. Mock mode (no LLM required) is still available as
the default.
