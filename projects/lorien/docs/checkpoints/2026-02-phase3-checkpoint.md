# Phase 3 Checkpoint Report

**Date:** 2026-02-11
**Scope:** Phases 0-3 (Simulation, Metrics, WorldCanon, Canon Persistence)

## Test Counts

| Suite | Tests | Status |
|-------|-------|--------|
| Engine (Python) | 124 | All pass |
| Visualization (TypeScript) | 68 | All pass |
| **Total** | **192** | **All pass** |

Build: visualization `tsc --noEmit && vite build` passes.

## Key Milestones

### Phase 1: Simulation Engine
- Event-sourced simulation with discrete ticks and seeded RNG
- Pacing physics (dramatic budget, stress, composure, hysteresis)
- Dinner party scenario: 6 agents, 5 locations, ~200 events per run
- 12-seed determinism sweep passing (seeds 1-10, 42, 51)

### Phase 2: Metrics Pipeline
- Tension scoring (8 sub-metrics aggregated into scalar field)
- Dramatic irony tracking via belief matrix
- Scene segmentation (location shifts, participant turnover, tension valleys, irony collapse)
- Significance scoring v0: 4-factor model (delta magnitude 40%, causal centrality 25%, novelty 20%, breadth 15%)

### Phase 3: WorldCanon + Canon Persistence
- **WorldCanon schema** (`schema/canon.py`): location memory, canonical entity containers, claim-state snapshots
- **Claims system**: generalizes secrets into rumors, public facts, propaganda, contested propositions with scoped belief propagation
- **Entity references** (`Event.entities`): typed refs to locations, artifacts, factions, institutions, claims, concepts
- **Canon persistence**: save/load WorldCanon state across simulation runs via `--canon`
- **Visualization forward-compat**: loader accepts `LOCATION_MEMORY`, `ARTIFACT_STATE`, `FACTION_STATE`, `INSTITUTION_STATE` delta kinds; preserves unknown future delta kinds with warnings

### Canon Persistence Demo Results

Script: `src/engine/scripts/demo_canon_persistence.py`

| Run | Seed | Canon | Events | Catastrophes | Conflicts |
|-----|------|-------|--------|--------------|-----------|
| Story A | 42 | -- | 200 | 5 | 7 |
| Story B (fresh) | 51 | -- | 200 | 4 | 9 |
| Story B (canon) | 51 | from A | 200 | 4 | 8 |

Key results:
- **18 belief differences** inherited from Story A into Story B (canon)
- **Location tension residue** carried: dining_table 0.97, foyer 0.52
- **Simulation divergence** from event index 0 (fresh: diana confide, canon: thorne reveal)
- All three runs pass determinism checks

## Determinism Verification

12-seed sweep (seeds 1-10, 42, 51) with `determinism_check.matches=true` on all runs. Canon-loaded runs also deterministic.

## Known Gaps

- **Phase 4 (Extraction + Narration through Canon pipeline):** Lore loop does not yet route canon state into lorebook generation. Prose generation works but does not incorporate canon-inherited world state into narrator context.
- **Claim propagation at scale:** Claims system functional but only tested with 2 example claims (`claim_thorne_health`, `claim_guild_pressure`). No stress test with many concurrent claims.
- **Entity registry completeness:** Entity refs are additive and typed, but the canon entity registry is not yet consumed downstream by metrics or extraction.
