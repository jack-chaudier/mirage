# NarrativeField — Project Context

## What This Is

NarrativeField simulates fictional worlds — agents with desires, secrets, flaws colliding under pressure — and maps the resulting narratives onto an interactive topology of threaded story arcs. Users explore the map, select character paths, and extract structured stories with mathematically grounded tension arcs and turning points.

**Core inversion:** Don't write the story. Simulate the world. Extract the stories that emerge.

## Project Phase

**IMPLEMENTATION.** Audit complete, simulation + storyteller pipeline working. Spec fixes and Phase 1-4 implementation ongoing.

## Key Design Documents

Read these BEFORE producing any output:

- `docs/design_v1.md` — v1: Original vision, mathematical framework, conceptual architecture
- `docs/design_v2.md` — v2: Refined plan incorporating critique. Event sourcing, pacing physics, meaning primitives
- `docs/design_v3.md` — **v3 (CANONICAL)**: Engineering specification. All 17 architectural decisions resolved. Complete schema, pacing physics, belief catalog, scene segmentation. **Always defer to this document.**

## Resolved Architectural Decisions

These are FINAL. Do not re-litigate unless you find a concrete engineering problem.

1. Event-primary graph — events are nodes, world-state is materialized view
2. CQRS read layer — periodic snapshots + index tables
3. Typed deltas — discriminated union (DeltaKind enum)
4. Discrete ticks + event queue — simultaneous action resolution
5. Pacing physics — dramatic budget, stress, composure, hysteresis, recovery
6. Vectorized feature scoring — goal vectors + cosine distance
7. Promises as search targets — not physics; post-hoc evaluation
8. Finite proposition catalog — belief enum matrix per agent × secret
9. JSD over defined future summaries — for counterfactual impact
10. Two-parameter catastrophe — stress × commitment cusp
11. TDA-lite — union-find H₀, cycle heuristics, no Vietoris-Rips
12. Flexible x-axis API — pinned time default
13. 3-tier fake data — story-critical + texture + ambiguity
14. Arc grammars — hard structure + soft scoring
15. Precomputed hover neighborhoods — BFS depth-3, O(1) lookup
16. First-class scene segmentation
17. MVP 1.5 "The Bridge" — two-location stress test

## MVP: "The Dinner Party Protocol"

- 6 agents, 1 evening (~2-3 hrs sim time)
- Locations: dining table, kitchen, balcony, foyer, bathroom (with privacy/adjacency/overhear rules)
- ~100-200 events per run
- Event types: CHAT, OBSERVE, SOCIAL_MOVE, REVEAL, CONFLICT, INTERNAL, PHYSICAL, CONFIDE, LIE, CATASTROPHE

## Build Order

| Phase | Focus | Key Deliverable |
|-------|-------|----------------|
| 1 | Renderer with fake data | Interactive 2D thread visualization |
| 2 | Dinner Party simulation | Agent sim feeding real events to renderer |
| 3 | Metrics pipeline | Tension, irony, thematic shift computation |
| 4 | Story extraction | Arc grammar → beat sheet → LLM prose |
| 5 | Counterfactual impact | Branch sim, JSD scoring |
| 6 | Story queries | NL → path search with grammar constraints |

## Tech Stack

- **Simulation:** Python 3.12+ (dataclasses, type hints, event sourcing)
- **Metrics:** Python, NumPy, SciPy, NetworkX
- **Visualization:** TypeScript, React, D3.js (2D primary), optional Three.js (3D mode)
- **Prose generation:** Claude API from structured beat sheets
- **Data format:** JSON event logs

## Output Conventions

- All specs go in `specs/` subdirectories (schema, simulation, visualization, metrics)
- Include concrete examples (JSON/Python) for every data structure
- Include edge cases and "NOT in scope" sections
- Cross-reference docs/design_v3.md decision numbers where relevant
- Flag dependencies on other specs explicitly

## Workflow Conventions

- Update CHANGELOG.md with every commit using Keep a Changelog format
- Changelog entries go in [Unreleased], moved to dated section on release
- Run tests before committing: `cd src/engine && pytest` and `cd src/visualization && npm test`
- Reference spec sections in PRs

## Audit & Implementation Guides

- `docs/CODEX_WORKFLOW.md` — 47-task implementation DAG across 4 phases
- `specs/audit/audit-synthesis.md` — 35 findings (7 CRITICAL, 11 HIGH)
- `docs/audits/AUDIT_REPORT.md` — Full audit report
- `docs/audits/STORYTELLER_AUDIT_REPORT.md` — Storyteller pipeline audit report
- `CHANGELOG.md` — Project changelog
