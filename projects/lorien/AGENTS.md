# NarrativeField

## Project Overview
NarrativeField simulates fictional worlds and extracts emergent stories. The MVP is "Dinner Party Protocol" — 6 agents, 1 evening, ~100-200 events, 5 locations.

## Architecture
- **Visualization** (Phase 1): TypeScript + React + Zustand + HTML Canvas + D3.js
- **Simulation** (Phase 2): Python 3.12+, dataclasses, Claude API
- **Metrics** (Phase 3): Python 3.12+, NumPy
- **Story Extraction** (Phase 4): Python 3.12+, Claude API

## Monorepo Structure
```
narrativefield/
├── AGENTS.md                    # This file
├── docs/                        # Design + architecture docs
│   ├── ARCHITECTURE.md
│   ├── CODEX_WORKFLOW.md
│   ├── design_v1.md
│   ├── design_v2.md
│   ├── design_v3.md
│   └── audits/
├── specs/                       # All 17 specification documents (READ-ONLY)
│   ├── MASTER_PLAN.md
│   ├── schema/                  # events.md, agents.md, world.md, scenes.md
│   ├── simulation/              # tick-loop.md, decision-engine.md, pacing-physics.md, dinner-party-config.md
│   ├── visualization/           # renderer-architecture.md, interaction-model.md, thread-layout.md, fake-data-visual-spec.md
│   ├── metrics/                 # tension-pipeline.md, irony-and-beliefs.md, scene-segmentation.md, story-extraction.md
│   └── integration/             # data-flow.md
├── examples/                    # Golden generated stories + per-run meta.json
├── src/
│   ├── visualization/           # React + TypeScript (Phase 1)
│   │   ├── AGENTS.md
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── vite.config.ts
│   │   └── src/
│   │       ├── App.tsx
│   │       ├── main.tsx
│   │       ├── types/           # TypeScript interfaces from specs
│   │       ├── store/           # Zustand store
│   │       ├── data/            # Loaders, parsers, fake data JSON
│   │       ├── layout/          # Thread layout spring-force algorithm
│   │       ├── canvas/          # 4-layer Canvas renderer + hit detection
│   │       ├── components/      # React UI (ControlPanel, InfoPanel, TimelineBar, BeatSheetPanel)
│   │       ├── constants/       # Colors, sizes, layout parameters
│   │       └── utils/           # Helpers
│   └── engine/                  # Python (Phases 2-4)
│       ├── AGENTS.md
│       ├── pyproject.toml
│       └── narrativefield/
│           ├── __init__.py
│           ├── schema/          # Python dataclasses from spec schema/
│           ├── simulation/      # Tick loop, decision engine, pacing
│           ├── metrics/         # Tension, irony, segmentation
│           ├── extraction/      # Arc grammar, beat sheets, LLM prose
│           ├── integration/     # Bundler, validators, file I/O
│           └── tests/
├── data/
│   ├── fake-dinner-party.nf-viz.json   # 70-event fake dataset (Phase 1)
│   └── scenarios/
│       └── dinner_party.json           # Scenario config
└── .codex/
    └── skills/
```

## Build & Test Commands
- **Visualization:** `cd src/visualization && npm install && npm run dev` (dev), `npm run build` (prod), `npm test` (tests)
- **Engine:** `cd src/engine && pip install -e ".[dev]"`, `pytest`, `python -m narrativefield.simulation.run`
- **Lint:** `cd src/visualization && npx eslint src/` and `cd src/engine && ruff check .`

## Spec Authority Chain
1. Individual spec files are authoritative for their domain
2. `pacing-physics.md` is authoritative for ALL pacing constants (Decision 19)
3. `dinner-party-config.md` is authoritative for character data
4. `scene-segmentation.md` is authoritative for segmentation algorithm (Decision 21)
5. `data-flow.md` defines all interface contracts between subsystems
6. `MASTER_PLAN.md` is the coordination document — consult it for implementation order and open questions

## PR Guidelines
- One PR per implementation step (not per phase)
- Each PR must include tests that verify the step's acceptance criteria
- Reference the spec section that defines the behavior being implemented
- Never modify spec files
- Update CHANGELOG.md with every PR (use [Unreleased] section)

## Implementation Artifacts

- `docs/CODEX_WORKFLOW.md` — 47-task implementation DAG across 4 phases
- `specs/audit/audit-synthesis.md` — 35 findings (7 CRITICAL, 11 HIGH)
- `docs/audits/AUDIT_REPORT.md` — Full audit report

## Existing Implementation

- Storyteller pipeline with LLM-driven prose generation
- LLM gateway with retry and exponential backoff
- Repetition guard for story output diversity
- Per-scene outcome tracking and run artifact reporting
