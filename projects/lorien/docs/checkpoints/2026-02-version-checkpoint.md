# 2026-02 Version Checkpoint

Date: 2026-02-11  
Scope: checkpoint-quality cleanup, deterministic validation, and portfolio-ready evidence packaging.

## Evidence Sources

- Quality-gate command outputs in this checkpoint pass.
- Generated deterministic science artifact:
  - `data/checkpoints/2026-02-checkpoint-science.json`
- Generated checkpoint graphic:
  - `assets/checkpoint-overview-2026-02.svg`

## Commands and Outcomes

| Area | Command | Result |
|---|---|---|
| Engine tests | `cd src/engine && ./.venv/bin/python -m pytest -q` | `84 passed in 2.59s` |
| Engine lint | `cd src/engine && ./.venv/bin/ruff check .` | `All checks passed!` |
| Visualization tests | `cd src/visualization && npm test` | `11 files passed, 62 tests passed` |
| Visualization lint | `cd src/visualization && npm run lint` | `eslint src --max-warnings=0` exit `0` |
| Visualization build | `cd src/visualization && npm run build` | `vite build` success (`79 modules`, `index-CX8PW9fK.js` gzip `90.16 kB`) |
| Checkpoint science export | `./src/engine/.venv/bin/python scripts/science_harness.py --seeds 1,2,3,4,5,6,7,8,9,10,42,51 --output data/checkpoints/2026-02-checkpoint-science.json` | Success |
| Checkpoint SVG render | `./src/engine/.venv/bin/python scripts/render_checkpoint_graphic.py --input data/checkpoints/2026-02-checkpoint-science.json --output assets/checkpoint-overview-2026-02.svg` | Success |

## Deterministic Sweep Snapshot

Source: `data/checkpoints/2026-02-checkpoint-science.json`

- `generated_at`: `2026-02-11T17:19:18.107286+00:00`
- `commit`: `45a47b25e0e433300cb84d911439ac624a37ac33`
- `run_count`: `12` seeds (`1-10, 42, 51`)
- `determinism_check.matches`: `true`
- `runs_with_catastrophe`: `12/12`
- `total_catastrophes`: `50`
- `dominant_catastrophe_agent`: `victor` (`34%` of catastrophes)

Representative seeds (ranking fields from artifact):

| Rank | Seed | Catastrophes | Catastrophe Agent Diversity | Locations Used | Total Sim Time |
|---|---|---|---|---|---|
| 1 | 9 | 8 | 6 | 5 | 125.0 |
| 2 | 10 | 7 | 6 | 4 | 121.0 |
| 3 | 6 | 7 | 5 | 5 | 125.0 |

Mean location share across sweep:

| Location | Mean Share |
|---|---|
| dining_table | 76.83% |
| balcony | 12.92% |
| foyer | 4.04% |
| kitchen | 3.17% |
| bathroom | 3.04% |

## Cleanup Notes

- Archived historical audits moved under `docs/audits/archive/`.
- Local `data/` debris from iterative runs was removed while preserving canonical fixtures:
  - `data/fake-dinner-party.nf-viz.json`
  - `data/dinner_party_001.nf-sim.json`
  - `data/dinner_party_001.nf-viz.json`

## Artifact Paths

- Checkpoint summary graphic: `assets/checkpoint-overview-2026-02.svg`
- Checkpoint science data: `data/checkpoints/2026-02-checkpoint-science.json`
- Audit index: `docs/audits/README.md`
- This report: `docs/checkpoints/2026-02-version-checkpoint.md`
