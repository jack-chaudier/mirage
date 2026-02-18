# NarrativeField Engine (Python)

## Tech Stack
- Python 3.12+ with dataclasses and type hints throughout
- NumPy for vectorized metric computation
- Claude API (anthropic SDK) for content generation and story prose
- pytest for testing

## Authoritative Specs
- Schema: `specs/schema/events.md`, `agents.md`, `world.md`, `scenes.md`
- Simulation: `specs/simulation/tick-loop.md`, `decision-engine.md`, `pacing-physics.md`, `dinner-party-config.md`
- Metrics: `specs/metrics/tension-pipeline.md`, `irony-and-beliefs.md`, `scene-segmentation.md`, `story-extraction.md`
- Integration: `specs/integration/data-flow.md`

## Key Rules
- pacing-physics.md is authoritative for ALL pacing constants (Decision 19)
- Metrics pipeline execution order is STRICT (Decision 20): irony → thematic → tension → scene segmentation
- BeatType has exactly 5 types: SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE (Decision 18)
- CATASTROPHE events are triggered by pacing physics, NOT by decision engine
- Catastrophe formula: stress * commitment² + suppression_bonus > 0.35 AND composure < 0.30

## Canonical Values (from pacing-physics.md)
- catastrophe_threshold: 0.35
- composure_gate: 0.30
- composure_mask_threshold: 0.40
- budget_recharge: 0.08/tick
- commitment_decay: 0.01/tick below 0.50
- catastrophe_commitment_boost: +0.10
- recovery_timer: 8 ticks

## Existing Modules

- **storyteller** — repetition_guard.py for output diversity
- **LLM gateway** — retry with exponential backoff
- **simulation** — tick_loop, decision_engine, pacing, scenarios
- **metrics** — tension, irony, thematic, segmentation
- **extraction** — arc grammar, beat sheets
- **integration** — event_bundler

## Workflow

- Update /CHANGELOG.md when making engine changes
