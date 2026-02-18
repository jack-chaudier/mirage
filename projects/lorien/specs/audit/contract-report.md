# Interface Contract Audit Report

> **Auditor:** integration-auditor (spec-auditor team)
> **Date:** 2026-02-06
> **Scope:** All 5 data-flow.md interface contracts traced field-by-field through producer and consumer specs
> **Primary source:** `specs/integration/data-flow.md`

---

## Executive Summary

**17 issues found across 5 contracts.** Of these:
- **5 MISMATCH** — field definitions conflict between specs
- **5 MISSING_PRODUCER** — data-flow.md declares fields no producer spec generates
- **4 MISSING_CONSUMER** — producer specs output fields data-flow.md omits
- **3 NAME_CONFLICT** — same concept has different names or shapes in different specs

No contract is fully clean. Contract 1 (SimulationOutput) and Contract 3 (NarrativeFieldPayload) have the most issues. The WorldStateSnapshot / SnapshotState naming divergence (Issue C2-1) is the single most impactful problem, as it affects both Contract 1 and Contract 2 and creates confusion about which fields are available at snapshot boundaries.

### Severity Ratings

| Severity | Count | Meaning |
|----------|-------|---------|
| **CRITICAL** | 3 | Will cause implementation failures; must resolve before Phase 1 |
| **HIGH** | 6 | Incorrect semantics or missing fields that will cause bugs |
| **MEDIUM** | 5 | Inconsistencies that create confusion but have workarounds |
| **LOW** | 3 | Cosmetic or documentation-only issues |

---

## Contract 1: SimulationOutput

**data-flow.md Section 3.1 (lines 123-232)**
**Producer:** tick-loop.md, pacing-physics.md, world.md, agents.md
**Consumer:** Metrics pipeline (tension-pipeline.md, irony-and-beliefs.md, scene-segmentation.md), Data Loader

### 1.1 SimulationMetadata

| Field | data-flow.md | Producer Spec | Verdict | Notes |
|-------|-------------|---------------|---------|-------|
| `simulation_id` | `string` | tick-loop.md: not explicitly produced | MISSING_PRODUCER | tick-loop.md has no SimulationMetadata assembly step. This is an integration-layer concern. |
| `scenario` | `string` ("dinner_party") | dinner-party-config.md: scenario name defined | MATCH | |
| `total_ticks` | `number` | tick-loop.md: available from tick counter at end | MATCH | |
| `total_sim_time` | `number` (minutes) | tick-loop.md: available from sim_time at end | MATCH | |
| `agent_count` | `number` | world.md: agent count from WorldDefinition | MATCH | |
| `event_count` | `number` | tick-loop.md: available from event log length | MATCH | |
| `snapshot_interval` | `number` — comment says "ticks between snapshots (default 20)" | world.md line ~660: `snapshot_interval: int = 20` with comment "Take WorldState snapshot every N events" | **MISMATCH** | **data-flow.md says "ticks", world.md says "events". tick-loop.md `should_snapshot` (line 951-956) confirms snapshots are event-count-based, not tick-based. data-flow.md comment is wrong.** |
| `timestamp` | `string` (ISO 8601) | tick-loop.md: not explicitly produced | MISSING_PRODUCER | No producer spec generates a timestamp. Integration-layer concern. |

**Issue C1-1 (HIGH):** `snapshot_interval` comment says "ticks" in data-flow.md:162 but the actual semantics are "events" per world.md and tick-loop.md. The numeric default (20) is correct, but the unit is wrong. Fix: change comment to "events between snapshots (default 20)".

**Issue C1-2 (LOW):** `simulation_id` and `timestamp` are declared but no producer spec shows how they are generated. This is expected (integration glue), but should be documented as "populated by simulation harness, not tick engine."

### 1.2 WorldStateSnapshot

| Field | data-flow.md | Producer (tick-loop.md / world.md) | Consumer (metrics specs) | Verdict | Notes |
|-------|-------------|-----------------------------------|--------------------------|---------|-------|
| `tick_id` | `number` | tick-loop.md: from tick counter | tension-pipeline.md: uses as key | MATCH | |
| `sim_time` | `number` | tick-loop.md: from sim clock | Metrics: used for temporal queries | MATCH | |
| `agents` | `Record<string, AgentState>` | tick-loop.md: from WorldState.agents | tension-pipeline.md: reads agent fields | MATCH | |
| `global_tension` | `number` | tick-loop.md: from computed tension | Metrics: baseline for tension delta | MATCH | |
| `snapshot_id` | **absent** | scenes.md SnapshotState: `str` "snap_{NNN}" | — | MISSING_CONSUMER | scenes.md has it, data-flow.md does not |
| `event_count` | **absent** | scenes.md SnapshotState: `int` | — | MISSING_CONSUMER | scenes.md has it, data-flow.md does not |
| `secrets` | **absent** | scenes.md SnapshotState: `dict[str, SecretDefinition]` | — | MISSING_CONSUMER | scenes.md has it, data-flow.md does not |
| `locations` | **absent** | scenes.md SnapshotState: `dict[str, Location]` | — | MISSING_CONSUMER | scenes.md has it, data-flow.md does not |
| `active_scene_id` | **absent** | scenes.md SnapshotState: `str` | — | — | Only meaningful after segmentation; not in raw snapshot |
| `belief_matrix` | **absent** | scenes.md SnapshotState: `dict` | — | — | Materialized view; derivable from agents[].beliefs |

**Issue C2-1 (CRITICAL / NAME_CONFLICT):** data-flow.md calls this type `WorldStateSnapshot`. scenes.md calls it `SnapshotState`. They have **different field sets**: WorldStateSnapshot has 4 fields; SnapshotState has 9 fields (adds snapshot_id, event_count, secrets, locations, active_scene_id, belief_matrix). tension-pipeline.md's MetricsContext (line 736) references snapshots as `dict[int, WorldState]` — a **third name**. These must be reconciled into a single canonical type.

**Recommendation:** Adopt scenes.md `SnapshotState` as canonical (it is the superset). Update data-flow.md WorldStateSnapshot to match. Update tension-pipeline.md to use the same type name. The extra fields (active_scene_id, belief_matrix) can be optional/computed-after-segmentation.

### 1.3 SecretDefinition

| Field | data-flow.md | world.md | Verdict |
|-------|-------------|----------|---------|
| `id` | `string` | `str` | MATCH |
| `holder` | `string` | `str` | MATCH |
| `about` | `string \| null` | `str \| None` | MATCH |
| `content_type` | `string` | `str` | MATCH |
| `content_display` | `string` | **absent** (uses `description` instead) | **NAME_CONFLICT** |
| `truth_value` | `boolean` | `bool` | MATCH |
| `description` | **absent** | `str` (human-readable description) | MISSING_CONSUMER |
| `initial_knowers` | **absent** | `list[str]` | MISSING_CONSUMER |
| `initial_suspecters` | **absent** | `list[str]` | MISSING_CONSUMER |
| `dramatic_weight` | **absent** | `float [0.0, 1.0]` | MISSING_CONSUMER |
| `reveal_consequences` | **absent** | `str` | MISSING_CONSUMER |

**Issue C1-3 (CRITICAL / NAME_CONFLICT):** data-flow.md uses `content_display` while world.md uses `description` for the human-readable text. These are the same concept with different names. One must be chosen as canonical.

**Issue C1-4 (HIGH):** data-flow.md SecretDefinition omits 4 fields that world.md defines: `initial_knowers`, `initial_suspecters`, `dramatic_weight`, `reveal_consequences`. These are needed by the simulation for initialization and by irony-and-beliefs.md for weighting. The data-flow.md contract must be updated to include them, or explicitly document that SimulationOutput serializes the full world.md SecretDefinition (not a subset).

### 1.4 LocationDefinition

| Field | data-flow.md | world.md Location | Verdict |
|-------|-------------|-------------------|---------|
| `id` | `string` | `str` | MATCH |
| `name` | `string` | `str` | MATCH |
| `privacy` | `number [0.0, 1.0]` | `float [0.0, 1.0]` | MATCH |
| `capacity` | `number` | `int` | MATCH |
| `adjacent` | `string[]` | `list[str]` | MATCH |
| `overhear_from` | `string[]` | `list[str]` | MATCH |
| `overhear_probability` | **absent** | `float [0.0, 1.0]` | MISSING_CONSUMER |
| `description` | **absent** | `str` | MISSING_CONSUMER |

**Issue C1-5 (HIGH):** data-flow.md LocationDefinition is missing `overhear_probability` and `description` from world.md. The `overhear_probability` is functionally critical — without it, overhearing mechanics cannot work. The `description` is needed for tooltips and prose generation.

### 1.5 RawEvent

| Field | data-flow.md | events.md | Verdict |
|-------|-------------|-----------|---------|
| All base Event fields | `type RawEvent = Event` | Event dataclass | MATCH |
| `metrics` default | "metrics fields are all 0.0 / empty" | Default factory: `{tension: 0.0, irony: 0.0, significance: 0.0, thematic_shift: {}}` | MATCH |

**Verdict: MATCH.** RawEvent is simply an Event alias with zero-valued metrics.

---

## Contract 2: MetricsPipelineOutput (EventLog -> Metrics)

**data-flow.md Section 3.2 (lines 234-314)**
**Producer:** tension-pipeline.md, irony-and-beliefs.md, scene-segmentation.md
**Consumer:** Bundler (which creates NarrativeFieldPayload), story-extraction.md

### 2.1 MetricsPipelineOutput Structure

| Field | data-flow.md | Producer | Verdict |
|-------|-------------|----------|---------|
| `events` | `list[Event]` (metrics populated) | All metrics specs mutate in place | MATCH |
| `scenes` | `list[Scene]` | scene-segmentation.md | MATCH |
| `belief_snapshots` | `list[BeliefSnapshot]` | data-flow.md Section 6: `extract_belief_snapshots()` | MATCH (internally defined) |

### 2.2 Per-Event Metrics After Pipeline

| Metric Field | data-flow.md | Producer Spec | events.md default | Verdict |
|-------------|-------------|---------------|-------------------|---------|
| `tension` | `float [0.0, 1.0]` | tension-pipeline.md: TensionPayload.tension | `0.0` in default | MATCH |
| `tension_components` | `dict` with 8 sub-metrics | tension-pipeline.md: TensionPayload.tension_components | **absent from default factory and EventMetrics TS interface** | **MISMATCH** |
| `irony` | `float [0.0, +inf)` | irony-and-beliefs.md: per-event irony | `0.0` in default | MATCH |
| `irony_collapse` | `null` or `{detected, drop, collapsed_beliefs, score}` | irony-and-beliefs.md Section 7.1: same shape | **absent from default factory and EventMetrics TS interface** | **MISMATCH** |
| `significance` | `float [0.0, 1.0]` | Phase 5 (future) | `0.0` in default | MATCH |
| `thematic_shift` | `dict[str, float]` | data-flow.md Section 7: thematic rules | `{}` in default | MATCH |

**Issue C2-2 (CRITICAL):** `tension_components` and `irony_collapse` are expected by data-flow.md Section 3.2 and by renderer-architecture.md (EventMetrics TS interface lines 231-238), but they are **absent from events.md** — both the Python default factory (lines 312-317) and the TypeScript EventMetrics interface (lines 357-362). events.md is the canonical Event schema. This creates an ambiguity: are these fields part of the Event or bolted on?

**Recommendation:** Update events.md to include `tension_components` and `irony_collapse` in both the Python default factory and TypeScript EventMetrics interface. These are populated by the metrics pipeline (not the simulation), which is already the case for tension and irony. The default values should be:
- `tension_components`: `{}` (empty dict, populated by tension pipeline)
- `irony_collapse`: `null` (populated by irony pipeline when collapse detected)

### 2.3 BeliefSnapshot

| Field | data-flow.md | irony-and-beliefs.md Section 7.2 | Verdict |
|-------|-------------|----------------------------------|---------|
| `tick_id` | `number` | `tick_id: 42` (in JSON example) | MATCH |
| `sim_time` | `number` | **absent** | **MISMATCH** |
| `beliefs` | `Record<string, Record<string, string>>` | Same shape in JSON example | MATCH |
| `agent_irony` | `Record<string, number>` | Same shape in JSON example | MATCH |
| `scene_irony` | `number` | Same in JSON example | MATCH |
| `pairwise_irony` | **absent** | `"pairwise_irony": {"thorne-marcus": 3.5, ...}` | MISSING_CONSUMER |

**Issue C2-3 (MEDIUM):** data-flow.md BeliefSnapshot includes `sim_time` but irony-and-beliefs.md output format does not. Conversely, irony-and-beliefs.md includes `pairwise_irony` but data-flow.md BeliefSnapshot does not. Both are useful: `sim_time` enables temporal interpolation in the renderer; `pairwise_irony` enables rich tooltip content.

**Recommendation:** Add both fields to the canonical BeliefSnapshot definition. `sim_time` is trivially available from the snapshot. `pairwise_irony` is computed by the irony pipeline.

### 2.4 IronyCollapse In-Event Format

| Field | data-flow.md Section 3.2 JSON | irony-and-beliefs.md Section 7.1 JSON | IronyCollapse dataclass (line 333) | Verdict |
|-------|-------------------------------|---------------------------------------|-------------------------------------|---------|
| `detected` | implicit (null = not detected) | `"detected": true` | **absent** (dataclass has no `detected` field) | **MISMATCH** |
| `drop` | — | `"drop": 0.29` | `drop: float` | MATCH |
| `collapsed_beliefs` | — | `[{agent, secret, from, to}]` | `collapsed_beliefs: list[tuple[str, str]]` (just agent_id, secret_id) | **MISMATCH** |
| `score` | — | `"score": 0.82` | **absent** | MISSING_PRODUCER |
| `event_id` | — | — | `event_id: str` | — (internal) |
| `irony_before` | — | — | `irony_before: float` | — (internal) |
| `irony_after` | — | — | `irony_after: float` | — (internal) |

**Issue C2-4 (HIGH):** The irony_collapse format is inconsistent across three representations within irony-and-beliefs.md itself:
1. The `IronyCollapse` dataclass (line 333): `collapsed_beliefs: list[tuple[str, str]]` (just agent_id, secret_id pairs)
2. The JSON output format (Section 7.1, line 705): `collapsed_beliefs: [{agent, secret, from, to}]` (richer, includes belief transitions)
3. The JSON output format includes `detected` and `score` fields not in the dataclass

data-flow.md Section 3.2 shows irony_collapse as `null` (when no collapse) or an object, matching the Section 7.1 JSON format — but the dataclass and JSON disagree on `collapsed_beliefs` shape and the presence of `detected`/`score`.

**Recommendation:** The JSON output format (Section 7.1) is the interface contract. Update the IronyCollapse dataclass to match: add `detected: bool`, `score: float`, and change `collapsed_beliefs` to `list[dict]` with `{agent, secret, from, to}` structure.

---

## Contract 3: NarrativeFieldPayload

**data-flow.md Section 3.3 (lines 316-405)**
**Producer:** Bundler (`bundle_for_renderer` function)
**Consumer:** renderer-architecture.md DataLoader, Zustand Store

### 3.1 Top-Level Shape

| Field | data-flow.md | renderer-architecture.md (Section 13.5, line 923) | Verdict |
|-------|-------------|---------------------------------------------------|---------|
| `metadata` | `SimulationMetadata` | **absent** | MISSING_CONSUMER |
| `agents` | `AgentManifest[]` | `AgentManifest[]` | MATCH (with field differences; see 3.2) |
| `locations` | `LocationDefinition[]` | `Location[]` | MATCH (type alias) |
| `secrets` | `SecretDefinition[]` | `Secret[]` | MATCH (type alias) |
| `events` | `Event[]` | `Event[]` | MATCH |
| `scenes` | `Scene[]` | `Scene[]` | MATCH |
| `belief_snapshots` | `BeliefSnapshot[]` | **absent** | MISSING_CONSUMER |

**Issue C3-1 (MEDIUM):** renderer-architecture.md's proposed NarrativeFieldPayload (Section 13.5) omits `metadata` and `belief_snapshots`. data-flow.md includes both. The renderer spec acknowledges this shape is "to be confirmed in data-flow.md" — so data-flow.md is authoritative. The renderer spec should be updated to match.

### 3.2 AgentManifest

| Field | data-flow.md | renderer-architecture.md | agents.md AgentState | Verdict |
|-------|-------------|--------------------------|---------------------|---------|
| `id` | `string` | `string` | `str` | MATCH |
| `name` | `string` | `string` | `str` | MATCH |
| `initial_location` | `string` | `string` | Location from initial state | MATCH |
| `goal_summary` | `string` | **absent** | No `goal_summary` field; has `goals: list[GoalVector]` | **MISMATCH** |
| `primary_flaw` | `string` | **absent** | No `primary_flaw` field; has `flaws: list[CharacterFlaw]` | **MISMATCH** |

**Issue C3-2 (MEDIUM):** data-flow.md defines AgentManifest with `goal_summary` and `primary_flaw` (string summaries for tooltip display). renderer-architecture.md defines it with only 3 fields (id, name, initial_location). agents.md AgentState has raw `goals: list[GoalVector]` and `flaws: list[CharacterFlaw]` but no summary strings. The `bundle_for_renderer` function (data-flow.md line 392-394) shows how these are computed: `summarize_goals(agent.goals)` and `agent.flaws[0].description`. The function `summarize_goals` is not defined anywhere.

**Recommendation:** Accept data-flow.md as authoritative (it is the integration spec). Update renderer-architecture.md to include `goal_summary` and `primary_flaw`. Define `summarize_goals()` in data-flow.md or agents.md.

### 3.3 Scene (in NarrativeFieldPayload)

| Field | data-flow.md (uses scenes.md Scene) | renderer-architecture.md (TS interface, line 268) | scene-segmentation.md (dataclass, line 352) | Verdict |
|-------|-------------------------------------|---------------------------------------------------|---------------------------------------------|---------|
| `id` | `string` | `string` | `str` | MATCH |
| `event_ids` | `string[]` | `string[]` | `list[str]` | MATCH |
| `location` | `string` | `string` | `str` | MATCH |
| `participants` | `string[]` | `string[]` | `set[str]` | MATCH (set serialized as array) |
| `time_start` | `number` | `number` | `float` | MATCH |
| `time_end` | `number` | `number` | `float` | MATCH |
| `tension_arc` | `number[]` | `number[]` | `list[float]` | MATCH |
| `dominant_theme` | `string` | `string` | `str` | MATCH |
| `scene_type` | `string` | `string` | `str` | MATCH |

**Verdict: MATCH.** Scene interface is consistent across all three specs. The only note is that `participants` is a Python `set[str]` in the dataclass but serializes as a JSON array, which is standard.

### 3.4 Event (in NarrativeFieldPayload)

| Field | data-flow.md (enriched Event) | renderer-architecture.md (TS Event, line 251) | events.md (canonical) | Verdict |
|-------|------------------------------|-----------------------------------------------|----------------------|---------|
| Core fields (id, sim_time, tick_id, etc.) | From events.md | Matches events.md TS | Canonical | MATCH |
| `metrics.tension` | `float [0.0, 1.0]` | `number` | `0.0` default | MATCH |
| `metrics.irony` | `float [0.0, +inf)` | `number` | `0.0` default | MATCH |
| `metrics.significance` | `float [0.0, 1.0]` | `number` | `0.0` default | MATCH |
| `metrics.thematic_shift` | `Record<string, number>` | `Record<string, number>` | `{}` default | MATCH |
| `metrics.tension_components` | `TensionComponents` (8 sub-metrics) | `TensionComponents` (8 sub-metrics) | **absent** | **MISMATCH** (see Issue C2-2) |

**Issue:** Same as C2-2. events.md is missing `tension_components` from EventMetrics.

### 3.5 format_version

**Issue C3-3 (MEDIUM):** data-flow.md Section 11 (line 815) declares that all file formats include `format_version: "1.0.0"`. However:
- `SimulationOutput` interface (line 135) has no `format_version` field
- `NarrativeFieldPayload` interface (line 330) has no `format_version` field
- No producer spec generates this value
- The bundle_for_renderer function (line 385) does not include it

**Recommendation:** Add `format_version: string` to both SimulationOutput and NarrativeFieldPayload interfaces. Ensure the bundler and simulation harness populate it.

---

## Contract 4: StoryExtractionRequest / StoryExtractionResponse

**data-flow.md Section 3.4 (lines 407-468)**
**Producer:** Renderer interaction state (user action)
**Consumer:** story-extraction.md

### 4.1 StoryExtractionRequest

| Field | data-flow.md | Renderer (interaction-model.md) | story-extraction.md | Verdict |
|-------|-------------|--------------------------------|---------------------|---------|
| `selection_type` | `"region" \| "arc" \| "query"` | Region select + arc crystallization in renderer | `search_stories` takes `StoryQuery` (line 920) | MATCH (conceptual) |
| `event_ids` | `string[]` | From region select or arc selection | `events: list[Event]` (resolved from IDs) | MATCH |
| `protagonist_agent_id` | `string?` | From arc crystallization (selected thread) | `StoryQuery.protagonist: str \| None` | MATCH |
| `tension_weights` | `TensionWeights` | From slider state in Zustand store | `score_arc` takes `weights: TensionWeights` | MATCH |
| `genre_preset` | `string` | From toolbar preset selector | `BeatSheet.genre_preset` | MATCH |
| `query_text` | `string?` | Phase 6 (future) | `StoryQuery` (Phase 6) | MATCH (both deferred) |

**Verdict: MATCH.** The story extraction request interface is clean and consistent.

### 4.2 StoryExtractionResponse

| Field | data-flow.md | story-extraction.md | Verdict |
|-------|-------------|---------------------|---------|
| `validation` | `ArcValidation {valid, violations}` | `validate_arc` returns validation result | MATCH |
| `beats` | `BeatClassification[] {event_id, beat_type}` | `classify_beats` returns `list[BeatType]` (not BeatClassification) | **MISMATCH** |
| `score` | `ArcScore` | `score_arc` returns `ArcScore` dataclass | MATCH |
| `beat_sheet` | `BeatSheet` | `BeatSheet` dataclass (Section 5.1) | MATCH |
| `suggestions` | `string[]` | Worked example shows suggestion text | MATCH |

**Issue C4-1 (MEDIUM):** data-flow.md defines `BeatClassification` as `{event_id: string, beat_type: BeatType}`. story-extraction.md's `classify_beats` function returns `list[BeatType]` (just the beat types, positionally aligned with events). The event_id mapping must be constructed by the caller. This is a minor impedance mismatch — the data-flow.md format is richer and easier to consume. story-extraction.md should be updated to return `BeatClassification[]` or the mapping should be done in the integration layer.

---

## Contract 5: LLMStoryRequest / LLMStoryResponse

**data-flow.md Section 3.5 (lines 470-505)**
**Producer:** story-extraction.md (LLM Prompt Builder)
**Consumer:** Claude API

### 5.1 LLMStoryRequest

| Field | data-flow.md | story-extraction.md | Verdict |
|-------|-------------|---------------------|---------|
| `prompt` | `string` | `build_llm_prompt(beat_sheet: BeatSheet) -> str` | MATCH |
| `model` | `string` ("claude-sonnet-4-5-20250929") | Not specified (uses Claude API) | MATCH |
| `max_tokens` | `number` (300 * beat_count) | Prompt template says "200-400 words per beat" | MATCH (consistent) |
| `temperature` | `number` (0.8) | Not specified | MATCH (integration config) |

### 5.2 LLMStoryResponse

| Field | data-flow.md | story-extraction.md | Verdict |
|-------|-------------|---------------------|---------|
| `prose` | `string` | LLM output text | MATCH |
| `arc_id` | `string` | From BeatSheet.arc_id | MATCH |
| `input_tokens` | `number` | API response metadata | MATCH |
| `output_tokens` | `number` | API response metadata | MATCH |

**Verdict: MATCH.** The LLM interface is clean. Token counts come from the API response, not the story extraction spec. The model version string will need updating for production, but that is expected.

---

## Cross-Contract Issues

### X-1: Snapshot Interval Unit Confusion (HIGH)

**Affected contracts:** C1, C2
**Location:** data-flow.md:162, world.md `snapshot_interval`, tick-loop.md `should_snapshot`

data-flow.md says "ticks between snapshots" but the actual implementation is "events between snapshots." Additionally, data-flow.md Section 6 (line 621) says "Belief snapshots are extracted at each WorldState snapshot interval (every 20 ticks)" — again using "ticks" when it should say "events."

**Fix:** Two lines in data-flow.md need the word "ticks" changed to "events":
1. Line 162: `snapshot_interval: number; // events between snapshots (default 20)`
2. Line 621: "Belief snapshots are extracted at each WorldState snapshot interval (every 20 events)"

### X-2: EventMetrics Schema Gap (CRITICAL)

**Affected contracts:** C2, C3
**Location:** events.md (Python lines 312-317, TS lines 357-362), data-flow.md Section 3.2, renderer-architecture.md lines 231-248

events.md (the canonical Event schema) does not include `tension_components` or `irony_collapse` in EventMetrics. Both data-flow.md and renderer-architecture.md expect these fields. The metrics pipeline (tension-pipeline.md, irony-and-beliefs.md) produces them.

**Impact:** An implementer coding against events.md will not include these fields. An implementer coding against renderer-architecture.md will expect them. This will cause a runtime type mismatch in Phase 1 (renderer) when loading fake data.

**Fix:** Update events.md to add `tension_components` and `irony_collapse` to both the Python dict default factory and the TypeScript EventMetrics interface.

### X-3: Three Names for WorldState Snapshot (CRITICAL)

**Affected contracts:** C1, C2
**Location:** data-flow.md (WorldStateSnapshot), scenes.md (SnapshotState), tension-pipeline.md (WorldState)

Three different specs use three different names for the same concept:
1. `WorldStateSnapshot` (data-flow.md) — 4 fields
2. `SnapshotState` (scenes.md) — 9 fields (superset)
3. `WorldState` (tension-pipeline.md MetricsContext) — referenced as `dict[int, WorldState]`

**Fix:** Choose one canonical name and field set. Recommended: use `WorldStateSnapshot` as the name (matches data-flow.md intent) but adopt the scenes.md field set (it is the superset). Update all three specs.

### X-4: format_version Orphan (LOW)

**Affected contracts:** C1, C3
**Location:** data-flow.md Section 11 (line 815)

`format_version` is declared as a top-level field in "all file formats" but is not present in any interface definition or producer function.

**Fix:** Add `format_version: string` to SimulationOutput and NarrativeFieldPayload interfaces. Add it to `bundle_for_renderer`.

---

## Missing Contract Coverage

The following data flows occur in the system but are NOT covered by explicit interface contracts in data-flow.md:

| Flow | From | To | Status |
|------|------|----|--------|
| Tension weights to client-side recompute | Zustand Store (slider state) | tension-computer.ts | **Covered implicitly** in data-flow.md Section 5 (Data Flow Summary Table) but no formal interface defined. renderer-architecture.md Section 9 defines it. |
| Index tables to metrics pipeline | Event log | IndexTables | **Covered** in data-flow.md Section 4. |
| Thematic shift computation | Events | Events (mutation) | **Covered** in data-flow.md Section 7 but no input/output interface — it is inline in the pipeline. |
| CausalIndex / ThreadIndex | Data Loader | Zustand Store | **Covered** in renderer-architecture.md Sections 4/8 but not in data-flow.md. |

No critical gaps in contract coverage.

---

## Summary of All Issues

| ID | Severity | Contract | Type | Summary |
|----|----------|----------|------|---------|
| C1-1 | HIGH | SimulationOutput | MISMATCH | `snapshot_interval` comment says "ticks", should say "events" |
| C1-2 | LOW | SimulationOutput | MISSING_PRODUCER | `simulation_id` and `timestamp` have no producer |
| C1-3 | CRITICAL | SimulationOutput | NAME_CONFLICT | SecretDefinition uses `content_display` vs world.md `description` |
| C1-4 | HIGH | SimulationOutput | MISSING_CONSUMER | SecretDefinition missing 4 fields from world.md |
| C1-5 | HIGH | SimulationOutput | MISSING_CONSUMER | LocationDefinition missing `overhear_probability`, `description` |
| C2-1 | CRITICAL | WorldStateSnapshot | NAME_CONFLICT | Three specs use three different type names and field sets |
| C2-2 | CRITICAL | MetricsOutput | MISMATCH | `tension_components` and `irony_collapse` absent from events.md |
| C2-3 | MEDIUM | MetricsOutput | MISMATCH | BeliefSnapshot: data-flow.md has `sim_time` (producer lacks it); producer has `pairwise_irony` (data-flow.md lacks it) |
| C2-4 | HIGH | MetricsOutput | MISMATCH | IronyCollapse dataclass vs JSON format: `collapsed_beliefs` shape differs; `detected`/`score` missing from dataclass |
| C3-1 | MEDIUM | NarrativeFieldPayload | MISSING_CONSUMER | renderer-architecture.md omits `metadata` and `belief_snapshots` |
| C3-2 | MEDIUM | NarrativeFieldPayload | MISMATCH | AgentManifest: data-flow.md has 5 fields, renderer-architecture.md has 3 |
| C3-3 | MEDIUM | NarrativeFieldPayload | MISSING_PRODUCER | `format_version` declared but never produced |
| C4-1 | MEDIUM | StoryExtraction | MISMATCH | `classify_beats` returns `list[BeatType]` vs `BeatClassification[]` |
| X-1 | HIGH | Cross-contract | MISMATCH | "ticks" vs "events" in two data-flow.md locations |
| X-2 | CRITICAL | Cross-contract | MISMATCH | events.md missing tension_components and irony_collapse |
| X-3 | CRITICAL | Cross-contract | NAME_CONFLICT | Three names for WorldState snapshot type |
| X-4 | LOW | Cross-contract | MISSING_PRODUCER | format_version orphan |

### Priority Fix Order

1. **X-2 + C2-2:** Add `tension_components` and `irony_collapse` to events.md EventMetrics (CRITICAL — blocks Phase 1 fake data)
2. **X-3 + C2-1:** Unify WorldStateSnapshot/SnapshotState/WorldState naming (CRITICAL — cross-spec confusion)
3. **C1-3:** Resolve `content_display` vs `description` in SecretDefinition (CRITICAL — naming conflict)
4. **C1-4 + C1-5:** Add missing fields to data-flow.md SecretDefinition and LocationDefinition (HIGH)
5. **C2-4:** Reconcile IronyCollapse dataclass with JSON output format (HIGH)
6. **X-1 + C1-1:** Fix "ticks" to "events" in snapshot_interval comments (HIGH)
7. **C3-1 + C3-2:** Update renderer-architecture.md NarrativeFieldPayload and AgentManifest (MEDIUM)
8. **C2-3:** Add `sim_time` and `pairwise_irony` to BeliefSnapshot (MEDIUM)
9. **C4-1:** Return `BeatClassification[]` from classify_beats (MEDIUM)
10. **C3-3 + X-4:** Add format_version to interfaces and producers (LOW)

---

## Appendix: Files Audited

| File | Lines Read | Role |
|------|-----------|------|
| `specs/integration/data-flow.md` | 1-853 (full) | Primary: defines all 5 contracts |
| `specs/simulation/tick-loop.md` | 1-1467 (full) | Producer: SimulationOutput |
| `specs/schema/events.md` | 1-1073 (full) | Canonical Event schema |
| `specs/schema/world.md` | 1-750 (full) | Location, SecretDefinition schemas |
| `specs/schema/agents.md` | 1-1290 (full) | AgentState, GoalVector, CharacterFlaw |
| `specs/schema/scenes.md` | 1-781 (full) | Scene, SnapshotState schemas |
| `specs/metrics/tension-pipeline.md` | 1-1078 (full) | Tension computation, TensionPayload |
| `specs/metrics/irony-and-beliefs.md` | 1-799 (full) | Irony computation, IronyCollapse, BeliefSnapshot |
| `specs/metrics/scene-segmentation.md` | 1-639 (full) | Scene segmentation, Scene output |
| `specs/metrics/story-extraction.md` | 1-989 (full) | Arc grammar, BeatSheet, LLM prompt |
| `specs/visualization/renderer-architecture.md` | 1-937 (full) | NarrativeFieldPayload consumer, Zustand store |
