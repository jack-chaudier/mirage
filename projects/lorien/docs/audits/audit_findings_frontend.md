# Frontend Types, Build, and Cross-System Alignment Audit

**Auditor:** frontend-auditor
**Date:** 2026-02-08
**Branch:** codex/integration/wps1-4

## Summary

- **Build:** PASS (tsc --noEmit + vite build, 71 modules, no errors)
- **Tests:** PASS (8/8 tests across 4 files)
- **Total findings:** 15 (3 HIGH, 6 MEDIUM, 6 LOW)

---

## 1. Enum Cross-Reference

### 1.1 EventType: MATCH

| TS (`src/visualization/src/types/events.ts:3-14`) | Python (`src/engine/narrativefield/schema/events.py:8-18`) |
|---|---|
| CHAT, OBSERVE, SOCIAL_MOVE, REVEAL, CONFLICT, INTERNAL, PHYSICAL, CONFIDE, LIE, CATASTROPHE | CHAT, OBSERVE, SOCIAL_MOVE, REVEAL, CONFLICT, INTERNAL, PHYSICAL, CONFIDE, LIE, CATASTROPHE |

**Result:** Exact match. All 10 values identical with identical string representations.

### 1.2 BeatType: MATCH

| TS (`src/visualization/src/types/events.ts:16-22`) | Python (`src/engine/narrativefield/schema/events.py:21-26`) |
|---|---|
| SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE | SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE |

**Result:** Exact match. 5 values, consistent with Decision 18.

TS extraction file (`src/visualization/src/types/extraction.ts:1`) imports BeatType from `./events`, so no duplication.

### 1.3 DeltaKind: MATCH

| TS (`src/visualization/src/types/events.ts:24-34`) | Python (`src/engine/narrativefield/schema/events.py:29-38`) |
|---|---|
| AGENT_EMOTION, AGENT_RESOURCE, AGENT_LOCATION, RELATIONSHIP, BELIEF, SECRET_STATE, WORLD_RESOURCE, COMMITMENT, PACING | AGENT_EMOTION, AGENT_RESOURCE, AGENT_LOCATION, RELATIONSHIP, BELIEF, SECRET_STATE, WORLD_RESOURCE, COMMITMENT, PACING |

**Result:** Exact match. 9 values identical.

### 1.4 DeltaOp: MATCH

| TS (`src/visualization/src/types/events.ts:36-39`) | Python (`src/engine/narrativefield/schema/events.py:41-43`) |
|---|---|
| SET, ADD | SET, ADD |

**Result:** Exact match.

### 1.5 FlawType: MATCH

| TS (`src/visualization/src/types/agents.ts:1-12`) | Python (`src/engine/narrativefield/schema/agents.py:8-18`) |
|---|---|
| PRIDE, LOYALTY, TRAUMA, AMBITION, JEALOUSY, COWARDICE, VANITY, GUILT, OBSESSION, DENIAL | PRIDE, LOYALTY, TRAUMA, AMBITION, JEALOUSY, COWARDICE, VANITY, GUILT, OBSESSION, DENIAL |

**Result:** Exact match. 10 values identical.

### 1.6 BeliefState: MATCH

| TS (`src/visualization/src/types/agents.ts:47-52`) | Python (`src/engine/narrativefield/schema/agents.py:132-136`) |
|---|---|
| UNKNOWN, SUSPECTS, BELIEVES_TRUE, BELIEVES_FALSE | UNKNOWN, SUSPECTS, BELIEVES_TRUE, BELIEVES_FALSE |

**Result:** Exact match. 4 values identical.

### 1.7 SceneType: TS ONLY (no Python enum)

- **TS** (`src/visualization/src/types/scenes.ts:5-12`): Defines `SceneType` enum with 6 values: CATASTROPHE, CONFRONTATION, REVELATION, BONDING, ESCALATION, MAINTENANCE
- **Python** (`src/engine/narrativefield/schema/scenes.py:25`): `scene_type: str = ""` -- plain string, no enum

**Finding F-01 (MEDIUM):** SceneType enum exists only in TypeScript. Python `Scene.scene_type` is an untyped string. The TS `Scene` interface at `scenes.ts:28` declares `scene_type: SceneType | string`, which is a union that accepts any string, so there is no runtime mismatch -- but the Python side has no validation against the enum values. If new scene types are added on the Python side, TS will silently accept them due to the `| string` union.

---

## 2. Interface / Dataclass Field Comparison

### 2.1 Event: MATCH

| Field | TS (`events.ts:68-92`) | Python (`events.py:177-195`) | Match? |
|---|---|---|---|
| id | string | str | YES |
| sim_time | number | float | YES |
| tick_id | number | int | YES |
| order_in_tick | number | int | YES |
| type | EventType | EventType | YES |
| source_agent | string | str | YES |
| target_agents | string[] | list[str] | YES |
| location_id | string | str | YES |
| causal_links | string[] | list[str] | YES |
| deltas | StateDelta[] | list[StateDelta] | YES |
| description | string | str | YES |
| dialogue | string \| null (optional) | Optional[str] | YES |
| content_metadata | Record<string,unknown> \| null (optional) | Optional[dict[str,Any]] | YES |
| beat_type | BeatType \| null (optional) | Optional[BeatType] | YES |
| metrics | EventMetrics | EventMetrics | YES |

**Result:** Perfect field-level match.

### 2.2 EventMetrics: STRUCTURAL MISMATCH

| Field | TS (`events.ts:59-66`) | Python (`events.py:139-148`) | Match? |
|---|---|---|---|
| tension | number | float | YES |
| irony | number | float | YES |
| significance | number | float | YES |
| thematic_shift | Record<string,number> | dict[str,float] | YES |
| tension_components | **TensionComponents** (typed 8-field interface) | **dict[str,float]** (untyped dict) | STRUCTURAL MISMATCH |
| irony_collapse | IronyCollapseInfo \| null | Optional[IronyCollapseInfo] | YES |

**Finding F-02 (MEDIUM):** `tension_components` type differs between TS and Python. The TS side uses a structured `TensionComponents` interface (defined at `renderer.ts:3-12`) with 8 named fields (danger, time_pressure, goal_frustration, relationship_volatility, information_gap, resource_scarcity, moral_cost, irony_density). The Python side uses `dict[str, float]`. This means:
- Python can emit extra/missing keys without type checking catching it.
- The TS side enforces the 8-field structure at compile time, providing better safety.
- This is a known pre-implementation audit finding (S-03+S-15) from the audit-synthesis.

### 2.3 StateDelta: MATCH

All 8 fields match between TS (`events.ts:41-50`) and Python (`events.py:49-60`).

### 2.4 IronyCollapseInfo: MINOR MISMATCH

| Field | TS (`events.ts:52-57`) | Python (`events.py:112-117`) | Match? |
|---|---|---|---|
| detected | boolean | bool | YES |
| drop | number | float | YES |
| collapsed_beliefs | Array<{agent,secret,from,to}> | list[CollapsedBelief] | MINOR |
| score | number | float | YES |

**Finding F-03 (LOW):** The TS side uses inline object type `{agent: string; secret: string; from: string; to: string}` while Python uses a separate `CollapsedBelief` dataclass with fields `agent`, `secret`, `from_state`, `to_state`. The JSON serialization uses `from` and `to` (see `events.py:100`), matching the TS field names `from` and `to`. Functionally equivalent at the wire level.

### 2.5 GoalVector: MATCH

All 7 fields match: safety, status, closeness, secrecy, truth_seeking, autonomy, loyalty.
- TS (`agents.ts:22-30`): number types + Record<string,number> for closeness
- Python (`agents.py:49-57`): float types + dict[str,float] for closeness

### 2.6 RelationshipState: MATCH

All 3 fields match: trust, affection, obligation.
- TS (`agents.ts:32-36`) vs Python (`agents.py:83-86`)

### 2.7 PacingState: MATCH

All 6 fields match: dramatic_budget, stress, composure, commitment, recovery_timer, suppression_count.
- TS (`agents.ts:38-45`) vs Python (`agents.py:101-108`)

### 2.8 AgentState: MATCH

All 10 fields match between TS (`agents.ts:63-78`) and Python (`agents.py:139-154`).

### 2.9 Location: MATCH

All 8 fields match between TS (`world.ts:1-10`) and Python (`world.py:7-16`).

### 2.10 SecretDefinition: MATCH

All 10 fields match between TS (`world.ts:12-23`) and Python (`world.py:44-55`).

### 2.11 WorldDefinition: MATCH

All 12 fields match between TS (`world.ts:25-42`) and Python (`world.py:91-108`).

### 2.12 Scene: MATCH

All 12 fields match between TS (`scenes.ts:14-30`) and Python (`scenes.py:10-26`).

### 2.13 SnapshotState: MATCH

All 9 fields match between TS (`scenes.ts:32-45`) and Python (`scenes.py:66-79`).

---

## 3. NarrativeFieldPayload vs bundler.py Output

TS payload interface (`src/visualization/src/types/payload.ts:37-46`):
```
format_version: string
metadata: SimulationMetadata
agents: AgentManifest[]
locations: LocationDefinition[]
secrets: SecretDefinitionPayload[]
events: Event[]
scenes: Scene[]
belief_snapshots: BeliefSnapshot[]
```

Python bundler output (`src/engine/narrativefield/integration/bundler.py:108-117`):
```python
{
    "format_version": FORMAT_VERSION,
    "metadata": metadata,
    "agents": agents_manifest,
    "locations": [loc.to_dict() ...],
    "secrets": [sec.to_dict() ...],
    "events": [e.to_dict() ...],
    "scenes": [s.to_dict() ...],
    "belief_snapshots": belief_snapshots,
}
```

**Result:** All 8 top-level keys match.

**Finding F-04 (MEDIUM):** The bundler adds `raw_event_count` to metadata (`bundler.py:103-104`), which is not declared in the TS `SimulationMetadata` interface (`payload.ts:6-15`). The extra field would be silently ignored by TypeScript but is not available for frontend consumption.

**Finding F-05 (LOW):** `SimulationMetadata` in TS (`payload.ts:6-15`) has exactly 8 fields: simulation_id, scenario, total_ticks, total_sim_time, agent_count, event_count, snapshot_interval, timestamp. The bundler constructs metadata from `inputs.metadata` dict directly (`bundler.py:102`), which could contain additional fields beyond these 8. The TS side will silently drop them since it's a typed interface.

### 3.1 AgentManifest Cross-Check

| TS (`payload.ts:17-23`) | Python bundler (`bundler.py:89-97`) | Match? |
|---|---|---|
| id | id | YES |
| name | name | YES |
| initial_location | initial_location (from `agent.location`) | YES |
| goal_summary | goal_summary (from `_summarize_goals()`) | YES |
| primary_flaw | primary_flaw (from `_primary_flaw()`) | YES |

**Result:** Exact match.

### 3.2 BeliefSnapshot Cross-Check

| TS (`payload.ts:25-31`) | Python bundler (`bundler.py:73-81`) | Match? |
|---|---|---|
| tick_id | tick_id | YES |
| sim_time | sim_time | YES |
| beliefs | beliefs | YES |
| agent_irony | agent_irony | YES |
| scene_irony | scene_irony | YES |

**Result:** Exact match.

---

## 4. Payload Validation Analysis

File: `src/visualization/src/data/validatePayload.ts`

### What it validates:
1. `format_version` must be "1.0.0" (line 54)
2. `metadata.event_count` must match `events.length` (line 58)
3. `metadata.agent_count` must match `agents.length` (line 63)
4. Secret `holder` must be non-empty string[] (line 75)
5. Event `source_agent` and `target_agents` must reference valid agent IDs (lines 83-86)
6. Tension components: all 8 keys must be numbers in [0,1] (lines 89-99)
7. Causal links must reference valid event IDs, no self-references (lines 102-109)
8. Scene event_ids must not overlap and must cover all events (lines 113-124)
9. Belief snapshots: sim_time must be number, tick_id must be number, all agent keys and secret keys must be valid, all belief values must be valid BeliefState (lines 127-142)

### What it does NOT validate:

**Finding F-06 (HIGH):** The validator does NOT validate individual event fields beyond source_agent, target_agents, tension_components, and causal_links. Specifically:
- No validation that `event.type` is a valid EventType enum value
- No validation that `event.id` is non-empty (only basic truthy check at line 82)
- No validation that `event.sim_time`, `tick_id` are valid numbers
- No validation that `event.description` is a string
- No validation that `event.metrics.tension`, `irony`, `significance` are valid numbers
- No validation of `event.deltas` structure (kind, op, value types)
- No validation of `event.beat_type` against BeatType enum

**Finding F-07 (HIGH):** The `parseNarrativeFieldPayload` function (`src/visualization/src/data/loader.ts:23-69`) validates the top-level shape (metadata is object, arrays are arrays) and parses metadata fields strictly, but then **casts arrays directly** via `as unknown as NarrativeFieldPayload['events']` (line 65). This means event objects, scene objects, location objects, secret objects, agent objects, and belief snapshots are **not validated at all** during parse. Malformed individual items will only be caught at runtime when accessed.

### validatePayload.ts is a CLI script, not runtime validation

**Finding F-08 (MEDIUM):** `validatePayload.ts` is a standalone CLI script (it has a `main()` function and calls `process.exit(1)` on failure). It is NOT called during `loadEventLog` in the store. The store's `loadEventLog` (`narrativeFieldStore.ts:177`) calls `parseNarrativeFieldPayload()` which only checks top-level structure. There is no runtime validation when loading data through the UI file picker.

---

## 5. Store Analysis

File: `src/visualization/src/store/narrativeFieldStore.ts`

**Finding F-09 (LOW):** In `loadEventLog` (line 180-183), `minTime` is computed using `reduce` starting from `Number.POSITIVE_INFINITY`. If `payload.events` is empty, `minTime` remains `POSITIVE_INFINITY`. The code handles this at line 193 (`Number.isFinite(minTime) ? minTime : 0`), but an empty events array would result in `timeDomain: [0, 1]` and `computedTension` as an empty Map. This is a reasonable default but no warning is given to the user.

**Finding F-10 (LOW):** The `useVisibleEvents` selector (line 336-352) creates a new filtered array on every render cycle since it uses `filter()`. With hundreds of events this is fine, but for larger datasets this could cause unnecessary re-renders. Not a current issue for the MVP scope of ~100-200 events.

---

## 6. React Component Review

### 6.1 InfoPanel (`src/visualization/src/components/InfoPanel.tsx`)

- Key props: Uses `key={idx}` for deltas list (line 73). Since deltas are an ordered list with no natural unique key, index is acceptable here.
- Loading states: Shows "Click an event to see details." when no event/agent selected (line 110). No loading spinner needed since data is synchronous from store.
- No issues found.

### 6.2 TimelineBar (`src/visualization/src/components/TimelineBar.tsx`)

- Key props: Uses `key={s.id}` for scenes (line 42). Correct -- scene IDs are unique.
- Potential division by zero: `(domain.max - domain.min)` used as denominator (lines 37-38). If `domain.max === domain.min`, this divides by zero (domain is guarded by `Math.max(min + 1e-6, max)` at line 13, so this is safe for events.length > 0). For empty events, domain is `{min: 0, max: 1}` (line 11), also safe.
- No issues found.

### 6.3 CanvasRenderer (`src/visualization/src/components/CanvasRenderer.tsx`)

**Finding F-11 (MEDIUM):** The `drawModel` useMemo (line 100-153) calls `useNarrativeFieldStore.getState().events` directly (line 106) instead of using the `allEvents` variable from the hook. This is functionally correct but bypasses React's dependency tracking -- if `allEvents` changes, the memo would recompute anyway since `events` (the filtered set) is in the dependency array. However, `allEvents` is NOT in the dependency array (line 142-153), so if `allEvents` changed without `events` changing, the layout would use stale data. In practice this is unlikely since any change to `allEvents` triggers `events` to recompute, but it's a correctness concern.

- Key props: No mapped lists in JSX that would need keys (tooltip and selection box are single elements).
- The cleanup logic at line 97 (`host.innerHTML = ''`) is appropriate for canvas element cleanup.

### 6.4 BeatSheetPanel (`src/visualization/src/components/BeatSheetPanel.tsx`)

- Key props: Uses `key={e.id}` for event cards (line 184). Correct -- event IDs are unique.
- Handles empty state: Returns `null` if no region selected (line 130).
- Loading state: Shows "Extracting..." during fetch (line 240).
- Error state: Displays extraction errors (lines 251-256).
- No issues found.

### 6.5 ControlPanel (`src/visualization/src/components/ControlPanel.tsx`)

- Key props: Uses `key={key}` for weight sliders (line 39), `key={a.id}` for agents (line 99), `key={t}` for event types (line 123). All correct.
- File upload handler properly resets input value (line 79).
- No issues found.

---

## 7. Data Flow / Loader Analysis

### 7.1 causalIndex.ts

- BFS depth-3 matches the spec (Decision 15: precomputed hover neighborhoods, BFS depth-3).
- Uses `as unknown as CausalIndex` cast (line 51) to satisfy interface. The Map structurally satisfies the `CausalIndex.get()` method signature. Acceptable.

### 7.2 tensionComputer.ts

- Weight normalization: Divides by totalWeight (line 34). If all weights are 0, returns 0 (line 32). Correct.
- Clamps to [0,1] (line 35). Correct.
- No issues found.

---

## 8. Cross-System Consistency: TensionWeights

| TS (renderer.ts:14-23) | Python (metrics/tension.py:31-40) |
|---|---|
| danger: number | danger: float = 0.125 |
| time_pressure: number | time_pressure: float = 0.125 |
| goal_frustration: number | goal_frustration: float = 0.125 |
| relationship_volatility: number | relationship_volatility: float = 0.125 |
| information_gap: number | information_gap: float = 0.125 |
| resource_scarcity: number | resource_scarcity: float = 0.125 |
| moral_cost: number | moral_cost: float = 0.125 |
| irony_density: number | irony_density: float = 0.125 |

**Result:** Same 8 fields. Python defaults to 0.125 (equal weight summing to 1.0), TS defaults to 1.0 per key in `tensionPresets.ts:5-14` (normalized at compute time).

**Finding F-12 (LOW):** Default weight semantics differ. Python defaults each to 0.125 (1/8), TS defaults each to 1.0. Both normalize before computing, so the mathematical result is the same (equal weighting). However, the displayed slider values differ: TS shows 1.0, Python would show 0.125. This could be confusing if comparing weight configs between systems.

---

## 9. IndexTables Cross-Check

| TS (`scenes.ts:47-56`) EventIndices | Python (`index_tables.py:10-22`) IndexTables |
|---|---|
| events: Record<string, Event> | event_by_id: dict[str, Event] |
| agent_timeline: Record<string, string[]> | events_by_agent: dict[str, list[str]] |
| location_events: Record<string, string[]> | events_by_location: dict[str, list[str]] |
| event_participants: Record<string, string[]> | participants_by_event: dict[str, list[str]] |
| secret_events: Record<string, string[]> | events_by_secret: dict[str, list[str]] |
| pair_interactions: Record<string, string[]> | events_by_pair: dict[tuple[str,str], list[str]] |
| forward_links: Record<string, string[]> | forward_links: dict[str, list[str]] |
| backward_links: Record<string, string[]> | (not in Python IndexTables) |

**Finding F-13 (HIGH):** The TS `EventIndices` interface has a `backward_links` field (`scenes.ts:55`) that does not exist in the Python `IndexTables` dataclass. If `EventIndices` is ever populated from Python bundler output, this field would be missing. Additionally, the Python `events_by_pair` uses `tuple[str,str]` keys which cannot be directly serialized to JSON (JSON keys must be strings). However, note that `EventIndices` is not currently used in the payload or bundler -- it's a TS-only convenience type. The actual causal index is built client-side in `causalIndex.ts`. This finding applies only if `EventIndices` is ever serialized across the boundary.

---

## 10. Minor / Informational Findings

**Finding F-14 (LOW):** The `EmotionalState` interface (`agents.ts:54-61`) defines specific named emotion fields (anger, fear, hope, shame, affection, suspicion), but `AgentState.emotional_state` is typed as `Record<string, number>` (`agents.ts:71`). The Python side also uses `dict[str, float]` (`agents.py:148`). The named fields in `EmotionalState` are never actually used by any code -- they appear to be a reference definition. Not a bug, but the unused interface could be confusing.

**Finding F-15 (MEDIUM):** `BeatSheetPanel.tsx` uses `BeatType` as a string union for the `beatColor` function (line 7-21). The switch cases use raw string values (`'setup'`, `'complication'`, etc.) rather than the `BeatType` enum members. While TypeScript allows this since BeatType enum values are these strings, it means the switch is not exhaustive-checked by the compiler. If a new BeatType value is added, no compile error would result -- just a fallback to the default gray color.

---

## Findings Summary

| ID | Severity | Area | Description | Files |
|---|---|---|---|---|
| F-01 | MEDIUM | SceneType enum | TS has SceneType enum, Python uses plain string | `types/scenes.ts:5-12`, `schema/scenes.py:25` |
| F-02 | MEDIUM | tension_components type | TS uses typed TensionComponents, Python uses dict[str,float] | `types/events.ts:64`, `schema/events.py:147` |
| F-03 | LOW | CollapsedBelief | TS inline type vs Python named dataclass (wire-compatible) | `types/events.ts:55`, `schema/events.py:92-109` |
| F-04 | MEDIUM | raw_event_count | Bundler adds undeclared field to metadata | `integration/bundler.py:103-104`, `types/payload.ts:6-15` |
| F-05 | LOW | SimulationMetadata | Bundler may pass extra fields not in TS interface | `integration/bundler.py:102`, `types/payload.ts:6-15` |
| F-06 | HIGH | Validator gaps | validatePayload.ts does not check event.type, numbers, deltas, beat_type | `data/validatePayload.ts:81-110` |
| F-07 | HIGH | Loader unsafe casts | parseNarrativeFieldPayload casts arrays without per-item validation | `data/loader.ts:62-67` |
| F-08 | MEDIUM | No runtime validation | validatePayload.ts is CLI-only, not called during UI data loading | `data/validatePayload.ts:42`, `store/narrativeFieldStore.ts:177` |
| F-09 | LOW | Empty events edge case | loadEventLog handles empty events without user warning | `store/narrativeFieldStore.ts:180-195` |
| F-10 | LOW | Selector perf | useVisibleEvents creates new array each render (OK for MVP scale) | `store/narrativeFieldStore.ts:336-352` |
| F-11 | MEDIUM | useMemo dependency | drawModel accesses getState() bypassing dependency array | `components/CanvasRenderer.tsx:106` |
| F-12 | LOW | Default weights | TS defaults to 1.0/key, Python to 0.125/key (math equivalent) | `constants/tensionPresets.ts:5-14`, `metrics/tension.py:33-40` |
| F-13 | HIGH | EventIndices mismatch | TS has backward_links not in Python; pair key types differ | `types/scenes.ts:55`, `schema/index_tables.py:10-22` |
| F-14 | LOW | EmotionalState unused | Named emotion interface defined but unused (AgentState uses Record) | `types/agents.ts:54-61` vs `agents.ts:71` |
| F-15 | MEDIUM | BeatType string switch | beatColor uses string literals, not enum refs; not exhaustive-checked | `components/BeatSheetPanel.tsx:8-21` |
