# Simulation Engine Audit Findings

> **Auditor:** sim-auditor
> **Date:** 2026-02-08
> **Scope:** tick_loop.py, decision_engine.py, pacing.py, run.py, types.py, scenarios/dinner_party.py, schema/*
> **Method:** Multi-seed stress test (seeds 1-10), invariant verification, spec compliance, edge cases, previous audit status

---

## Executive Summary

The simulation engine is **structurally sound**. Ten seeds ran to completion with zero invariant violations, all pacing constants match the spec, catastrophe gates work correctly, and trust repair hysteresis is properly applied. All 10 EventTypes were exercised across seeds. However, the audit identified **3 HIGH**, **4 MEDIUM**, and **3 LOW** issues.

| Severity | Count |
|----------|-------|
| HIGH     | 3     |
| MEDIUM   | 4     |
| LOW      | 3     |

---

## Multi-Seed Stress Test Results

All 10 seeds terminated via `event_limit` (200 events) within 26-29 sim_time units and 32-33 ticks.

| Seed | Events | Ticks | SimTime | Catastrophes | Secrets | SM% | Dramatic% |
|------|--------|-------|---------|--------------|---------|-----|-----------|
| 1    | 200    | 32    | 27.5    | 1            | 11      | 46% | 14%       |
| 2    | 200    | 33    | 28.5    | 1            | 12      | 40% | 14%       |
| 3    | 200    | 32    | 27.5    | 1            | 15      | 38% | 17%       |
| 4    | 200    | 32    | 27.5    | 0            | 10      | 42% | 14%       |
| 5    | 200    | 33    | 28.0    | 0            | 12      | 44% | 14%       |
| 6    | 200    | 33    | 26.5    | 0            | 11      | 37% | 10%       |
| 7    | 200    | 32    | 27.0    | 1            | 15      | 42% | 14%       |
| 8    | 200    | 33    | 28.2    | 1            | 11      | 47% | 12%       |
| 9    | 200    | 33    | 28.2    | 1            | 11      | 46% | 13%       |
| 10   | 200    | 32    | 26.8    | 0            | 11      | 47% | 12%       |

**Key observations:**
- Catastrophes occurred in 6/10 seeds (always exactly 1 per run)
- SOCIAL_MOVE dominates: 37-47% of all events
- Dramatic events (CONFLICT, REVEAL, CONFIDE, LIE, CATASTROPHE) are 10-17% of events
- All runs terminate at event_limit; no seed reaches sim_time limit or tick_limit
- Secret discovery ranges from 10-15 across seeds (good variability)
- No departures occur (no agent ever leaves the party)
- Bathroom is never used as a location (move scoring disincentivizes it effectively)
- Dining table is the primary location (60-80% of events), balcony is secondary

---

## Invariant Verification

All invariants passed across all 10 seeds:

- [PASS] Monotonic (tick_id, order_in_tick) ordering
- [PASS] Causal links reference preceding events
- [PASS] All locations valid
- [PASS] No self-targeting events
- [PASS] Valid agent IDs in all events
- [PASS] Relationship trust/affection in [-1,1], obligation in [0,1]
- [PASS] Pacing bounds: stress [0,1], composure [0.05,1], budget [0,1], commitment [0,1]
- [PASS] Alcohol level in [0,1]
- [PASS] All beliefs are valid BeliefState enum values
- [PASS] All Event.metrics are typed EventMetrics instances
- [PASS] Bathroom capacity never exceeded
- [PASS] No non-adjacent moves

---

## Findings

### HIGH-1: SOCIAL_MOVE Dominance (37-47% of events)

**Files:** `decision_engine.py:755-773`, `tick_loop.py:223-259`
**Severity:** HIGH

SOCIAL_MOVE events consume 37-47% of all event slots across every seed. This means nearly half the simulation's event budget is spent on movement rather than meaningful character interactions. The event_limit of 200 effectively becomes ~110 non-movement events.

**Root cause analysis:**
1. Each agent generates one SOCIAL_MOVE candidate per adjacent location (up to 3 at dining table)
2. The softmax sampling in `choose_actions` (`tick_loop.py:240-258`) penalizes INTERNAL (-0.35) and OBSERVE (-0.10) but does NOT penalize SOCIAL_MOVE
3. SOCIAL_MOVE is non-dramatic, so it avoids masking and budget penalties
4. The recency penalty (`decision_engine.py:644-659`) adds extra penalty for moving within 2 ticks, but agents can still move every other tick

**Impact:** Dilutes narrative density. A reader/viewer sees "Marcus moves to balcony, Elena moves to kitchen, Victor moves to balcony, Diana moves to dining table" as filler rather than story.

**Recommendation:** Add a SOCIAL_MOVE penalty (e.g., -0.15) in `choose_actions` similar to the INTERNAL penalty, or reduce `max_actions_per_tick` from 6 to 4-5 and filter excess moves.

---

### HIGH-2: Foyer Privacy Mismatch (Code: 0.6, Spec: 0.2)

**Files:** `scenarios/dinner_party.py:49`, spec: `pacing-physics.md:581`
**Severity:** HIGH

The dinner party config sets foyer privacy to 0.6 (semi-private), but `pacing-physics.md` Section 7.1 explicitly states foyer privacy = 0.2 (semi-public, masking enforced).

**Impact:** With privacy=0.6, the foyer is treated as a semi-private space where:
- Budget recharge gets the private bonus (+0.04)
- Stress decay gets the private bonus (-0.02)
- Masking is NOT enforced (privacy >= 0.3 threshold)
- Composure recovery is allowed

This makes the foyer a more attractive escape destination than the spec intended. The spec designed the foyer as semi-public to keep masking active there.

**Recommendation:** Change foyer privacy to 0.2 in `dinner_party.py:49` or update the spec if the deviation is intentional. Note that `decision_engine.py:341-343` already caps effective foyer privacy at 0.35 for move scoring, suggesting the developer was aware of the tension.

---

### HIGH-3: Snapshot global_tension Always Zero

**Files:** `tick_loop.py:921-928,948-955`
**Severity:** HIGH

All snapshots set `global_tension: 0.0` as a hardcoded placeholder. The metrics pipeline computes tension post-hoc, but the simulation never populates this field in its own snapshots.

**Impact:** Any downstream consumer expecting global_tension from simulation snapshots will see zero. The field exists in `SnapshotState` (`scenes.py:77`) but is never populated. Additionally, `run_simulation` returns plain dicts, not typed `SnapshotState` objects.

**Recommendation:** Either (a) remove global_tension from simulation snapshots and compute it exclusively in metrics, or (b) compute a simple proxy (e.g., mean stress across agents) to populate the field.

---

### MEDIUM-1: Event Immutability Violations (S-28 Still Present)

**Files:** `tick_loop.py:685,718` (witness events), `tick_loop.py:852` (pacing deltas)
**Severity:** MEDIUM

Two mutation patterns exist:

1. **Witness event causal_links overwritten** (`tick_loop.py:685,718`): After `action_to_event` creates an OBSERVE event with causal links from `_find_causal_links`, the witness generation code overwrites `obs.causal_links = [event.id]`. The Event dataclass is not frozen, so this silently works, but it means the original causal links are discarded.

2. **Pacing deltas appended post-creation** (`tick_loop.py:852`): `apply_tick_updates` extends `target_event.deltas` after the event is already in the event log. Any consumer that processes events in order may see different delta counts depending on when they read.

**Impact:** Not a correctness issue in current code (events are only consumed after the tick completes), but breaks the conceptual contract that events are immutable after creation. Would cause bugs if events are ever streamed to consumers during a tick.

---

### MEDIUM-2: Fragile String Matching for PHYSICAL Actions (S-27 Still Present)

**Files:** `tick_loop.py:464-492`, `decision_engine.py:138-143,527-532`
**Severity:** MEDIUM

PHYSICAL action effects are determined by string matching on `action.content`:
```python
if "pour" in content or "refill" in content or "serve" in content:
    # ...alcohol for multiple agents
if "drink" in content:
    # ...alcohol for self
```

**Impact:** Any content text containing these substrings (e.g., "Think about pouring a drink") would unintentionally trigger alcohol deltas. Content text is developer-controlled in the current codebase, so this is low-risk now, but would break if content becomes LLM-generated or user-provided.

**Recommendation:** Use structured `content_metadata` (e.g., `{"physical_subtype": "pour_wine"}`) instead of string matching on content.

---

### MEDIUM-3: Catastrophe Ordering Contradicts Spec

**Files:** `tick_loop.py:868-872`, spec: `pacing-physics.md:400`
**Severity:** MEDIUM

The spec says catastrophes resolve LAST in a tick (`order_in_tick=CATASTROPHE_ORDER`), but `execute_tick` processes catastrophes FIRST (order starts at 0, catastrophes get 0..N, then actions get N+1..M).

**Impact:** The current ordering (catastrophes first) is actually better for the chain-reaction mechanic described in `pacing-physics.md` Section 13.1, since bystander stress from catastrophes is applied via `apply_tick_updates` AFTER all events are generated. But it contradicts the spec comment. The ordering within a tick is mostly semantic since all events share the same `tick_id`.

**Recommendation:** Update the spec comment to match implementation, since the current behavior is more correct for the desired mechanics.

---

### MEDIUM-4: self_sacrifice Flaw Effect Includes CHAT (Spec Deviation)

**Files:** `decision_engine.py:453`, spec: `decision-engine.md:540-542`
**Severity:** MEDIUM

The spec says `self_sacrifice` applies to `(CONFIDE, REVEAL)`. The code also includes `CHAT`:
```python
case "self_sacrifice":
    if action.action_type in (EventType.CONFIDE, EventType.REVEAL, EventType.CHAT):
        return strength * 0.3
```

**Impact:** Elena and Diana (who have guilt flaws with self_sacrifice effect) get a +0.21 to +0.18 bonus on CHAT actions when their trigger is active. This makes them chattier than the spec intends, pulling them toward social accommodation behavior. This is arguably a reasonable extension (guilt-driven over-accommodation), but it deviates from spec.

---

### LOW-1: run_simulation Produces Dict Snapshots, Not Typed SnapshotState

**Files:** `tick_loop.py:921-956`
**Severity:** LOW

Despite `SnapshotState` existing in `schema/scenes.py`, `run_simulation` returns snapshots as plain dictionaries. The dict has keys `{tick_id, sim_time, agents, global_tension}` which is a subset of `SnapshotState`'s fields.

---

### LOW-2: No Termination Variety Across Seeds

**Files:** `tick_loop.py:891-908`
**Severity:** LOW

All 10 seeds terminate via `event_limit` (200 events). None reach `max_sim_time` (150.0), `tick_limit` (300), stalemate, or agent departure conditions. Sim_time only reaches ~27-29 minutes out of 150. This suggests:
- The event_limit is the binding constraint
- The sim_time/tick_limit values are never reached under default config
- The stalemate condition (all agents in recovery + low stress) never triggers naturally

This means the other termination conditions are essentially dead code in the default configuration.

---

### LOW-3: Pacing Update Order in Spec vs Code (Non-Issue)

**Files:** `pacing.py:431-438`, spec: `pacing-physics.md:619-648`
**Severity:** LOW (Informational)

The spec says update order should be `stress -> composure -> commitment -> budget -> recovery -> suppression`. The code computes in order `budget, stress, composure, commitment, recovery, suppression`. Since all update functions read from the OLD `agent.pacing` state (not the newly computed values), the ordering difference has no functional impact. The code correctly computes all values from the pre-tick state.

---

## Previous Audit Findings Status

| Finding | Status | Notes |
|---------|--------|-------|
| S-01 (dominant_flaw.name -> .flaw_type) | **FIXED** | `tick_loop.py:85` uses `dominant.flaw_type.value` correctly |
| S-02 (danger secret lookup) | N/A | Metrics pipeline issue, not in simulation engine scope |
| S-03+S-15 (EventMetrics typed dataclass) | **FIXED** | `events.py:140-174` has full typed `EventMetrics` with `tension_components` and `irony_collapse` |
| S-04+S-14 (WorldStateSnapshot naming) | **PARTIAL** | `SnapshotState` exists in `scenes.py` but `run_simulation` still returns plain dicts |
| S-05 (content_display -> description) | **FIXED** | `Event.description` field used consistently |
| S-06+S-18 (content_metadata on Event) | **FIXED** | `events.py:193` has `content_metadata: Optional[dict[str, Any]] = None` |
| S-27 (fragile string matching) | **PRESENT** | Still uses `"pour" in content` pattern in `tick_loop.py:465` |
| S-28 (event mutability) | **PRESENT** | `tick_loop.py:685,852` still mutates events post-creation |

---

## Spec Compliance Summary

| Check | Result |
|-------|--------|
| All 38 PacingConstants match spec values | PASS |
| catastrophe_potential formula: `stress * commitment^2 + suppression_bonus` | PASS |
| Catastrophe gate: potential >= 0.35 AND composure < 0.30 | PASS |
| Recovery timer blocks catastrophe check | PASS |
| Trust repair hysteresis: positive changes divided by 3.0 | PASS |
| Negative trust changes at full strength | PASS |
| GoalVector has 7 dimensions | PASS |
| DRAMATIC_EVENT_TYPES correct set | PASS |
| CONFIDE budget cost = BUDGET_COST_MINOR | PASS |
| Foyer privacy value | **FAIL** (code: 0.6, spec: 0.2) |
| Catastrophe ordering | **FAIL** (code: first, spec: last) |
| self_sacrifice effect scope | **FAIL** (code adds CHAT) |

---

## Edge Case Results

| Test | Result |
|------|--------|
| event_limit=1 | PASS (produces exactly 1 event) |
| max_sim_time=0 | PASS (produces 0 events) |
| tick_limit=0 | PASS (produces 0 events) |
| All agents in recovery + low stress | PASS (terminates via stalemate) |

---

## Audit Script

The programmatic audit script is at `$PROJECT_ROOT/src/engine/audit_sim.py`. Run with:
```bash
cd $PROJECT_ROOT/src/engine
source .venv/bin/activate
python audit_sim.py
```
