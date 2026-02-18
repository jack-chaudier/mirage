# Appendix C: Data Schemas

> Annotated schemas and the metrics-vs-storyteller delta table.

## C.1 Event Schema

Source: `events.py:177-234`, `Event` dataclass.

```json
{
  "id": "EVT_042",                        // Unique event ID
  "sim_time": 23.5,                       // Simulation time in minutes
  "tick_id": 47,                          // Which tick generated this event
  "order_in_tick": 0,                     // Ordering within the tick
  "type": "reveal",                       // EventType enum value (see below)
  "source_agent": "elena",               // Agent who initiated the event
  "target_agents": ["thorne"],            // Agents targeted by the event
  "location_id": "balcony",              // Where the event occurs
  "causal_links": ["EVT_039", "EVT_041"],// Events that causally precede this one
  "deltas": [                             // State changes caused by this event
    {
      "kind": "belief",                   // DeltaKind enum (see C.3)
      "agent": "thorne",
      "agent_b": null,
      "attribute": "secret_affair_01",
      "op": "set",
      "value": "believes_true",
      "reason_code": "REVEAL_DIRECT",
      "reason_display": "Elena reveals the affair to Thorne."
    },
    {
      "kind": "relationship",
      "agent": "thorne",
      "agent_b": "elena",
      "attribute": "trust",
      "op": "add",
      "value": -0.4,
      "reason_code": "BETRAYAL_LEARNED",
      "reason_display": "Thorne's trust in Elena collapses."
    }
  ],
  "description": "Elena confesses the affair to Thorne on the balcony.",
  "dialogue": "I can't keep lying to you, James. Not anymore.",
  "content_metadata": {                   // Optional: type-specific metadata
    "secret_id": "secret_affair_01",
    "public_statement": false
  },
  "beat_type": "turning_point",           // BeatType enum or null (see C.4)
  "metrics": {                            // EventMetrics, populated by pipeline
    "tension": 0.72,
    "irony": 1.85,
    "significance": 0.0,                  // Not yet computed (defaults to 0.0)
    "thematic_shift": {
      "truth_deception": 0.2,
      "loyalty_betrayal": -0.1
    },
    "tension_components": {
      "danger": 0.7,
      "time_pressure": 0.45,
      "goal_frustration": 0.6,
      "relationship_volatility": 0.85,
      "information_gap": 0.3,
      "resource_scarcity": 0.5,
      "moral_cost": 0.75,
      "irony_density": 0.92
    },
    "irony_collapse": {                   // null if no collapse
      "detected": true,
      "drop": 1.2,
      "collapsed_beliefs": [
        {
          "agent": "thorne",
          "secret": "secret_affair_01",
          "from": "unknown",
          "to": "believes_true"
        }
      ],
      "score": 1.2
    }
  }
}
```

## C.2 Agent State Schema

Source: `agents.py`, `AgentState` dataclass.

```json
{
  "id": "thorne",
  "name": "James Thorne",
  "location": "dining_table",
  "goals": {
    "safety": 0.4,
    "status": 0.9,
    "closeness": {"elena": 0.7, "marcus": 0.6},
    "secrecy": 0.3,
    "truth_seeking": 0.6,
    "autonomy": 0.7,
    "loyalty": 0.8
  },
  "flaws": [
    {
      "flaw_type": "pride",
      "strength": 0.8,
      "trigger": "status_threat",
      "effect": "overweight_status",
      "description": "..."
    }
  ],
  "pacing": {
    "dramatic_budget": 1.0,
    "stress": 0.1,
    "composure": 0.9,
    "commitment": 0.0,
    "recovery_timer": 0,
    "suppression_count": 0
  },
  "emotional_state": {
    "anger": 0.0,
    "fear": 0.0,
    "hope": 0.3,
    "shame": 0.0,
    "affection": 0.4,
    "suspicion": 0.1
  },
  "relationships": {
    "elena": {"trust": 0.8, "affection": 0.6, "obligation": 0.3},
    "marcus": {"trust": 0.7, "affection": 0.4, "obligation": 0.2}
  },
  "beliefs": {
    "secret_affair_01": "unknown",
    "secret_embezzle_01": "unknown"
  },
  "alcohol_level": 0.0,
  "commitments": []
}
```

### Belief States (`agents.py`)

| Value | Meaning |
|-------|---------|
| `unknown` | No information about this secret |
| `suspects` | Has reason to believe but not confirmed |
| `believes_true` | Believes the secret is true |
| `believes_false` | Believes the secret is false |

## C.3 Enum Reference

### EventType (`events.py:8-18`)

| Value | Description |
|-------|-------------|
| `chat` | Conversation between agents |
| `observe` | Agent observes/overhears something |
| `social_move` | Agent changes location |
| `reveal` | Agent reveals a secret |
| `conflict` | Direct confrontation |
| `internal` | Internal monologue/realization |
| `physical` | Physical action (pour drink, etc.) |
| `confide` | Agent confides in another |
| `lie` | Agent deliberately deceives |
| `catastrophe` | Emotional breaking point |

### BeatType (`events.py:21-26`)

| Value | Description |
|-------|-------------|
| `setup` | Establishing scene/characters |
| `complication` | New tension introduced |
| `escalation` | Existing tension increases |
| `turning_point` | Major shift in dynamics |
| `consequence` | Fallout from turning point |

### DeltaKind (`events.py:29-38`)

| Value | Description |
|-------|-------------|
| `agent_emotion` | Change to emotional_state dict |
| `agent_resource` | Change to alcohol_level etc. |
| `agent_location` | Agent moves to new location |
| `relationship` | Change to trust/affection/obligation |
| `belief` | Change to belief about a secret |
| `secret_state` | Change to secret's revealed status |
| `world_resource` | Change to world-level resource |
| `commitment` | Agent gains a commitment |
| `pacing` | Change to pacing state variable |

## C.4 SimulationOutput Format

Top-level JSON produced by `run_simulation` + `run.py`:

```json
{
  "format_version": "1.0",
  "metadata": {
    "scenario": "dinner_party",
    "seed": 42,
    "event_count": 156,
    "total_ticks": 78,
    "total_sim_time": 150.0,
    "time_scale": 4.0,
    "python_version": "3.14.2",
    "git_commit": "abc1234",
    "config_hash": "sha256:..."
  },
  "initial_state": {
    "agents": { ... },
    "world_definition": { ... }
  },
  "snapshots": [
    {
      "snapshot_id": "snap_0001",
      "tick_id": 20,
      "sim_time": 10.0,
      "event_count": 40,
      "agents": { ... },
      "tension_proxy": 0.35
    }
  ],
  "events": [ ... ],
  "secrets": [ ... ],
  "locations": [ ... ]
}
```

## C.5 NarrativeFieldPayload (Visualization)

Produced by `pipeline.py` → `bundle_for_renderer()`, consumed by the TypeScript visualization:

```json
{
  "metadata": { ... },
  "agents": [
    {"id": "thorne", "name": "James Thorne", "color": "#E69F00"}
  ],
  "locations": [ ... ],
  "secrets": [ ... ],
  "events": [ ... ],
  "scenes": [
    {
      "scene_id": "scene_0",
      "event_ids": ["EVT_001", "EVT_002", ...],
      "time_start": 0.0,
      "time_end": 12.5,
      "location": "dining_table",
      "participants": ["thorne", "elena", "marcus"]
    }
  ],
  "belief_snapshots": [ ... ]
}
```

## C.6 Delta Table: Metrics Segmentation vs Storyteller Scene Splitting

Two independent segmentation systems exist with different thresholds:

| Parameter | Metrics Segmentation | Storyteller Scene Splitter |
|-----------|---------------------|---------------------------|
| **Source** | `segmentation.py:25-38` | `scene_splitter.py:17-34` |
| **Purpose** | Scene boundaries for viz/analysis | Scene chunks for prose generation |
| **Input** | Bundled events (post-pipeline) | Arc events (pre-prose) |
| **Jaccard threshold** | < 0.3 | < 0.5 |
| **Time gap** | > 5.0 minutes | > 10.0 minutes |
| **Tension rule** | Valley below peak×0.3 sustained for 3 events | Not used |
| **Beat transition** | Not used | TURNING_POINT → CONSEQUENCE |
| **Min scene size** | 3 events | 3 events |
| **Midpoint fallback** | No | Yes (when chunk ≥ 2× target and no natural boundaries) |
| **Target chunk size** | N/A (variable) | 10 events (aim for 8-12) |

The metrics segmentation is stricter on participant similarity (Jaccard 0.3 vs 0.5) and more lenient on time gaps (5 min vs 10 min). The storyteller splitter adds beat-type awareness and a midpoint fallback for oversized chunks.
