# World Schema Specification

> **Status:** CANONICAL — defines locations, secrets, and world-level structures.
> **Implements:** Decision #17 (MVP dinner party setting), doc3.md Location schema, CLAUDE.md location list
> **Consumers:** simulation (tick-loop, decision-engine, dinner-party-config), visualization (renderer, thread-layout), metrics (scene-segmentation)
> **Depends on:** `specs/schema/events.md` (events reference location_id), `specs/schema/agents.md` (agents reference location, beliefs reference secret_id)

---

## Table of Contents

1. [Location](#1-location)
2. [Dinner Party Map](#2-dinner-party-map)
3. [Seating Arrangement](#3-seating-arrangement)
4. [Movement Rules](#4-movement-rules)
5. [SecretDefinition](#5-secretdefinition)
6. [Dinner Party Secrets](#6-dinner-party-secrets)
7. [WorldDefinition](#7-worlddefinition)
8. [JSON Examples](#8-json-examples)
9. [Validation Rules](#9-validation-rules)
10. [NOT In Scope](#10-not-in-scope)

---

## 1. Location

A place where events can occur. Locations form a graph via adjacency. Privacy level determines how likely agents are to engage in dramatic behavior there.

### Python

```python
from dataclasses import dataclass

@dataclass
class Location:
    """A place where events can occur."""
    id: str                          # Unique location ID. E.g. "dining_table"
    name: str                        # Display name. E.g. "Dining Table"
    privacy: float                   # [0.0, 1.0] — 0 = fully public, 1 = fully private
    capacity: int                    # Max agents that can be present simultaneously
    adjacent: list[str]              # IDs of directly connected locations (bidirectional)
    overhear_from: list[str]         # IDs of locations whose conversations can be overheard from here
    overhear_probability: float      # [0.0, 1.0] — probability of overhearing per tick when adjacent conversation occurs
    description: str                 # Human-readable description for tooltips/prose
```

### TypeScript

```typescript
interface Location {
    id: string;
    name: string;
    privacy: number;                 // [0.0, 1.0]
    capacity: number;
    adjacent: string[];
    overhear_from: string[];
    overhear_probability: number;    // [0.0, 1.0]
    description: string;
}
```

### Privacy Levels

Privacy determines how social masking affects agent behavior (from `specs/schema/agents.md` Section 3):

| Privacy | Meaning | Effect |
|---------|---------|--------|
| 0.0 | Fully public | Maximum social masking — agents strongly suppress conflict |
| 0.0 - 0.3 | Semi-public | High suppression — agents maintain polite facades |
| 0.3 - 0.6 | Semi-private | Moderate suppression — some honesty possible |
| 0.6 - 0.8 | Private | Low suppression — agents speak freely |
| 0.8 - 1.0 | Fully private | No suppression — agents express true feelings |

### Overhearing Mechanics

When agents at location A have a conversation (CHAT, CONFIDE, REVEAL, CONFLICT, LIE), agents at any location that lists A in its `overhear_from` may generate an OBSERVE event:

```python
import random

def check_overhear(
    conversation_location: str,
    listener_location: str,
    locations: dict[str, Location]
) -> bool:
    """Check if an agent at listener_location overhears conversation at conversation_location."""
    listener_loc = locations[listener_location]
    if conversation_location in listener_loc.overhear_from:
        return random.random() < listener_loc.overhear_probability
    return False
```

The resulting OBSERVE event has:
- `source_agent`: the listener
- `target_agents`: empty (observation, not interaction)
- `location_id`: the listener's location (where they are, not where the conversation happened)
- Deltas: typically BELIEF changes based on what was overheard
- `causal_links`: links to the event that was overheard

---

## 2. Dinner Party Map

Five locations forming the dinner party setting (from CLAUDE.md).

```
                    +-----------+
                    |  BALCONY  |
                    | (private) |
                    +-----+-----+
                          |
    +-----------+   +-----+-----+   +-----------+
    |  KITCHEN  |---|   DINING   |---|   FOYER   |
    | (semi-prv)|   |   TABLE    |   | (transit) |
    +-----------+   | (public)   |   +-----------+
                    +-----+-----+
                          |
                    +-----+-----+
                    | BATHROOM  |
                    | (private) |
                    +-----------+
```

### Location Definitions

```json
[
    {
        "id": "dining_table",
        "name": "Dining Table",
        "privacy": 0.1,
        "capacity": 6,
        "adjacent": ["kitchen", "balcony", "foyer", "bathroom"],
        "overhear_from": [],
        "overhear_probability": 0.0,
        "description": "The main dining table. Seats six. The social center of the evening — everyone can see and hear everything here."
    },
    {
        "id": "kitchen",
        "name": "Kitchen",
        "privacy": 0.5,
        "capacity": 3,
        "adjacent": ["dining_table"],
        "overhear_from": ["dining_table"],
        "overhear_probability": 0.3,
        "description": "An open-plan kitchen adjacent to the dining area. Semi-private — you can slip away under the pretense of getting more wine, but the dining table is within earshot."
    },
    {
        "id": "balcony",
        "name": "Balcony",
        "privacy": 0.7,
        "capacity": 3,
        "adjacent": ["dining_table"],
        "overhear_from": [],
        "overhear_probability": 0.0,
        "description": "A small balcony off the dining room. Private enough for real conversations. Stepping outside signals you want a moment — or a confrontation away from the group."
    },
    {
        "id": "foyer",
        "name": "Foyer",
        "privacy": 0.2,
        "capacity": 4,
        "adjacent": ["dining_table"],
        "overhear_from": ["dining_table"],
        "overhear_probability": 0.15,
        "description": "The entrance hallway. People pass through here to arrive, leave, or make phone calls. Semi-public — you can catch snippets from the dining room."
    },
    {
        "id": "bathroom",
        "name": "Bathroom",
        "privacy": 0.9,
        "capacity": 2,
        "adjacent": ["dining_table"],
        "overhear_from": [],
        "overhear_probability": 0.0,
        "description": "A small bathroom down the hall from the dining room. The most private space available. Maximum capacity of 2 makes it ideal for intense private conversations — or hiding."
    }
]
```

### Adjacency Matrix

| From \ To | dining_table | kitchen | balcony | foyer | bathroom |
|-----------|:---:|:---:|:---:|:---:|:---:|
| dining_table | - | 1 | 1 | 1 | 1 |
| kitchen | 1 | - | 0 | 0 | 0 |
| balcony | 1 | - | - | 0 | 0 |
| foyer | 1 | 0 | 0 | - | 0 |
| bathroom | 1 | 0 | 0 | 0 | - |

The dining table is the hub. All other locations connect only to the dining table (star topology). This means:
- Moving from kitchen to balcony requires passing through the dining table (2 moves)
- Everyone sees you arrive at / leave the dining table
- The dining table is a bottleneck — you can't avoid the group without being noticed

### Overhear Matrix

| Listener Location | Can Overhear From |
|------------------|-------------------|
| dining_table | (nothing — you're in the middle of it) |
| kitchen | dining_table (30% chance) |
| balcony | (nothing — too far, door closed) |
| foyer | dining_table (15% chance) |
| bathroom | (nothing — soundproofed) |

---

## 3. Seating Arrangement

When at the dining table, agents have assigned seats that affect who they can easily converse with. Adjacent seats enable low-cost CHAT events; non-adjacent seats require "speaking across the table" (slightly higher social cost, fully public).

### Seating Layout

```
        +-------+
        |  (1)  |  ← Head of table (Thorne, the host)
+-------+-------+-------+
|  (6)  |       |  (2)  |
+-------+ TABLE +-------+
|  (5)  |       |  (3)  |
+-------+-------+-------+
        |  (4)  |  ← Foot of table
        +-------+
```

### Default Seat Assignments

| Seat | Agent | Notes |
|------|-------|-------|
| 1 (head) | thorne | Host position. Adjacent to elena (2) and victor (6) |
| 2 | elena | Thorne's wife beside him. Adjacent to thorne (1) and marcus (3) |
| 3 | marcus | Beside Elena (significant). Adjacent to elena (2) and diana (4) |
| 4 (foot) | diana | Elena's friend across from Thorne. Adjacent to marcus (3) and lydia (5) |
| 5 | lydia | The observer. Adjacent to diana (4) and victor (6) |
| 6 | victor | The journalist. Adjacent to lydia (5) and thorne (1) |

### Seating Adjacency

Agents in adjacent seats can have private-ish asides (still at the table, but lower overhear chance for non-adjacent agents). The seating adjacency is stored as:

```python
SEATING_ADJACENCY: dict[str, list[str]] = {
    "thorne": ["elena", "victor"],
    "elena":  ["thorne", "marcus"],
    "marcus": ["elena", "diana"],
    "diana":  ["marcus", "lydia"],
    "lydia":  ["diana", "victor"],
    "victor": ["lydia", "thorne"],
}
```

**Seating effects on interaction:**
- **Adjacent seats:** CHAT events cost 0 dramatic budget. Whispering (quiet CONFIDE) possible with 0.2 overhear probability from non-adjacent agents at the table.
- **Non-adjacent seats:** CHAT events cost 0 dramatic budget but are fully public (all agents at the table are implicit witnesses). Cannot whisper.
- **Seating is strategic:** Elena sits between Thorne and Marcus — she's the bridge between husband and lover. Victor sits next to Thorne — positioned to probe. Lydia is across from Thorne — she can observe but not easily whisper to him.

---

## 4. Movement Rules

### Travel Time

Moving between locations takes time (measured in ticks):

| From → To | Ticks | Notes |
|-----------|-------|-------|
| dining_table ↔ kitchen | 1 | Open-plan, very close |
| dining_table ↔ balcony | 1 | Through a door, but close |
| dining_table ↔ foyer | 1 | Short hallway |
| dining_table ↔ bathroom | 2 | Down the hall, further away |
| kitchen ↔ balcony | 2 | Must pass through dining area |
| kitchen ↔ foyer | 2 | Must pass through dining area |
| kitchen ↔ bathroom | 3 | Must pass through dining area + hallway |
| balcony ↔ foyer | 2 | Must pass through dining area |
| balcony ↔ bathroom | 3 | Must pass through dining area + hallway |
| foyer ↔ bathroom | 3 | Must pass through dining area + hallway |

For MVP simplicity, movement is **instant within adjacent locations** (1 tick) but requires passing through the dining table for non-adjacent locations. During the tick(s) spent in transit through the dining table, the agent IS at the dining table and can be interacted with / observed.

### Movement as a SOCIAL_MOVE Event

When an agent moves:
1. A SOCIAL_MOVE event is generated at the origin location
2. The AGENT_LOCATION delta records the destination
3. If the move passes through the dining table (non-adjacent move), an intermediate tick places the agent at the dining table
4. All agents at the origin see the departure
5. All agents at the destination see the arrival (next tick)

### Interception

Agents cannot be physically intercepted mid-move in MVP. However:
- If agent A is at the dining table when agent B passes through, A can observe B's movement
- If A then generates a SOCIAL_MOVE to follow B, this happens in the next tick (B arrives first)
- The decision engine can score "follow" actions based on whether following an agent would be useful

### Capacity Enforcement

If a location is at capacity when an agent tries to move there:
- The SOCIAL_MOVE action is invalid and the decision engine should not propose it
- If somehow generated (e.g., two agents move simultaneously), the agent with higher `order_in_tick` is redirected back to origin
- The dining table (capacity 6) can always hold all agents

### Social Meaning of Movement

Movement generates narrative signal:
- Leaving the dining table alone → "needs a break" or "avoiding someone"
- Two agents leaving together → "having a private conversation" (other agents notice)
- Following someone who just left → "something is up"
- Going to the bathroom → socially neutral excuse (lowest suspicion)
- Going to the balcony → signals desire for privacy or fresh air

The decision engine should factor in the social signal of movement when scoring SOCIAL_MOVE actions. An agent who needs to have a private conversation should consider the social cost of being seen leaving with someone.

---

## 5. SecretDefinition

A piece of information that exists in the world and can be known, unknown, or misbelieved by agents. Secrets are the primary driver of dramatic irony and information asymmetry.

### Python

```python
@dataclass
class SecretDefinition:
    """A secret that exists in the world. Agents have beliefs about it (see BeliefMatrix)."""
    id: str                          # Unique secret ID. E.g. "secret_affair_01"
    description: str                 # Human-readable description. E.g. "Elena and Marcus are having an affair"
    truth_value: bool                # Ground truth — is this actually true?
    holder: list[str]                # Agent(s) who originated/hold primary knowledge of this secret (e.g. ["elena", "marcus"] for an affair)
    about: str | None                # Which agent(s) this secret is about, if any
    content_type: str                # Category: "affair", "financial", "identity", "investigation", "knowledge"
    initial_knowers: list[str]       # Agent IDs who start knowing this secret (BELIEVES_TRUE)
    initial_suspecters: list[str]    # Agent IDs who start suspecting (SUSPECTS)
    dramatic_weight: float           # [0.0, 1.0] — how dramatically important this secret is
                                     # Used by metrics to weight irony and significance scores
    reveal_consequences: str         # Description of what happens if this secret becomes public
```

### TypeScript

```typescript
interface SecretDefinition {
    id: string;
    description: string;
    truth_value: boolean;
    holder: string[];                // Agent(s) who hold primary knowledge
    about: string | null;
    content_type: string;
    initial_knowers: string[];
    initial_suspecters: string[];
    dramatic_weight: number;         // [0.0, 1.0]
    reveal_consequences: string;
}
```

### Content Types

| Content Type | Description | Typical Dramatic Weight |
|-------------|-------------|------------------------|
| `affair` | Romantic/sexual secret | 0.8-1.0 |
| `financial` | Money, debt, embezzlement | 0.6-0.9 |
| `identity` | Hidden identity, false persona | 0.7-1.0 |
| `investigation` | Someone is investigating someone else | 0.5-0.7 |
| `knowledge` | Someone knows something they shouldn't | 0.3-0.6 |
| `plan` | Secret intention or scheme | 0.5-0.8 |

---

## 6. Dinner Party Secrets

Five secrets create the web of information asymmetry for the dinner party.

```json
[
    {
        "id": "secret_affair_01",
        "description": "Elena and Marcus are having a romantic affair behind Thorne's back.",
        "truth_value": true,
        "holder": ["elena"],
        "about": "elena",
        "content_type": "affair",
        "initial_knowers": ["elena", "marcus", "diana"],
        "initial_suspecters": ["lydia"],
        "dramatic_weight": 1.0,
        "reveal_consequences": "Thorne's marriage and business partnership are both destroyed. The social fabric of the evening tears apart."
    },
    {
        "id": "secret_embezzle_01",
        "description": "Marcus has been embezzling money from the business he shares with Thorne.",
        "truth_value": true,
        "holder": ["marcus"],
        "about": "marcus",
        "content_type": "financial",
        "initial_knowers": ["marcus"],
        "initial_suspecters": ["lydia", "victor"],
        "dramatic_weight": 0.9,
        "reveal_consequences": "Marcus faces criminal charges. Thorne realizes his business partner betrayed him twice. Victor gets his story."
    },
    {
        "id": "secret_diana_debt",
        "description": "Diana owes Marcus a large sum of money, making her financially dependent on him.",
        "truth_value": true,
        "holder": ["diana"],
        "about": "diana",
        "content_type": "financial",
        "initial_knowers": ["diana", "marcus"],
        "initial_suspecters": [],
        "dramatic_weight": 0.6,
        "reveal_consequences": "Diana's loyalty conflict becomes public. Elena realizes Diana might not be a trustworthy confidante. Marcus has leverage."
    },
    {
        "id": "secret_lydia_knows",
        "description": "Lydia has noticed financial discrepancies in the firm's books and suspects Marcus of wrongdoing.",
        "truth_value": true,
        "holder": ["lydia"],
        "about": "lydia",
        "content_type": "knowledge",
        "initial_knowers": ["lydia"],
        "initial_suspecters": [],
        "dramatic_weight": 0.5,
        "reveal_consequences": "If Marcus learns Lydia suspects him, she becomes a threat he needs to neutralize. If Thorne learns, he has an ally."
    },
    {
        "id": "secret_victor_investigation",
        "description": "Victor is secretly investigating Marcus's business dealings for a journalistic expose.",
        "truth_value": true,
        "holder": ["victor"],
        "about": "victor",
        "content_type": "investigation",
        "initial_knowers": ["victor"],
        "initial_suspecters": ["marcus"],
        "dramatic_weight": 0.7,
        "reveal_consequences": "If Marcus confirms Victor is investigating, he'll either try to buy Victor off, discredit him, or flee. If Thorne learns, Victor becomes an ally or a threat."
    }
]
```

### Secret Interconnections

The secrets form a web of mutual dependency:

```
secret_affair_01 ←→ secret_embezzle_01
    (Marcus is double-betraying Thorne: affair + money)
    (Exposing one increases risk of exposing the other)

secret_affair_01 ←→ secret_diana_debt
    (Diana knows the affair but owes Marcus money)
    (Her debt constrains her loyalty to Elena)

secret_embezzle_01 ←→ secret_lydia_knows
    (Lydia's suspicions could confirm the embezzlement)
    (If combined with Victor's investigation = critical mass)

secret_embezzle_01 ←→ secret_victor_investigation
    (Victor is trying to prove what Lydia suspects)
    (If they share notes, Marcus is finished)

secret_lydia_knows ←→ secret_victor_investigation
    (Natural allies if they discover each other's knowledge)
    (Together they have enough to confront Marcus)
```

This creates a "pressure cooker" dynamic: multiple secrets that reinforce each other, where revealing one increases the pressure on others.

---

## 7. WorldDefinition

The complete world specification that the simulation engine loads at startup.

### Python

```python
@dataclass
class WorldDefinition:
    """Complete world specification loaded at simulation start."""
    id: str                                      # World ID. E.g. "dinner_party_01"
    name: str                                    # Display name. E.g. "The Dinner Party"
    description: str                             # Human-readable world description
    sim_duration_minutes: float                  # Expected duration in sim-minutes
    ticks_per_minute: float                      # How many ticks per sim-minute (controls granularity)

    locations: dict[str, Location]               # location_id → Location
    secrets: dict[str, SecretDefinition]         # secret_id → SecretDefinition
    seating: dict[str, list[str]] | None         # agent_id → adjacent_agent_ids (if applicable)

    # Thematic focus (from doc3.md THEMATIC_AXES)
    primary_themes: list[str]                    # Which thematic axes are most relevant
                                                 # E.g. ["loyalty_betrayal", "truth_deception"]

    # Pacing parameters (can be tuned per world)
    snapshot_interval: int = 20                  # Take WorldState snapshot every N events (Decision #2)
    catastrophe_threshold: float = 0.35          # base threshold for stress * commitment^2 (Decision #10)
                                                  # effective = max(0.1, base - suppression_count * 0.05)
                                                  # See specs/simulation/pacing-physics.md
    composure_gate: float = 0.30                  # Composure gate for catastrophe (Decision #10)
    trust_repair_multiplier: float = 3.0         # Hysteresis: trust repair costs Nx (Decision #5)
```

### TypeScript

```typescript
interface WorldDefinition {
    id: string;
    name: string;
    description: string;
    sim_duration_minutes: number;
    ticks_per_minute: number;

    locations: Record<string, Location>;
    secrets: Record<string, SecretDefinition>;
    seating: Record<string, string[]> | null;

    primary_themes: string[];

    snapshot_interval: number;
    catastrophe_threshold: number;
    composure_gate: number;
    trust_repair_multiplier: number;
}
```

---

## 8. JSON Examples

### Complete Dinner Party WorldDefinition

```json
{
    "id": "dinner_party_01",
    "name": "The Dinner Party",
    "description": "Six guests gather for an evening dinner. Beneath the polished surface, a web of secrets, affairs, and financial betrayal threatens to unravel.",
    "sim_duration_minutes": 150.0,
    "ticks_per_minute": 2.0,

    "locations": {
        "dining_table": {
            "id": "dining_table",
            "name": "Dining Table",
            "privacy": 0.1,
            "capacity": 6,
            "adjacent": ["kitchen", "balcony", "foyer", "bathroom"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "The main dining table. Seats six. The social center of the evening."
        },
        "kitchen": {
            "id": "kitchen",
            "name": "Kitchen",
            "privacy": 0.5,
            "capacity": 3,
            "adjacent": ["dining_table"],
            "overhear_from": ["dining_table"],
            "overhear_probability": 0.3,
            "description": "An open-plan kitchen adjacent to the dining area."
        },
        "balcony": {
            "id": "balcony",
            "name": "Balcony",
            "privacy": 0.7,
            "capacity": 3,
            "adjacent": ["dining_table"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "A small balcony off the dining room. Private enough for real conversations."
        },
        "foyer": {
            "id": "foyer",
            "name": "Foyer",
            "privacy": 0.2,
            "capacity": 4,
            "adjacent": ["dining_table"],
            "overhear_from": ["dining_table"],
            "overhear_probability": 0.15,
            "description": "The entrance hallway."
        },
        "bathroom": {
            "id": "bathroom",
            "name": "Bathroom",
            "privacy": 0.9,
            "capacity": 2,
            "adjacent": ["dining_table"],
            "overhear_from": [],
            "overhear_probability": 0.0,
            "description": "A small bathroom down the hall. The most private space available."
        }
    },

    "secrets": {
        "secret_affair_01": {
            "id": "secret_affair_01",
            "description": "Elena and Marcus are having a romantic affair behind Thorne's back.",
            "truth_value": true,
            "holder": ["elena"],
            "about": "elena",
            "content_type": "affair",
            "initial_knowers": ["elena", "marcus", "diana"],
            "initial_suspecters": ["lydia"],
            "dramatic_weight": 1.0,
            "reveal_consequences": "Thorne's marriage and business partnership are both destroyed."
        },
        "secret_embezzle_01": {
            "id": "secret_embezzle_01",
            "description": "Marcus has been embezzling money from the business he shares with Thorne.",
            "truth_value": true,
            "holder": ["marcus"],
            "about": "marcus",
            "content_type": "financial",
            "initial_knowers": ["marcus"],
            "initial_suspecters": ["lydia", "victor"],
            "dramatic_weight": 0.9,
            "reveal_consequences": "Marcus faces criminal charges. Thorne realizes double betrayal."
        },
        "secret_diana_debt": {
            "id": "secret_diana_debt",
            "description": "Diana owes Marcus a large sum of money.",
            "truth_value": true,
            "holder": ["diana"],
            "about": "diana",
            "content_type": "financial",
            "initial_knowers": ["diana", "marcus"],
            "initial_suspecters": [],
            "dramatic_weight": 0.6,
            "reveal_consequences": "Diana's loyalty conflict becomes public."
        },
        "secret_lydia_knows": {
            "id": "secret_lydia_knows",
            "description": "Lydia has noticed financial discrepancies and suspects Marcus.",
            "truth_value": true,
            "holder": ["lydia"],
            "about": "lydia",
            "content_type": "knowledge",
            "initial_knowers": ["lydia"],
            "initial_suspecters": [],
            "dramatic_weight": 0.5,
            "reveal_consequences": "If Marcus learns, Lydia becomes a threat."
        },
        "secret_victor_investigation": {
            "id": "secret_victor_investigation",
            "description": "Victor is investigating Marcus for a journalistic expose.",
            "truth_value": true,
            "holder": ["victor"],
            "about": "victor",
            "content_type": "investigation",
            "initial_knowers": ["victor"],
            "initial_suspecters": ["marcus"],
            "dramatic_weight": 0.7,
            "reveal_consequences": "Marcus discovers the investigation and reacts."
        }
    },

    "seating": {
        "thorne": ["elena", "victor"],
        "elena":  ["thorne", "marcus"],
        "marcus": ["elena", "diana"],
        "diana":  ["marcus", "lydia"],
        "lydia":  ["diana", "victor"],
        "victor": ["lydia", "thorne"]
    },

    "primary_themes": ["loyalty_betrayal", "truth_deception"],

    "snapshot_interval": 20,
    "catastrophe_threshold": 0.35,
    "composure_gate": 0.30,
    "trust_repair_multiplier": 3.0
}
```

### Simulation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sim_duration_minutes` | 150.0 | 2.5-hour dinner party |
| `ticks_per_minute` | 2.0 | 1 tick = 30 seconds sim-time. 300 total ticks for the evening. |
| `snapshot_interval` | 20 events | ~1 snapshot per 10 minutes of sim-time |
| Target event count | 100-200 | Per CLAUDE.md MVP spec |
| Events per tick | 0.33-0.67 avg | Not every tick produces an event — many ticks are quiet |

---

## 9. Validation Rules

### Location Validation

1. `id` must be non-empty and unique among all locations
2. `name` must be non-empty
3. `privacy` must be in [0.0, 1.0]
4. `capacity` must be >= 1
5. `adjacent` must only contain valid location IDs
6. Adjacency must be symmetric: if A lists B, B must list A
7. `overhear_from` must only contain valid location IDs
8. `overhear_from` entries must be adjacent or near-adjacent locations (no overhearing from far-away rooms)
9. `overhear_probability` must be in [0.0, 1.0]
10. A location cannot list itself in `adjacent` or `overhear_from`

### Secret Validation

1. `id` must be non-empty and unique among all secrets
2. `holder` must reference a valid agent ID
3. `about` (if non-null) must reference a valid agent ID
4. `initial_knowers` must reference valid agent IDs
5. `initial_suspecters` must reference valid agent IDs
6. `initial_knowers` and `initial_suspecters` must not overlap (can't both know and suspect)
7. `holder` must be in `initial_knowers` (the originator knows the secret)
8. `dramatic_weight` must be in [0.0, 1.0]
9. `content_type` must be one of the recognized types

### WorldDefinition Validation

1. Must have at least one location
2. Must have at least one secret (for meaningful drama)
3. `sim_duration_minutes` must be > 0
4. `ticks_per_minute` must be > 0
5. `snapshot_interval` must be >= 1
6. `catastrophe_threshold` must be in (0.0, 1.0] (this is the base threshold before suppression bonus)
7. `composure_gate` must be in [0.0, 1.0)
8. `trust_repair_multiplier` must be >= 1.0
9. If `seating` is provided, all agent IDs must be valid and seating adjacency must be symmetric
10. All location IDs referenced in agent states must exist in `locations`
11. All secret IDs referenced in agent beliefs must exist in `secrets`

---

## 10. NOT In Scope

- **Agent definitions** — agent state and initial values. See `specs/schema/agents.md` and `specs/simulation/dinner-party-config.md`.
- **Event definitions** — event schema and types. See `specs/schema/events.md`.
- **Scene definitions** — how events group into scenes. See `specs/schema/scenes.md`.
- **Movement scoring** — how the decision engine scores SOCIAL_MOVE actions. See `specs/simulation/decision-engine.md`.
- **World generation** — procedural world creation. Out of scope for MVP.
- **MVP 1.5 "The Bridge"** — two-location world defined in Decision #17. Separate world definition file when needed.

---

## Dependencies

This spec is upstream of:
- `specs/simulation/dinner-party-config.md` (uses locations and secrets to configure the simulation)
- `specs/simulation/tick-loop.md` (checks capacity, adjacency, overhearing)
- `specs/simulation/decision-engine.md` (scores movement actions based on location properties)
- `specs/metrics/scene-segmentation.md` (uses location continuity as segmentation signal)
- `specs/visualization/thread-layout.md` (may use location for Y-axis grouping)

This spec depends on:
- `specs/schema/events.md` (Event.location_id references Location.id)
- `specs/schema/agents.md` (AgentState.location references Location.id, AgentState.beliefs references SecretDefinition.id)
