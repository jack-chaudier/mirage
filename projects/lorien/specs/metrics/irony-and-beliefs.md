# Irony and Beliefs Specification

> **Spec:** `specs/metrics/irony-and-beliefs.md`
> **Owner:** metrics-architect
> **Status:** Draft
> **Depends on:** `specs/schema/events.md` (#1), `specs/schema/agents.md` (#2)
> **Blocks:** `specs/integration/data-flow.md` (#17)
> **Doc3 decisions:** #8 (finite proposition catalog), #5 (pacing physics — composure masks true beliefs)

---

## 1. Overview

Dramatic irony is the gap between what the audience knows (the ground truth) and what the characters believe. It is one of the most powerful narrative devices: the audience sees the bomb under the table, but the characters don't.

This spec defines:
1. The **belief matrix** data structure (agent x secret -> BeliefState)
2. How to compute **per-agent irony** and **scene-level irony**
3. What makes a secret **relevant** to a scene
4. How **irony collapse** (a reveal event) is detected and scored
5. The full **initial belief matrix** for the dinner party
6. A **10-event trace** showing belief matrix evolution

### Relationship to Tension Pipeline

The `irony_density` sub-metric in `specs/metrics/tension-pipeline.md` is a simplified version of the irony computation defined here. That sub-metric uses the same belief matrix and scoring but is scoped to a single event. This spec defines the full system, including scene-level irony, irony collapse detection, and evolution tracking.

---

## 2. The Belief Matrix

### 2.1 Data Structure

```python
# The belief matrix is: Dict[str, Dict[str, BeliefState]]
# Keyed as: beliefs[agent_id][secret_id] -> BeliefState

class BeliefState(Enum):
    UNKNOWN        = "unknown"          # hasn't encountered this proposition
    SUSPECTS       = "suspects"         # has partial evidence, not certain
    BELIEVES_TRUE  = "believes_true"    # confident it's true
    BELIEVES_FALSE = "believes_false"   # confident it's false (may be wrong)

# Ground truth is stored in Secret.truth_value: bool
```

### 2.2 Belief Transitions

Beliefs transition through specific event types. Not all transitions are allowed.

```
Valid transitions:

UNKNOWN → SUSPECTS         (via OBSERVE, CHAT with hints, overhearing)
UNKNOWN → BELIEVES_TRUE    (via REVEAL, CONFIDE)
UNKNOWN → BELIEVES_FALSE   (via LIE, misleading OBSERVE)

SUSPECTS → BELIEVES_TRUE   (via REVEAL, CONFIDE, accumulating evidence)
SUSPECTS → BELIEVES_FALSE  (via LIE, convincing counter-evidence)
SUSPECTS → UNKNOWN         (NOT allowed — you can't unsuspect)

BELIEVES_TRUE → BELIEVES_FALSE    (via convincing LIE — rare, requires high trust in liar)
BELIEVES_FALSE → BELIEVES_TRUE    (via REVEAL with proof — overcoming denial)
BELIEVES_FALSE → SUSPECTS         (via OBSERVE that contradicts current belief)

BELIEVES_TRUE → UNKNOWN    (NOT allowed — knowledge doesn't disappear)
BELIEVES_FALSE → UNKNOWN   (NOT allowed)
```

### 2.3 Transition Triggers by Event Type

| Event Type | Possible Belief Transitions |
|---|---|
| CHAT | UNKNOWN → SUSPECTS (if topic is adjacent to secret) |
| OBSERVE | UNKNOWN → SUSPECTS; BELIEVES_FALSE → SUSPECTS (saw contradicting evidence) |
| REVEAL | UNKNOWN → BELIEVES_TRUE; SUSPECTS → BELIEVES_TRUE; BELIEVES_FALSE → BELIEVES_TRUE |
| CONFIDE | UNKNOWN → BELIEVES_TRUE; SUSPECTS → BELIEVES_TRUE (told privately) |
| LIE | UNKNOWN → BELIEVES_FALSE; SUSPECTS → BELIEVES_FALSE (if trust in liar is high) |
| CONFLICT | Can trigger SUSPECTS via witnessed behavior (e.g., defensive reaction implies hiding something) |
| CATASTROPHE | Can trigger BELIEVES_TRUE via involuntary confession or breakdown |

Belief deltas are expressed as StateDelta with `kind: DeltaKind.BELIEF`:

```python
StateDelta(
    kind=DeltaKind.BELIEF,
    agent="lydia",              # the agent whose belief changes
    agent_b="marcus",           # context: whose secret, or who told them
    attribute="secret_embezzle_01",  # which secret
    op=DeltaOp.SET,
    value="believes_false",     # new BeliefState value
    reason_code="LIED_TO",
    reason_display="Marcus told Lydia the discrepancy was a clerical error",
)
```

---

## 3. Irony Computation

### 3.1 Per-Agent Irony Score

```python
def agent_irony(
    agent_id: str,
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, Secret],
    agents_present: set[str],
) -> float:
    """
    How ironically positioned is this agent?

    Scoring (from specs/schema/agents.md Section 5):
    - Actively wrong belief (BELIEVES_FALSE when truth is TRUE, or vice versa): 2.0
    - Relevant unknown (UNKNOWN on a secret that is about them or held by them): 1.5
    - General unknown (UNKNOWN on a true secret, low relevance): 0.5
    - Suspects (SUSPECTS something that's true): 0.25
    - Correct belief: 0.0 (no irony)

    Weighted by relevance: how much does this secret matter to this agent right now?
    """
    score = 0.0
    for secret_id, secret in secrets.items():
        relevance = secret_relevance(secret, agent_id, agents_present)
        if relevance < 0.1:
            continue

        belief = beliefs.get(agent_id, {}).get(secret_id, BeliefState.UNKNOWN)

        # Actively wrong: maximum irony
        if belief == BeliefState.BELIEVES_TRUE and not secret.truth_value:
            score += 2.0 * relevance
        elif belief == BeliefState.BELIEVES_FALSE and secret.truth_value:
            score += 2.0 * relevance

        # Relevant unknown: high irony if the secret is about/held by this agent
        elif belief == BeliefState.UNKNOWN:
            if secret.about == agent_id or secret.holder == agent_id:
                score += 1.5  # the secret is about them and they don't know!
            elif relevance >= 0.3:
                score += 0.5  # general ignorance

        # Suspects but doesn't know: small irony
        elif belief == BeliefState.SUSPECTS:
            if secret.truth_value:
                score += 0.25 * relevance

    return score
```

### 3.2 Secret Relevance

```python
def secret_relevance(
    secret: Secret,
    agent_id: str,
    agents_present: set[str],
) -> float:
    """
    How relevant is this secret to this agent in this context?

    Returns:
    - 1.0: The secret is ABOUT this agent (they're the subject but may not know)
    - 0.9: The secret is about someone this agent is in a relationship with
           AND both are present (the irony is visible to the audience)
    - 0.7: The secret is HELD BY this agent (they're a conspirator)
    - 0.5: The secret is about someone present, and this agent interacts with them
    - 0.2: The secret exists in the world but has no direct connection to this agent
    - 0.0: No relevance
    """
    if secret.about == agent_id:
        return 1.0

    if secret.holder == agent_id:
        return 0.7

    if secret.about in agents_present:
        # The subject of the secret is here — irony from proximity
        return 0.5

    return 0.2
```

### 3.3 Scene-Level Irony

```python
def scene_irony(
    agents_present: set[str],
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, Secret],
) -> float:
    """
    Total irony across all agents present in the scene.

    This measures how "ironically charged" a scene is — how many characters
    are operating under false or incomplete information about things that matter.
    """
    total = sum(
        agent_irony(a, beliefs, secrets, agents_present)
        for a in agents_present
    )

    # Normalize by agent count to allow cross-scene comparison
    if not agents_present:
        return 0.0
    return total / len(agents_present)
```

### 3.4 Pairwise Irony (Between Two Agents)

A particularly useful metric for the renderer: how ironically positioned are two specific agents relative to each other?

```python
def pairwise_irony(
    agent_a: str,
    agent_b: str,
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, Secret],
) -> float:
    """
    Irony between two agents = sum of cases where:
    - A knows something B doesn't (or vice versa)
    - A believes the opposite of B about the same secret
    - One of them is wrong about a secret that's about the other
    """
    score = 0.0
    for secret_id, secret in secrets.items():
        belief_a = beliefs.get(agent_a, {}).get(secret_id, BeliefState.UNKNOWN)
        belief_b = beliefs.get(agent_b, {}).get(secret_id, BeliefState.UNKNOWN)

        # Asymmetric knowledge: one knows, the other doesn't
        if _knows(belief_a) and not _knows(belief_b):
            relevance = max(
                secret_relevance(secret, agent_a, {agent_a, agent_b}),
                secret_relevance(secret, agent_b, {agent_a, agent_b}),
            )
            score += 1.0 * relevance
        elif _knows(belief_b) and not _knows(belief_a):
            relevance = max(
                secret_relevance(secret, agent_a, {agent_a, agent_b}),
                secret_relevance(secret, agent_b, {agent_a, agent_b}),
            )
            score += 1.0 * relevance

        # Contradictory beliefs: one believes true, the other believes false
        if (belief_a == BeliefState.BELIEVES_TRUE and belief_b == BeliefState.BELIEVES_FALSE) or \
           (belief_a == BeliefState.BELIEVES_FALSE and belief_b == BeliefState.BELIEVES_TRUE):
            score += 2.0

        # One is wrong about a secret that's about the other
        if secret.about == agent_b and _is_wrong(belief_a, secret):
            score += 1.5
        if secret.about == agent_a and _is_wrong(belief_b, secret):
            score += 1.5

    return score


def _knows(belief: BeliefState) -> bool:
    return belief in {BeliefState.BELIEVES_TRUE, BeliefState.BELIEVES_FALSE}


def _is_wrong(belief: BeliefState, secret: Secret) -> bool:
    if belief == BeliefState.BELIEVES_TRUE and not secret.truth_value:
        return True
    if belief == BeliefState.BELIEVES_FALSE and secret.truth_value:
        return True
    return False
```

---

## 4. Irony Collapse

### 4.1 Definition

An **irony collapse** occurs when a REVEAL event causes one or more agents' beliefs to align with ground truth on a previously unknown or misbelieved secret. The "ironic charge" that was building up discharges in a single event.

### 4.2 Detection

```python
def detect_irony_collapse(
    event: Event,
    beliefs_before: dict[str, dict[str, BeliefState]],
    beliefs_after: dict[str, dict[str, BeliefState]],
    secrets: dict[str, Secret],
) -> IronyCollapse | None:
    """
    Check if this event caused an irony collapse.

    Returns IronyCollapse if scene_irony dropped significantly, None otherwise.
    """
    agents_present = {event.source_agent} | set(event.target_agents)

    irony_before = scene_irony(agents_present, beliefs_before, secrets)
    irony_after = scene_irony(agents_present, beliefs_after, secrets)

    drop = irony_before - irony_after

    # Threshold: a drop of >= 0.5 normalized irony constitutes a collapse
    if drop >= 0.5:
        # Identify which secrets collapsed
        collapsed_secrets = []
        for secret_id, secret in secrets.items():
            for agent_id in agents_present:
                b_before = beliefs_before.get(agent_id, {}).get(secret_id, BeliefState.UNKNOWN)
                b_after = beliefs_after.get(agent_id, {}).get(secret_id, BeliefState.UNKNOWN)

                was_ironic = _is_wrong(b_before, secret) or (
                    b_before == BeliefState.UNKNOWN
                    and secret_relevance(secret, agent_id, agents_present) >= 0.5
                )
                now_correct = (
                    (b_after == BeliefState.BELIEVES_TRUE and secret.truth_value)
                    or (b_after == BeliefState.BELIEVES_FALSE and not secret.truth_value)
                )

                if was_ironic and now_correct:
                    collapsed_secrets.append((agent_id, secret_id))

        return IronyCollapse(
            event_id=event.id,
            irony_before=irony_before,
            irony_after=irony_after,
            drop=drop,
            collapsed_beliefs=collapsed_secrets,
        )

    return None


@dataclass
class IronyCollapse:
    event_id: str
    detected: bool                                # True when irony collapse is detected
    irony_before: float
    irony_after: float
    drop: float
    collapsed_beliefs: list[dict]                 # [{agent, secret, from, to}] — belief transitions
    score: float = 0.0                            # [0.0, 1.0] — computed by irony_collapse_score()
```

### 4.3 Scoring Irony Collapse

An irony collapse is a candidate **turning point** in the arc grammar. Its narrative weight depends on:

```python
def irony_collapse_score(collapse: IronyCollapse) -> float:
    """
    How dramatically significant is this irony collapse?

    Factors:
    1. Magnitude of the drop (larger drop = more dramatic)
    2. Number of beliefs that collapsed simultaneously (chain reaction = more dramatic)
    3. Whether any collapsed belief was BELIEVES_FALSE→BELIEVES_TRUE (denial broken)
    """
    magnitude = min(collapse.drop / 2.0, 1.0)  # normalize: 2.0 drop is max drama
    breadth = min(len(collapse.collapsed_beliefs) / 4.0, 1.0)  # 4+ beliefs collapsing is exceptional

    # Bonus for denial breaking (someone who was actively wrong gets corrected)
    # This is detected by checking if any belief went from BELIEVES_FALSE to BELIEVES_TRUE
    denial_bonus = 0.0  # computed when we have access to before/after states

    return 0.5 * magnitude + 0.3 * breadth + 0.2 * denial_bonus
```

---

## 5. Initial Belief Matrix for the Dinner Party

> **Source of truth:** Character definitions and initial beliefs come from `specs/schema/agents.md` Section 9 and `specs/schema/world.md`. This section reproduces the belief matrix and annotates it for irony analysis.

### 5.1 Characters

| ID | Name | Role |
|---|---|---|
| thorne | Thorne Ashford | Host, businessman, married to Elena |
| elena | Elena Thorne | Thorne's wife, having an affair with Marcus |
| marcus | Marcus Webb | Elena's secret lover, embezzling from Thorne |
| lydia | Lydia Chen | Colleague who suspects Marcus's financial wrongdoing |
| diana | Diana Reeves | Knows about the affair, owes Marcus money |
| victor | Victor Osei | Journalist investigating Marcus |

### 5.2 Secrets

| Secret ID | Holder | About | Content Type | Truth Value | Description |
|---|---|---|---|---|---|
| `secret_affair_01` | elena, marcus | elena, marcus | affair | true | Elena and Marcus are having a romantic affair behind Thorne's back |
| `secret_embezzle_01` | marcus | marcus | financial | true | Marcus has been embezzling money from the business he shares with Thorne |
| `secret_diana_debt` | diana, marcus | diana | financial | true | Diana owes Marcus a large sum of money, making her financially dependent on him |
| `secret_lydia_knows` | lydia | lydia | knowledge | true | Lydia has noticed financial discrepancies and suspects Marcus of wrongdoing |
| `secret_victor_investigation` | victor | victor | investigation | true | Victor is secretly investigating Marcus's business dealings for a journalistic expose |

### 5.3 Initial Belief Matrix (Tick 0)

From `specs/schema/agents.md` Section 9:

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **thorne** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **elena** | BELIEVES_TRUE | UNKNOWN | SUSPECTS | UNKNOWN | UNKNOWN |
| **marcus** | BELIEVES_TRUE | BELIEVES_TRUE | UNKNOWN | UNKNOWN | SUSPECTS |
| **lydia** | SUSPECTS | SUSPECTS | UNKNOWN | BELIEVES_TRUE | UNKNOWN |
| **diana** | BELIEVES_TRUE | UNKNOWN | BELIEVES_TRUE | UNKNOWN | UNKNOWN |
| **victor** | UNKNOWN | SUSPECTS | UNKNOWN | UNKNOWN | BELIEVES_TRUE |

**Reading the matrix:** Each cell answers "What does [row agent] believe about [column secret]?"

**Irony hotspots at tick 0:**
- Thorne is UNKNOWN on ALL 5 secrets — maximum irony potential. He doesn't know about the affair (about his wife), the embezzlement (from his business), or anything else.
- Marcus has the most to hide (BELIEVES_TRUE on affair + embezzlement) and SUSPECTS Victor is investigating him — maximum stress.
- Lydia SUSPECTS the affair and the embezzlement but lacks courage to act — maximum internal tension.
- Diana knows the affair (BELIEVES_TRUE) but owes Marcus money (BELIEVES_TRUE on diana_debt) — maximum loyalty conflict.
- Victor is investigating (BELIEVES_TRUE on his own investigation) and SUSPECTS embezzlement — if he and Lydia share notes, Marcus is finished.

**Initial per-agent irony scores (all agents present at dining table):**

Using the canonical scoring from `specs/schema/agents.md`:
- Actively wrong (BELIEVES_FALSE when true): 2.0
- Relevant unknown (secret about/held by this agent): 1.5
- General unknown (true secret, low relevance): 0.5
- Suspects true thing: 0.25

| Agent | Irony Score | Key Contributors |
|---|---|---|
| thorne | 4.0 | UNKNOWN on affair_01 (1.5, about his wife) + UNKNOWN on embezzle_01 (1.5, about his business) + UNKNOWN on diana_debt (0.5) + UNKNOWN on lydia_knows (0.5) |
| elena | 1.25 | UNKNOWN on embezzle_01 (0.5) + SUSPECTS diana_debt (0.25) + UNKNOWN on lydia_knows (0.5) + UNKNOWN on victor_investigation (0.5, low relevance but about her lover's enemy) |
| marcus | 1.75 | UNKNOWN on diana_debt (0.5) + UNKNOWN on lydia_knows (1.5, about him — Lydia's suspicion targets him) + SUSPECTS victor_investigation (0.25) |
| lydia | 1.25 | SUSPECTS affair_01 (0.25) + SUSPECTS embezzle_01 (0.25) + UNKNOWN on diana_debt (0.5) + UNKNOWN on victor_investigation (0.5, natural ally) |
| diana | 1.0 | UNKNOWN on embezzle_01 (0.5) + UNKNOWN on lydia_knows (0.5) + UNKNOWN on victor_investigation (0.5) |
| victor | 1.5 | UNKNOWN on affair_01 (0.5) + SUSPECTS embezzle_01 (0.25) + UNKNOWN on diana_debt (0.5) + UNKNOWN on lydia_knows (0.5, natural ally) |

**Initial scene_irony (all 6 present):** (4.0 + 1.25 + 1.75 + 1.25 + 1.0 + 1.5) / 6 = **1.79**

---

## 6. Belief Matrix Evolution: 10-Event Trace

### Event Sequence

Starting from the initial matrix above (Section 5.3), we trace 10 events that progressively shift the belief landscape. All characters and secrets match the canonical definitions in `specs/schema/agents.md` and `specs/schema/world.md`.

---

**Event E001** — Tick 3, sim_time 4.5 min
```
Type: CHAT
Source: lydia → Target: [thorne]
Location: dining_table
Description: "Lydia mentions the firm's recent quarterly report, watching Thorne's reaction."
Belief changes: None
```

Matrix unchanged. Lydia is probing — she SUSPECTS embezzlement and is looking for confirmation from the business owner. Thorne, who is UNKNOWN on embezzle_01, doesn't pick up on the significance.

---

**Event E002** — Tick 7, sim_time 10.5 min
```
Type: OBSERVE
Source: victor → Target: []
Location: dining_table
Description: "Victor watches Marcus pour wine for Elena, noticing the familiarity of the gesture."
Belief changes:
    victor.secret_affair_01: UNKNOWN → SUSPECTS
Deltas: [{kind: BELIEF, agent: "victor", attribute: "secret_affair_01", op: SET, value: "suspects"}]
```

**Updated matrix row for Victor:**

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **victor** | **SUSPECTS** | SUSPECTS | UNKNOWN | UNKNOWN | BELIEVES_TRUE |

Victor, already investigating Marcus for financial wrongdoing, now picks up a second thread. This is dramatically significant: the affair and the embezzlement are linked secrets (exposing one increases risk of exposing the other).

---

**Event E003** — Tick 12, sim_time 18.0 min
```
Type: OBSERVE
Source: thorne → Target: []
Location: dining_table
Description: "Thorne catches Elena touching Marcus's arm while laughing."
Belief changes:
    thorne.secret_affair_01: UNKNOWN → SUSPECTS
Deltas: [{kind: BELIEF, agent: "thorne", attribute: "secret_affair_01", op: SET, value: "suspects"}]
```

**Updated matrix row for Thorne:**

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **thorne** | **SUSPECTS** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

Thorne's irony drops: from 4.0 to ~3.25 (SUSPECTS on affair_01 gives 0.25 instead of 1.5). But the narrative tension INCREASES because the audience knows he's getting closer to the truth about his wife.

---

**Event E004** — Tick 16, sim_time 24.0 min
```
Type: CONFIDE
Source: diana → Target: [elena]
Location: kitchen
Description: "Diana pulls Elena aside: 'I know about you and Marcus. Be careful tonight — Thorne seems off.'"
Belief changes: None (Diana already BELIEVES_TRUE on affair_01, Elena already BELIEVES_TRUE)
```

No belief matrix change. But this event creates a **trust delta** — Diana is signaling alliance with Elena:
```
Deltas: [
    {kind: RELATIONSHIP, agent: "elena", agent_b: "diana", attribute: "trust", op: ADD, value: 0.15,
     reason_code: "CONFIDED_LOYALTY", reason_display: "Diana warns Elena about Thorne's mood"},
    {kind: AGENT_EMOTION, agent: "elena", attribute: "fear", op: ADD, value: 0.2},
]
```

The audience sees the irony: Diana is constrained by her debt to Marcus (secret_diana_debt). Her "loyalty" to Elena is complicated by her financial dependence on Elena's lover.

---

**Event E005** — Tick 20, sim_time 30.0 min
```
Type: CHAT
Source: marcus → Target: [thorne, elena]
Location: dining_table
Description: "Marcus compliments Thorne on the wine selection, asking about the business."
Belief changes: None
```

Matrix unchanged. Social maintenance event. But audience tension is high because Thorne SUSPECTS the affair and is sitting with both Marcus and Elena. Marcus is embezzling from Thorne AND having an affair with his wife.

---

**Event E006** — Tick 25, sim_time 37.5 min
```
Type: OBSERVE
Source: lydia → Target: []
Location: dining_table
Description: "Lydia notices Victor taking notes on his phone under the table while watching Marcus."
Belief changes:
    lydia.secret_victor_investigation: UNKNOWN → SUSPECTS
Deltas: [{kind: BELIEF, agent: "lydia", attribute: "secret_victor_investigation", op: SET, value: "suspects"}]
```

**Updated matrix row for Lydia:**

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **lydia** | SUSPECTS | SUSPECTS | UNKNOWN | BELIEVES_TRUE | **SUSPECTS** |

This is a critical moment. Lydia and Victor are natural allies — they both suspect Marcus. If they connect, they'll have enough combined evidence to confront him. The audience sees this potential convergence.

---

**Event E007** — Tick 29, sim_time 43.5 min
```
Type: LIE
Source: marcus → Target: [lydia]
Location: dining_table
Description: "Lydia asks Marcus about the Hartwell account discrepancy. Marcus says 'It was a clerical error, already fixed.'"
Belief changes:
    lydia.secret_embezzle_01: SUSPECTS → SUSPECTS (lie fails — Lydia's evidence outweighs his denial)
```

Matrix unchanged. Lydia already SUSPECTS embezzle_01 and has seen the actual numbers. Marcus's dismissal doesn't reduce her suspicion. But this event increases Lydia's conviction and damages trust:
```
Deltas: [
    {kind: RELATIONSHIP, agent: "lydia", agent_b: "marcus", attribute: "trust", op: ADD, value: -0.2,
     reason_code: "SUSPECTED_LYING", reason_display: "Lydia doesn't believe Marcus's explanation"}
]
```

---

**Event E008** — Tick 34, sim_time 51.0 min
```
Type: CONFIDE
Source: lydia → Target: [victor]
Location: balcony
Description: "Lydia corners Victor on the balcony: 'I've seen the books. Something doesn't add up with Marcus.'"
Belief changes:
    victor.secret_lydia_knows: UNKNOWN → BELIEVES_TRUE (Lydia directly revealed her knowledge)
    victor.secret_embezzle_01: SUSPECTS → BELIEVES_TRUE (Lydia's evidence confirms his suspicion)
Deltas: [
    {kind: BELIEF, agent: "victor", attribute: "secret_lydia_knows", op: SET, value: "believes_true"},
    {kind: BELIEF, agent: "victor", attribute: "secret_embezzle_01", op: SET, value: "believes_true"},
]
```

**Updated matrix row for Victor:**

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **victor** | SUSPECTS | **BELIEVES_TRUE** | UNKNOWN | **BELIEVES_TRUE** | BELIEVES_TRUE |

The natural allies have connected. Victor now has corroboration for his investigation. The irony shifts dramatically: Marcus doesn't know that two people at this dinner party have combined forces against him. His irony score jumps because secret_lydia_knows and secret_victor_investigation are now actively being used against him while he's UNKNOWN/SUSPECTS on them.

---

**Event E009** — Tick 38, sim_time 57.0 min
```
Type: OBSERVE
Source: marcus → Target: []
Location: dining_table
Description: "Marcus sees Lydia and Victor returning from the balcony together, whispering."
Belief changes:
    marcus.secret_victor_investigation: SUSPECTS → SUSPECTS (suspicion reinforced but no state change)
    marcus.secret_lydia_knows: UNKNOWN → SUSPECTS (why is Lydia talking to the journalist?)
Deltas: [
    {kind: BELIEF, agent: "marcus", attribute: "secret_lydia_knows", op: SET, value: "suspects"},
    {kind: AGENT_EMOTION, agent: "marcus", attribute: "fear", op: ADD, value: 0.3},
    {kind: PACING, agent: "marcus", attribute: "stress", op: ADD, value: 0.25},
]
```

**Updated matrix row for Marcus:**

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **marcus** | BELIEVES_TRUE | BELIEVES_TRUE | UNKNOWN | **SUSPECTS** | SUSPECTS |

Marcus is now aware that something is happening between his two potential adversaries. His stress spikes. Irony partially collapses for Marcus (he's becoming aware of the threat), but it remains high because he still doesn't know the full extent of what they've shared.

---

**Event E010** — Tick 42, sim_time 63.0 min
```
Type: CATASTROPHE
Source: thorne → Target: [elena, marcus]
Location: dining_table
Description: "Thorne, stressed by suspicion and alcohol, sees Elena and Marcus
exchange a look. His composure breaks. He says: 'Elena, how long have you
and Marcus been so... close?'"
Belief changes:
    thorne.secret_affair_01: SUSPECTS → BELIEVES_TRUE (his own outburst confirms his suspicion)
    lydia.secret_affair_01: SUSPECTS → BELIEVES_TRUE (witnesses the confrontation)
    diana.secret_affair_01: BELIEVES_TRUE → BELIEVES_TRUE (no change, already knew)
    victor.secret_affair_01: SUSPECTS → BELIEVES_TRUE (witnesses the confrontation)
Deltas: [
    {kind: BELIEF, agent: "thorne", attribute: "secret_affair_01", op: SET, value: "believes_true"},
    {kind: BELIEF, agent: "lydia", attribute: "secret_affair_01", op: SET, value: "believes_true"},
    {kind: BELIEF, agent: "victor", attribute: "secret_affair_01", op: SET, value: "believes_true"},
    {kind: PACING, agent: "thorne", attribute: "composure", op: SET, value: 0.1},
    {kind: PACING, agent: "thorne", attribute: "stress", op: SET, value: 0.0},
    {kind: AGENT_EMOTION, agent: "elena", attribute: "fear", op: ADD, value: 0.5},
    {kind: AGENT_EMOTION, agent: "marcus", attribute: "fear", op: ADD, value: 0.5},
]
```

This is a **catastrophe event** (Decision #10 — cusp catastrophe). Thorne's accumulated stress + suspicion from E003 exceeds the threshold while his composure was degraded by alcohol.

The affair secret collapses publicly. But note: the embezzlement secret is still hidden. Victor now knows about BOTH the affair and the embezzlement — he has maximum leverage. Marcus is being hit on one front while a second front (the investigation) is about to open.

### 6.1 Final Belief Matrix (After 10 Events)

|  | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|---|---|---|---|---|---|
| **thorne** | **BELIEVES_TRUE** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **elena** | BELIEVES_TRUE | UNKNOWN | SUSPECTS | UNKNOWN | UNKNOWN |
| **marcus** | BELIEVES_TRUE | BELIEVES_TRUE | UNKNOWN | **SUSPECTS** | SUSPECTS |
| **lydia** | **BELIEVES_TRUE** | SUSPECTS | UNKNOWN | BELIEVES_TRUE | **SUSPECTS** |
| **diana** | BELIEVES_TRUE | UNKNOWN | BELIEVES_TRUE | UNKNOWN | UNKNOWN |
| **victor** | **BELIEVES_TRUE** | **BELIEVES_TRUE** | UNKNOWN | **BELIEVES_TRUE** | BELIEVES_TRUE |

**Changes from initial (bold above):**
- Thorne: UNKNOWN → SUSPECTS → BELIEVES_TRUE on affair_01 (E003, E010)
- Marcus: UNKNOWN → SUSPECTS on lydia_knows (E009)
- Lydia: SUSPECTS → BELIEVES_TRUE on affair_01 (E010); UNKNOWN → SUSPECTS on victor_investigation (E006)
- Victor: UNKNOWN → SUSPECTS → BELIEVES_TRUE on affair_01 (E002, E010); SUSPECTS → BELIEVES_TRUE on embezzle_01 (E008); UNKNOWN → BELIEVES_TRUE on lydia_knows (E008)

### 6.2 Irony Score Evolution

| Tick | Scene Irony (avg per agent) | Key Change |
|---|---|---|
| 0 (initial) | 1.79 | Baseline: Thorne knows nothing, Marcus has most to hide |
| 3 (E001) | 1.79 | No change (probing CHAT) |
| 7 (E002) | 1.58 | Victor SUSPECTS affair (his irony drops) |
| 12 (E003) | 1.38 | Thorne SUSPECTS affair (his irony drops from 4.0 to ~3.25) |
| 16 (E004) | 1.38 | No belief change (trust delta only) |
| 20 (E005) | 1.38 | No change (social maintenance) |
| 25 (E006) | 1.30 | Lydia SUSPECTS victor_investigation (natural allies converging) |
| 29 (E007) | 1.30 | No belief change (lie failed) |
| 34 (E008) | 1.00 | Victor learns embezzle_01 + lydia_knows (major info exchange) |
| 38 (E009) | 0.92 | Marcus SUSPECTS lydia_knows (partial irony collapse for him) |
| 42 (E010) | **0.63** | IRONY COLLAPSE — affair_01 goes public; 3 agents update to BELIEVES_TRUE |

**Event E010 triggers an irony collapse:** Scene irony drops from 0.92 to 0.63 (delta = 0.29). The affair secret is now widely known. But importantly, new dramatic potential is CREATED:

The overall irony doesn't drop to zero because:
1. `secret_embezzle_01` remains unknown to Thorne, Elena, Diana (high relevance — it's about Thorne's business)
2. `secret_diana_debt` remains hidden from most agents
3. `secret_victor_investigation` is unknown to Thorne, Elena, Diana, and Marcus only SUSPECTS
4. Victor now holds maximum knowledge (knows affair + embezzlement + Lydia's awareness) — he is the most dangerous person at the party

This illustrates the **layered irony principle**: the dinner party has multiple irony channels. Collapsing one (the affair) actually increases the dramatic stakes of others (the embezzlement is now the primary ticking bomb, and Victor holds the detonator).

---

## 7. Irony Metrics Output Format

### 7.1 Per-Event Irony (in Event.metrics)

```json
{
    "irony": 0.63,
    "irony_collapse": {
        "detected": true,
        "drop": 0.29,
        "collapsed_beliefs": [
            {"agent": "thorne", "secret": "secret_affair_01", "from": "suspects", "to": "believes_true"},
            {"agent": "lydia", "secret": "secret_affair_01", "from": "suspects", "to": "believes_true"},
            {"agent": "victor", "secret": "secret_affair_01", "from": "suspects", "to": "believes_true"}
        ],
        "score": 0.82
    }
}
```

### 7.2 Belief Matrix Snapshot (for tooltip/detail views)

```json
{
    "tick_id": 42,
    "beliefs": {
        "thorne": {
            "secret_affair_01": "believes_true",
            "secret_embezzle_01": "unknown",
            "secret_diana_debt": "unknown",
            "secret_lydia_knows": "unknown",
            "secret_victor_investigation": "unknown"
        },
        "elena": { "..." : "..." },
        "marcus": { "..." : "..." },
        "lydia": { "..." : "..." },
        "diana": { "..." : "..." },
        "victor": { "..." : "..." }
    },
    "scene_irony": 0.63,
    "agent_irony": {
        "thorne": 2.5,
        "elena": 1.25,
        "marcus": 1.25,
        "lydia": 0.5,
        "diana": 1.0,
        "victor": 0.0
    },
    "pairwise_irony": {
        "thorne-marcus": 3.5,
        "victor-marcus": 2.8,
        "lydia-marcus": 1.5,
        "diana-marcus": 1.2
    }
}
```

---

## 8. Edge Cases

### Agent leaves the party
When an agent departs (location changes to `null` or "gone"), they are excluded from scene_irony computations. Their beliefs persist (they still believe what they believed) but they no longer contribute to the present scene's ironic charge.

### A lie creates a new false belief
When a LIE event sets an agent's belief to BELIEVES_FALSE on a true secret, irony INCREASES. This is correct: the liar has created dramatic irony by making someone believe a falsehood.

### Simultaneous reveals (same tick, different order_in_tick)
Process in order_in_tick sequence. If event A reveals secret X to agent 1, and event B (same tick, higher order) reveals secret X to agent 2, the belief matrix updates after A are visible when processing B.

### Secret truth_value changes
In the dinner party MVP, secret truth_values are static. If a future scenario allows truth changes (e.g., a plan that gets cancelled), the irony computation must re-evaluate all beliefs against the new truth.

### All beliefs align with truth
When scene_irony hits 0.0, there is zero dramatic irony. This can only happen if every agent has correct beliefs about every relevant secret. In practice, this is unlikely before the end of a scenario, because knowledge of non-relevant secrets remains incomplete.

---

## 9. NOT In Scope

- **Probabilistic beliefs:** Agents do not have continuous credences (P=0.73 that the affair is true). They have discrete states. This simplification is per Decision #8.
- **Higher-order beliefs:** "Thorne believes that Elena believes that he doesn't know about the affair." Not tracked in the belief matrix. Only first-order beliefs (agent x secret -> state).
- **Belief propagation rules:** How exactly an agent updates beliefs from evidence is the simulation engine's responsibility (`specs/simulation/decision-engine.md`). This spec defines what the irony pipeline reads, not how the sim writes.
- **Irony visualization:** How irony values map to visual properties (color, glow, annotation) is in `specs/visualization/renderer-architecture.md`.

---

## 10. Dependencies

| Depends On | What It Provides |
|---|---|
| `specs/schema/events.md` | Event schema, EventType, StateDelta with DeltaKind.BELIEF |
| `specs/schema/agents.md` | BeliefState enum, Agent.beliefs structure |
| `specs/simulation/dinner-party-config.md` | Character definitions, secrets, initial relationships |

| Depended On By | What It Consumes |
|---|---|
| `specs/metrics/tension-pipeline.md` | irony_density sub-metric uses the irony functions defined here |
| `specs/metrics/scene-segmentation.md` | Irony collapse detection feeds scene boundary detection |
| `specs/metrics/story-extraction.md` | Irony collapse scores factor into turning point identification |
| `specs/integration/data-flow.md` | Belief matrix snapshot format for interface contracts |
