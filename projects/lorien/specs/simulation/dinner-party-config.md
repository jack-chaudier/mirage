# Dinner Party Configuration Specification

> **Status:** Draft
> **Author:** sim-designer
> **Dependencies:** specs/schema/agents.md (#2) — DONE, specs/schema/world.md (#3) — DONE
> **Dependents:** specs/visualization/fake-data-visual-spec.md (#12)
> **Doc3 Decisions:** #17 (MVP Dinner Party Protocol), #13 (3-tier fake data)

---

## 1. Purpose

This spec is the **narrative design document** for the Dinner Party scenario. It defines WHY these specific characters, secrets, relationships, and starting conditions were chosen, what conflict arcs they're designed to produce, and how the initial state creates inevitable drama.

The actual data structures (agent JSON, secret JSON, location JSON) are defined in:
- `specs/schema/agents.md` Section 9 — all 6 character definitions with complete initial state
- `specs/schema/world.md` Sections 2, 5, 6 — locations, seating, and secret definitions

This spec adds: backstories, design rationale, conflict arc analysis, expected narrative trajectories, and scenario configuration.

**NOT in scope:**
- Data structure definitions (see agents.md, world.md)
- Simulation mechanics (see tick-loop.md, decision-engine.md, pacing-physics.md)
- Visualization (see renderer-architecture.md)

---

## 2. Scenario Overview

**Title:** The Dinner Party
**Duration:** ~150 minutes of simulated time (one evening)
**Target tick count:** 60-150 ticks
**Target event count:** 100-200 events
**Location:** James Thorne's upscale apartment. Five rooms: dining table, kitchen, balcony, foyer, bathroom.

**Premise:** James Thorne hosts a dinner party for his wife Elena, his business partner Marcus, and three other guests. Beneath the surface of polite conversation, a web of secrets — an affair, embezzlement, hidden debts, and a covert investigation — creates a pressure cooker of dramatic irony. The question isn't WHETHER the evening will explode, but WHEN, HOW, and WHO gets caught in the blast.

---

## 3. Character Profiles and Design Rationale

### 3.1 James Thorne — The Unknowing Host

**Backstory:** James Thorne built his wealth through a property development firm co-owned with Marcus Webb. He's proud, successful, and believes his life is in order: a beautiful wife, a thriving business, respected friends. He's hosting tonight's dinner to celebrate a new deal. What he doesn't know is that his wife is sleeping with his business partner, who is also stealing from him. Thorne's pride is both his armor and his blind spot — he can't conceive that the people closest to him could betray him.

**Design rationale:** Thorne is the audience's empathy anchor. His ignorance creates the core dramatic irony — the audience knows what he doesn't, and every friendly exchange between Marcus and Elena becomes charged with hidden meaning. His PRIDE flaw (strength 0.8) means he'll defend his self-image aggressively once the truth starts to emerge, potentially turning from victim to aggressor.

**Goal vector highlights:**
- Status: 0.9 (highest of any character) — he needs to be respected
- Loyalty: 0.8 — he values commitment and expects it from others
- Truth-seeking: 0.6 — he wants truth, but not urgently (he doesn't suspect anything)

**Intended arc:** Ignorance -> growing suspicion -> confrontation -> either devastating revelation or a final catastrophe where he discovers everything at once.

### 3.2 Elena Thorne — The Trapped Wife

**Backstory:** Elena married James for security but fell out of love years ago. She began an affair with Marcus six months ago — initially thrilling, now increasingly stressful as the deception takes its toll. She confided in Diana about the affair, which she both needed (unburdening) and regrets (one more person who knows). Tonight she has to sit between her husband and her lover and pretend everything is normal.

**Design rationale:** Elena is the character most likely to catastrophe. Her starting state is already pressured: stress=0.3, composure=0.7, commitment=0.4 (she's committed to maintaining secrecy). Her GUILT flaw makes her over-accommodate Thorne, which paradoxically increases her stress. Her COWARDICE flaw means she'll avoid confrontation and try to flee — but the dinner table is inescapable. The combination of guilt, avoidance, and external pressure is designed to build toward an involuntary confession.

**Goal vector highlights:**
- Secrecy: 0.9 (highest of any character) — she desperately needs the affair hidden
- Safety: 0.7 — she wants to avoid exposure and danger
- Closeness to Marcus: 0.8 — she's drawn to him even when it's dangerous

**Intended arc:** Anxious masking -> small slips -> growing panic -> either catastrophic confession or successful escape to the balcony for a private breakdown.

**Seating is critical:** Elena sits between Thorne (seat 1) and Marcus (seat 3). Every time she turns to speak to one, the other is watching. This physical positioning forces the tension.

### 3.3 Marcus Webb — The Double Betrayer

**Backstory:** Marcus is charismatic, ambitious, and increasingly desperate. He began embezzling from the firm eighteen months ago to cover gambling debts. The affair with Elena started as opportunism — she was lonely, he was charming — but has become a genuine complication. He suspects Victor might be investigating him and is watching Victor carefully. Marcus is the most complex character: he has the most secrets, the most to lose, and the highest starting commitment (0.6).

**Design rationale:** Marcus is the narrative bomb. He's simultaneously betraying his partner through both affair and theft. His AMBITION flaw makes him double down when cornered (overcommit) rather than cutting losses. His DENIAL flaw means he refuses to acknowledge the harm he's causing. This combination is designed to push him deeper into deception until the structure collapses. His high starting stress (0.4) and commitment (0.6) mean he's already in the catastrophe zone's outer envelope.

**Goal vector highlights:**
- Secrecy: 1.0 (maximum) — he has the most to hide
- Safety: 0.8 — he's the most paranoid
- Closeness to Victor: -0.3 (negative) — he wants distance from the investigator

**Intended arc:** Charming deflection -> lies under pressure -> escalating desperation -> either a masterful bluff that temporarily succeeds or a catastrophic unraveling when Victor and Lydia's separate suspicions converge.

**Catastrophe subtype:** If Marcus catastrophes, his AMBITION flaw + anger = "desperate_gambit" — a reckless accusation or power play, not a breakdown. He'd rather attack than crumble.

### 3.4 Lydia Cross — The Silent Witness

**Backstory:** Lydia has worked as Thorne's executive assistant for five years. She's quiet, observant, and fiercely loyal to Thorne. She noticed discrepancies in the firm's books three months ago and suspects Marcus is stealing, but hasn't told Thorne because: (a) she's not certain, (b) she's afraid of Marcus, and (c) she knows the accusation would destroy the partnership and potentially her own job. She also suspects Elena and Marcus might be more than friends based on body language she's observed at the office.

**Design rationale:** Lydia is the potential hero who CAN'T act. Her COWARDICE flaw (0.7) and loyalty flaw create a painful bind: she knows the truth (or suspects it), wants to protect Thorne, but can't bring herself to cause the confrontation that revealing the truth would require. She's the audience's "tell him!" character. If she can overcome her cowardice — perhaps provoked by watching Thorne be humiliated, or encouraged by Victor — the evening pivots. If she can't, she becomes a tragic figure: she knew and said nothing.

**Goal vector highlights:**
- Truth-seeking: 0.8 — she wants truth revealed (but can't do it herself)
- Loyalty: 0.9 (highest of any character) — she's devoted to Thorne
- Safety: 0.8 — she's risk-averse

**Intended arc:** Silent observation -> internal anguish -> either a breakthrough moment (revealing what she knows, possibly to Victor as an ally) or a quiet withdrawal where she watches the evening collapse and says nothing.

**Key interaction potential:** Lydia sits next to Victor (seats 5 and 6). If they start talking, two investigators might compare notes — which would be devastating for Marcus.

### 3.5 Diana Forrest — The Conflicted Confidante

**Backstory:** Diana is Elena's oldest friend. She knows about the affair because Elena confided in her during a tearful phone call. What Elena doesn't know is that Diana owes Marcus a significant sum of money — a loan Marcus made when Diana's business was failing. This creates an impossible loyalty conflict: Diana wants to protect Elena, but crossing Marcus could mean financial ruin. She's also envious of Elena's comfortable life, even as she sympathizes with Elena's unhappiness — a contradiction she doesn't fully admit to herself.

**Design rationale:** Diana is the wild card. She's pulled in multiple directions by competing loyalties. Her GUILT flaw makes her over-accommodate everyone, which means she'll say what people want to hear — a recipe for getting caught in contradictions. Her JEALOUSY flaw (mild, 0.4) could surface if she perceives that Elena doesn't appreciate what she has. Diana's most dramatic potential is as an accidental catalyst: in trying to manage everyone, she might slip — mentioning the affair to the wrong person, or letting Marcus's debt come up in conversation.

**Goal vector highlights:**
- Closeness to Elena: 0.7 — she values the friendship deeply
- Secrecy: 0.7 — she has her own secret (the debt) to protect
- Loyalty: 0.7 — she feels duty-bound but conflicted about to whom

**Intended arc:** Careful balancing act -> increasing strain -> either a deliberate betrayal (choosing Marcus over Elena out of financial self-interest) or an accidental reveal (guilt-driven slip) or a noble sacrifice (warning Elena despite the cost).

**Obligation asymmetry:** Diana has obligation 0.7 toward Marcus (the debt) and obligation 0.2 toward Elena (friendship). This means when forced to choose, the decision engine will weight Marcus's interests more heavily — which creates dramatic tension because the audience (and Diana herself) would want her to choose Elena.

### 3.6 Victor Hale — The Obsessive Journalist

**Backstory:** Victor is a freelance investigative journalist and Thorne's college friend. He was invited as a guest, but he's really here to observe Marcus. Victor has been investigating Marcus's business dealings for months — he smells a story about financial fraud but doesn't yet have proof. He doesn't know about the affair; his focus is entirely on the money. Victor is smart, socially adept when he wants to be, but has a journalist's tunnel vision — he sees people as sources, not as friends.

**Design rationale:** Victor is the external catalyst. He has no personal stake in the emotional dynamics (affair, loyalty, guilt) — he just wants the story. This makes him dangerous because he'll push where others self-censor. His OBSESSION flaw (0.8) means he'll pursue Marcus relentlessly, asking pointed questions, dropping hints that he knows things, testing reactions. His VANITY flaw (0.4) means he'll sometimes show off his knowledge to seem clever — which could tip Marcus off prematurely.

**Goal vector highlights:**
- Truth-seeking: 1.0 (maximum) — the only agent at maximum truth-seeking
- Status: 0.6 — he wants to be the smartest in the room
- Safety: 0.3 (lowest of any character) — he'll take risks for the story

**Intended arc:** Careful probing -> increasingly aggressive questions -> either a breakthrough (getting Lydia to confirm his suspicions) or a premature reveal (vanity-driven hint that warns Marcus).

**Key dynamic:** Victor's investigation and Lydia's suspicions are about the SAME SECRET (the embezzlement). If they connect, Marcus is finished. The seating arrangement places them next to each other (seats 5 and 6), creating opportunity. But Victor might alienate Lydia with his intensity (his obsession flaw can burn relationships).

---

## 4. Designed Conflict Arcs

The character configuration guarantees at least 3 natural conflict arcs will emerge. These aren't scripted — they're structural inevitabilities given the starting conditions.

### 4.1 Arc A: The Betrayal Triangle (Thorne - Elena - Marcus)

**Why it's inevitable:** Elena sits between Thorne and Marcus. She has high affection for Marcus (0.8) and low affection for Thorne (0.1). Her guilt flaw makes her over-accommodate Thorne, which looks suspicious to an observer. Marcus's secrecy goal (1.0) makes him cautious, but his denial flaw means he underestimates the danger. Any perceptive observer (Lydia, Victor) might notice the body language.

**Expected trigger:** Elena's composure erodes through alcohol and stress. Around tick 40-60, her composure drops below 0.40 (masking threshold), and she stops being able to hide her emotional state in public. Someone notices.

**Possible outcomes:**
- Elena confesses voluntarily (CONFIDE to Diana or REVEAL to Thorne)
- Elena catastrophes involuntarily (blurts the truth under pressure)
- Victor or Lydia piece it together from observation and confront someone
- The affair is never revealed this evening (rare given the pressure, but possible if Elena flees early)

### 4.2 Arc B: The Investigation (Victor - Marcus - Lydia)

**Why it's inevitable:** Victor is investigating Marcus (obsession flaw, truth-seeking 1.0). Lydia suspects Marcus and sits next to Victor. Marcus suspects Victor is investigating (SUSPECTS belief). These three characters create a three-body problem of suspicion and counter-suspicion.

**Expected trigger:** Victor's obsession flaw drives him to probe Marcus early (asking about the business, mentioning financial details). Marcus notices and gets defensive (stress rises). Lydia watches this exchange and sees confirmation of her suspicions.

**Possible outcomes:**
- Victor and Lydia compare notes (if Victor approaches her in a private space)
- Victor confronts Marcus directly (obsession + vanity → public accusation)
- Marcus preemptively attacks Victor (desperate_gambit catastrophe)
- The investigation goes cold (if Victor is too aggressive too early and Marcus deflects)

### 4.3 Arc C: The Loyalty Test (Diana - Elena - Marcus)

**Why it's inevitable:** Diana knows Elena's secret (affair) and owes Marcus money (debt). When pressure mounts on Marcus, he may lean on Diana (obligation 0.7) to keep quiet or even lie on his behalf. Elena is counting on Diana to keep her confidence. These competing obligations must eventually collide.

**Expected trigger:** Marcus discovers that Diana knows about the affair (through conversation or by overhearing Elena mention it). He then has leverage: "If you tell anyone about the affair, I'll call in the debt." OR: Thorne directly asks Diana if she's noticed anything strange about Elena.

**Possible outcomes:**
- Diana betrays Elena to protect herself (obligation to Marcus > loyalty to Elena)
- Diana warns Elena that Marcus is pressuring her (choosing friendship over money)
- Diana breaks down under the conflicting pressures (guilt catastrophe)
- Diana manages to stay neutral (rare — the pressures are designed to force a choice)

### 4.4 Potential Arc D: The Alliance (Lydia + Victor vs Marcus)

**Not guaranteed but structurally enabled.** If Lydia and Victor share their suspicions, they form a natural alliance against Marcus. This would be a SETUP -> COMPLICATION -> ESCALATION sequence as they gather evidence, followed by a TURNING_POINT when they confront him.

### 4.5 Potential Arc E: Thorne's Discovery

**The meta-arc.** Whatever path the evening takes, the central dramatic question is: does Thorne find out? His high pride (0.8) means that WHEN he discovers the betrayal(s), his reaction will be extreme — likely a catastrophe. The audience is waiting for this moment. The longer it's delayed, the higher the eventual impact (suppression_count building in other characters adds to this).

---

## 5. The Belief Matrix at Start

Reproduced from agents.md Section 9 for completeness. This is the engine of dramatic irony.

| Agent \ Secret | affair | embezzle | diana_debt | lydia_knows | victor_inv |
|---|---|---|---|---|---|
| **Thorne** | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| **Elena** | TRUE | UNKNOWN | SUSPECTS | UNKNOWN | UNKNOWN |
| **Marcus** | TRUE | TRUE | UNKNOWN | UNKNOWN | SUSPECTS |
| **Lydia** | SUSPECTS | SUSPECTS | UNKNOWN | TRUE | UNKNOWN |
| **Diana** | TRUE | UNKNOWN | TRUE | UNKNOWN | UNKNOWN |
| **Victor** | UNKNOWN | SUSPECTS | UNKNOWN | UNKNOWN | TRUE |

**Key irony observations:**
- Thorne is UNKNOWN on everything — maximum irony (5 unknowns about things that affect him directly)
- Marcus knows 2 secrets and suspects 1 — he's the most informed but also the most threatened
- Lydia suspects 2 things and knows 1 — she has partial knowledge that could become complete
- Victor and Lydia both suspect the embezzlement independently — they're unknowing allies
- Diana and Elena have complementary knowledge — Elena knows the affair, Diana knows the debt
- Nobody except Marcus knows ALL of Marcus's betrayals

**Information pathways for secret propagation:**
```
affair_01:    Elena → Diana (already done)
              Elena → Marcus (both know)
              Diana → anyone (guilt might push her)
              Lydia → anyone (she only suspects, needs confirmation)
              Observation → anyone who watches Elena and Marcus closely

embezzle_01:  Marcus → nobody willingly
              Lydia → Thorne (if she overcomes cowardice)
              Lydia → Victor (natural alliance if they talk)
              Victor → Thorne (if he confirms his suspicion)
              Victor → anyone (if vanity makes him hint)

diana_debt:   Diana → Elena (if guilt overwhelms)
              Marcus → anyone (if he weaponizes it)

lydia_knows:  Lydia → Thorne (loyal confession)
              Lydia → Victor (comparing notes)
              Lydia → Marcus (confrontation, unlikely given cowardice)

victor_inv:   Victor → anyone (vanity-driven hints)
              Marcus → anyone (if he exposes Victor to discredit)
```

---

## 6. Relationship Matrix at Start

### 6.1 Trust Matrix (A trusts B)

|  | Thorne | Elena | Marcus | Lydia | Diana | Victor |
|---|---|---|---|---|---|---|
| **Thorne** | - | 0.8 | 0.7 | 0.3 | 0.5 | 0.6 |
| **Elena** | 0.3 | - | 0.7 | 0.2 | 0.7 | 0.3 |
| **Marcus** | 0.2 | 0.5 | - | 0.1 | 0.2 | -0.3 |
| **Lydia** | 0.7 | 0.2 | -0.2 | - | 0.4 | 0.5 |
| **Diana** | 0.4 | 0.7 | 0.1 | 0.3 | - | 0.3 |
| **Victor** | 0.5 | 0.3 | -0.1 | 0.3 | 0.2 | - |

**Key asymmetries:**
- Thorne trusts Elena (0.8) but Elena barely trusts Thorne (0.3) — the affair has eroded her trust
- Thorne trusts Marcus (0.7) but Marcus barely trusts Thorne (0.2) — Marcus knows what he's doing
- Victor distrusts Marcus (-0.1) while Marcus distrusts Victor (-0.3) — mutual suspicion
- Lydia trusts Thorne (0.7) but Thorne barely knows Lydia (0.3) — the loyal assistant is invisible to the boss

### 6.2 Affection Matrix (A feels affection toward B)

|  | Thorne | Elena | Marcus | Lydia | Diana | Victor |
|---|---|---|---|---|---|---|
| **Thorne** | - | 0.6 | 0.4 | 0.2 | 0.3 | 0.3 |
| **Elena** | 0.1 | - | 0.8 | 0.1 | 0.5 | 0.1 |
| **Marcus** | 0.1 | 0.5 | - | 0.0 | 0.1 | -0.2 |
| **Lydia** | 0.4 | 0.1 | -0.1 | - | 0.3 | 0.2 |
| **Diana** | 0.3 | 0.6 | -0.1 | 0.2 | - | 0.2 |
| **Victor** | 0.4 | 0.1 | 0.0 | 0.1 | 0.1 | - |

**Key pattern:** Thorne's affection toward Elena (0.6) vs Elena's affection toward Thorne (0.1) and toward Marcus (0.8). This is the emotional core of the betrayal triangle.

---

## 7. Pacing State at Start

Characters don't all start at zero stress. Their pre-existing situations create asymmetric starting conditions.

| Agent | Budget | Stress | Composure | Commitment | Recovery | Rationale |
|---|---|---|---|---|---|---|
| Thorne | 1.0 | 0.1 | 0.9 | 0.0 | 0 | Relaxed host, no worries |
| Elena | 1.0 | 0.3 | 0.7 | 0.4 | 0 | Pre-stressed, committed to secrecy |
| Marcus | 1.0 | 0.4 | 0.8 | 0.6 | 0 | Most stressed, deeply committed to deceptions |
| Lydia | 1.0 | 0.2 | 0.8 | 0.3 | 0 | Mildly stressed by knowledge she carries |
| Diana | 1.0 | 0.15 | 0.85 | 0.3 | 0 | Slightly stressed by conflicting loyalties |
| Victor | 1.0 | 0.1 | 0.9 | 0.2 | 0 | Calm, focused — investigating, not stressed |

**Design note:** Marcus starts with the highest stress (0.4) AND highest commitment (0.6). His catastrophe potential at start: 0.4 * 0.6^2 = 0.144 — not near the 0.35 threshold yet, but with the least headroom. He's the most likely first catastrophe candidate.

Elena starts with the second-highest stress (0.3) and moderate commitment (0.4). Her potential: 0.3 * 0.4^2 = 0.048. She has more room, but her composure is lower (0.7) and will erode faster due to her guilt flaw increasing stress through over-accommodation.

---

## 8. Expected Narrative Phases

Based on the pacing constants, character designs, and typical interaction patterns, the dinner party should unfold in roughly these phases:

### Phase 1: "The Calm Surface" (Ticks 1-20, ~10 minutes sim-time)

**What happens:** Small talk, compliments, wine pouring, seating. Agents chat with neighbors. Victor observes Marcus. Elena is slightly anxious but composure holds. Lydia watches quietly.

**Pacing state:** Low stress across the board. Dramatic budget is full but nobody is motivated to spend it. The masking system enforces politeness at the dining table.

**Expected events:** ~15-25 events. Mostly CHAT, OBSERVE, PHYSICAL (drinks). Maybe 1-2 INTERNAL events from stressed characters.

### Phase 2: "Undercurrents" (Ticks 20-45, ~20 minutes sim-time)

**What happens:** Victor starts asking pointed questions about the business. Marcus deflects with charm but stress rises. Elena excuses herself to the kitchen or bathroom. Lydia notices something. Diana tries to keep everyone comfortable. Thorne is oblivious and happy.

**Pacing state:** Stress building for Marcus (0.5+) and Elena (0.4+). Composure starting to erode from alcohol (everyone's had 2-3 drinks). Victor's obsession flaw starts driving more aggressive probing. First dramatic actions may appear (a private CONFIDE or a pointed question that counts as mild CONFLICT).

**Expected events:** ~30-50 events. CHAT, OBSERVE, SOCIAL_MOVE (characters start moving around), first CONFIDE or REVEAL events. 1-2 minor conflicts.

### Phase 3: "Cracks Appear" (Ticks 45-75, ~25 minutes sim-time)

**What happens:** Composure is dropping below masking thresholds for stressed characters. The social mask starts to slip. Marcus snaps at someone. Elena's body language toward Marcus becomes noticeable. Victor pushes too hard and Marcus retaliates. Lydia considers speaking up.

**Pacing state:** Marcus approaching catastrophe territory (stress 0.6+, commitment 0.7+). Elena's composure below 0.5. Dramatic budget has been used and partially recharged. Recovery timers cycling.

**Expected events:** ~30-50 events. CONFLICT events appear. REVEAL and LIE events. SOCIAL_MOVE becomes strategic (seeking private spaces). OBSERVE events become more significant (spotting cracks in facades).

### Phase 4: "The Break" (Ticks 75-100, ~20 minutes sim-time)

**What happens:** A catastrophe event (most likely Marcus or Elena). The explosion triggers a cascade: secrets are revealed, relationships reconfigure, alliances form and break. This is the turning point of the evening.

**Pacing state:** At least one catastrophe fires. Stress spikes for witnesses. Composure crashes. Recovery timers activate, forcing a brief calm.

**Expected events:** ~25-40 events. 1-2 CATASTROPHE events. Multiple REVEAL events triggered by the catastrophe. CONFLICT events between newly opposed characters. SOCIAL_MOVE as characters flee or pursue.

### Phase 5: "Aftermath" (Ticks 100-150, ~15 minutes sim-time)

**What happens:** The consequences play out. Characters confront each other with new knowledge. Alliances solidify or break. Some characters may leave the party. The evening winds down toward a new, damaged equilibrium.

**Pacing state:** Stress elevated but budget depleted. Recovery timers winding down. Commitment very high for involved characters. Second catastrophes possible but rare (cooldown prevents immediate repeats).

**Expected events:** ~20-40 events. CONFLICT, CHAT (processing what happened), SOCIAL_MOVE (leaving), INTERNAL (reflection). The event rate decreases as the evening winds down.

---

## 9. Scenario Configuration Object

The complete configuration to initialize the simulation:

```python
dinner_party_config = ScenarioConfig(
    world=WorldDefinition(
        id="dinner_party_01",
        name="The Dinner Party",
        description="Six guests, five rooms, five secrets, one evening.",
        sim_duration_minutes=150.0,
        ticks_per_minute=2.0,
        locations=DINNER_PARTY_LOCATIONS,     # from world.md Section 2
        secrets=DINNER_PARTY_SECRETS,         # from world.md Section 6
        seating=SEATING_ADJACENCY,            # from world.md Section 3
        primary_themes=["loyalty_betrayal", "truth_deception"],
        snapshot_interval=20,
        catastrophe_threshold=0.35,           # from pacing-physics.md
        composure_minimum=0.30,               # from pacing-physics.md
        trust_repair_multiplier=3.0,
    ),
    agents=[
        THORNE_INITIAL_STATE,                 # from agents.md Section 9
        ELENA_INITIAL_STATE,
        MARCUS_INITIAL_STATE,
        LYDIA_INITIAL_STATE,
        DIANA_INITIAL_STATE,
        VICTOR_INITIAL_STATE,
    ],
    pacing_constants=PacingConstants(),        # from pacing-physics.md Section 3
    random_seed=42,                           # deterministic by default
    max_ticks=300,
    max_sim_time=150.0,
)
```

**Note on catastrophe constants:** The world.md spec uses `catastrophe_threshold=0.6` and `composure_minimum=0.2` while pacing-physics.md uses `0.35` and `0.30` respectively. The pacing-physics values were designed to produce catastrophes in the tick 50-80 range for the dinner party. The world.md values would delay catastrophes until much later, potentially past the end of the simulation. **Recommendation: use the pacing-physics.md values (0.35, 0.30).** This is flagged as a reconciliation item.

---

## 10. Validation Criteria

A successful dinner party simulation run should:

1. **Produce 100-200 events** (per CLAUDE.md MVP spec)
2. **Include at least 1 catastrophe event** (the pacing system should guarantee this given the starting conditions)
3. **Have at least 3 different event types** appear in significant quantity (CHAT > 20, CONFLICT > 5, at least 2 REVEAL or CONFIDE)
4. **Show escalation:** the average tension of events in the second half should be higher than the first half
5. **Show spatial variety:** at least 3 of the 5 locations should have events
6. **Not terminate before tick 40** (if it ends too early, starting conditions or constants need adjustment)
7. **Produce at least 2 recognizable narrative arcs** when the event log is analyzed by the story extraction pipeline

---

## 11. Reconciliation Notes

Differences between agents.md, world.md, and the simulation specs that need resolution:

| Topic | agents.md / world.md | pacing-physics.md | Resolution Needed |
|---|---|---|---|
| Catastrophe threshold | 0.6 | 0.35 | Use 0.35 (pacing-physics designed around it) |
| Composure minimum | 0.2 | 0.30 | Use 0.30 (pacing-physics designed around it) |
| PacingState fields | No suppression_count | Has suppression_count | Add to agents.md |
| Budget recharge rate | 0.05/tick | 0.08/tick | Use 0.08 (pacing-physics is authoritative) |
| Commitment decay | No passive decay | 0.01/tick below 0.50 | Use pacing-physics rules |
| Catastrophe aftermath | commitment reset to 0 | commitment +0.10 | Use pacing-physics (more nuanced) |
| Recovery timer on catastrophe | 10 ticks | 8 ticks | Use 8 (pacing-physics is authoritative) |

**Principle:** The pacing-physics.md spec is authoritative for all pacing-related constants and update rules. The agents.md spec defines the data structures. Where they conflict on constants or behavior, pacing-physics.md prevails.

---

## 12. Relationship to Other Specs

| Spec | Relationship |
|---|---|
| **agents.md** | Contains the 6 character JSON definitions referenced here. |
| **world.md** | Contains locations, secrets, seating, and WorldDefinition. |
| **pacing-physics.md** | Defines the pacing constants used in initial state and behavior. |
| **decision-engine.md** | Uses GoalVector and CharacterFlaw values defined here (via agents.md) to score actions. |
| **tick-loop.md** | Uses this configuration as the initial world state for `initialize_world()`. |
| **fake-data-visual-spec.md** | Will use these characters and expected narrative phases to design the 70-event visual test trace. |
