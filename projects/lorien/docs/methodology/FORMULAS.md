# Appendix A: Formulas

> All constants and formulas with `file:line` citations.

## A.1 Decision Engine — Utility Computation

### Per-Agent Score (`decision_engine.py:1037-1044`)

```
Score = base_utility(agent, action, world)
      + flaw_bias(agent, action, perception, world)
      + pacing_modifier(agent, action, world, location)
      + relationship_modifier(agent, action, world)
      - recency_penalty(agent, action, world)
      + action_noise(rng)
```

**SOCIAL_MOVE penalty**: -0.20 applied after scoring (`decision_engine.py:1045-1046`).

**Selection**: Argmax with deterministic tiebreak — sort by `(-score, action_key)` (`decision_engine.py:1049-1050`).

### Base Utility

Goal-vector dot product: each action type has an `ActionEffects` profile mapping to 7 goal dimensions (safety, status, closeness, secrecy, truth, autonomy, loyalty). Utility = dot product of agent goals × action effects (`decision_engine.py:14-40`).

### Cross-Agent Action Selection (`tick_loop.py:224-262`)

Softmax sampling over resolved actions:

```
temperature = 0.4

Adjusted score penalties before softmax:
  INTERNAL:    -0.35
  OBSERVE:     -0.10
  SOCIAL_MOVE: -0.20

weights[i] = exp((score[i] - max_score) / temperature)
```

Weighted random selection without replacement by agent (`tick_loop.py:241-259`).

---

## A.2 Pacing Update Rules

All constants defined in `PacingConstants` (`pacing.py:14-73`).

### Dramatic Budget (`pacing.py:133-167`)

```
For each event sourced by this agent:
  CONFLICT:     budget -= 0.35
  CONFIDE:      budget -= 0.15
  REVEAL:       budget -= 0.35 (major) or 0.15 (minor)
  LIE:          budget -= 0.35
  CATASTROPHE:  budget -= 0.50

If no dramatic action this tick:
  budget += 0.08 (base recharge)
  if location.privacy >= 0.3:
    budget += 0.04 (private bonus)

budget = clamp(budget, 0.0, 1.0)
```

Major reveal: secret with `dramatic_weight >= 0.8` (`pacing.py:120-130`).

### Stress (`pacing.py:223-273`)

```
For each event this tick:
  CONFLICT involving agent:  stress += 0.12
  CONFLICT witnessed:        stress += 0.05
  CONFLICT overheard:        stress += 0.03
  CATASTROPHE (same rules as CONFLICT)
  REVEAL/CONFIDE target + disturbing secret: stress += 0.08
  LIE target:               stress += 0.10

If agent NOT in conflict/catastrophe:
  stress -= 0.03 (base decay)
  if location.privacy >= 0.3:
    stress -= 0.02 (private bonus)

stress = clamp(stress, 0.0, 1.0)
```

Disturbing secret: about the agent, or dramatic_weight >= 0.5 (`pacing.py:216-220`).

### Composure (`pacing.py:189-204`)

```
composure -= 0.06 × drink_count
if stress > 0.5:
  composure -= 0.02
if location.privacy >= 0.3 AND stress < 0.3 AND no conflict:
  composure += 0.01

composure = clamp(composure, 0.05, 1.0)
```

Drink count is determined by AGENT_RESOURCE deltas on `alcohol_level` (`pacing.py:170-186`).

### Commitment (`pacing.py:283-313`)

```
For each event sourced by this agent:
  CHAT in public + contains opinion: commitment += 0.10
  CONFLICT:                          commitment += 0.15
  REVEAL:                            commitment += 0.20

If commitment <= 0.50:
  commitment -= 0.01 (decay)

commitment = clamp(commitment, 0.0, 1.0)
```

### Recovery Timer (`pacing.py:316-333`)

```
For each event sourced by this agent:
  CATASTROPHE: timer = max(timer, 6)
  CONFLICT/LIE: timer = max(timer, 4)
  REVEAL/CONFIDE: timer = max(timer, 2)

if timer > 0:
  timer -= 1
```

### Suppression Count (`pacing.py:336-349`)

```
if agent sourced a dramatic action:
  return 0  (reset)
if stress >= 0.60:
  return count + 1
else:
  return count  (unchanged)
```

### Catastrophe Potential (`pacing.py:91-107`)

```
potential = stress × commitment² + suppression_count × 0.03

Fires when ALL of:
  potential >= 0.35
  composure < 0.30
  recovery_timer == 0
```

### Catastrophe Aftermath (`pacing.py:352-414`)

```
stress     = stress × 0.5
composure  = 0.30
commitment += 0.10  (if < 0.80)
recovery_timer = 8
suppression_count = 0
```

---

## A.3 Tension Components

All in `tension.py`. Default weight = 1/8 for all 8 components (`tension.py:32-40`).

### Danger (`tension.py:65-95`)

```
physical:
  CATASTROPHE → 1.0
  CONFLICT    → 0.8
  OBSERVE (overheard CONFLICT)     → 0.45
  OBSERVE (overheard CATASTROPHE)  → 0.65

social:
  Trust delta < -0.3 → |delta| (capped at 1.0)
  REVEAL → 0.7
  LIE    → 0.5
  CONFIDE → 0.3

combined = max(physical, social) + 0.15 × min(physical, social)
```

### Time Pressure (`tension.py:98-132`)

```
scores = []

# Evening progression (quadratic ramp)
progress = sim_time / max_sim_time
scores.append(progress²)

# Recovery pressure
if recovery_timer > 0:
  scores.append((1 - recovery_timer/8) × 0.6)

# Secret convergence: count agents who suspect each relevant secret
if suspect_count >= 2:
  scores.append(suspect_count / 4.0)

# Composure loss
if (1 - composure) > 0.3:
  scores.append((1 - composure) × 0.5)

result = max(scores)
```

### Goal Frustration (`tension.py:135-143`)

```
0.6 × stress + 0.4 × (1 - dramatic_budget)
```

### Relationship Volatility (`tension.py:146-160`)

```
current_abs = sum(|delta.value| for relationship deltas in event)
current_scaled = current_abs / 0.6  (clamped to [0,1])

recent_abs = sum(|delta.value| for relationship deltas in recent events)
recent_scaled = recent_abs / (0.6 × len(recent_events))

result = max(current_scaled, recent_scaled)
```

Recent events: last 5 events involving the same participants, from a 30-event rolling window (`tension.py:347-356`).

### Information Gap (`tension.py:163-190`)

```
For each secret:
  weight = dramatic_weight (clamped, min 0.1)
  states = unique belief states among agents present
  diversity = (len(states) - 1) / 3.0

result = weighted_sum(diversity × weight) / sum(weights)
```

4 possible belief states → diversity: 2 states=0.33, 3=0.67, 4=1.0.

### Resource Scarcity (`tension.py:193-199`)

```
budget_scarcity = 1 - dramatic_budget
composure_loss  = 1 - composure
timer = recovery_timer / 8.0

result = max(budget_scarcity, composure_loss) × 0.85 + timer × 0.15
```

### Moral Cost (`tension.py:202-219`)

```
CATASTROPHE → 0.9
LIE         → 0.5 + 0.5 × truth_seeking
CONFLICT    → 0.35 + 0.25 × loyalty
REVEAL      → 0.25 + 0.5 × secrecy
CONFIDE     → 0.2
else        → 0.0
```

### Irony Density (`tension.py:222-224`)

```
irony / 2.0  (clamped to [0,1])
```

### Final Tension Score (`tension.py:383-389`)

```
tension = sum(weight[k] × component[k] for k in 8 components)
tension = clamp(tension, 0.0, 1.0)
```

Weights are normalized to sum to 1.0 (`tension.py:58-62`).

---

## A.4 Irony Scoring

### Per-Agent Irony (`irony.py:29-74`)

```
For each secret:
  relevance = secret_relevance(secret, agent, agents_present)
  if relevance < 0.1: skip

  if belief is actively wrong:  score += 2.0 × relevance
  elif unknown + relevant:      score += 1.5 × relevance
  elif unknown:                 score += 0.5 × relevance
  elif suspects (true secret):  score += 0.25 × relevance
  elif correct:                 score += 0.0
```

### Secret Relevance (`irony.py:14-26`)

```
about_self:      1.0
holder:          0.7
about_present:   0.5
other:           0.2
```

### Scene Irony (`irony.py:77-85`)

```
scene_irony = mean(agent_irony(a) for a in agents_present)
```

### Irony Collapse Detection (`irony.py:159-169`)

```
drop = prev_scene_irony - curr_scene_irony
if drop >= 0.5:
  collapse = IronyCollapseInfo(detected=True, drop=drop, score=drop)
```

---

## A.5 Force-Directed Layout

### Parameters (`threadLayout.ts:33-46`)

| Parameter | Default | Unit |
|-----------|---------|------|
| `attractionStrength` | 0.3 | force coefficient |
| `repulsionStrength` | 0.2 | force coefficient |
| `interactionBonus` | 0.5 | added to attraction when agents interact |
| `laneSpringStrength` | 0.1 | spring constant back to base lane |
| `inertia` | 0.7 | weight of previous position |
| `minSeparation` | 20 | pixels |
| `lanePadding` | 40 | pixels |
| `iterations` | 50 | max per time-sample |
| `convergenceThreshold` | 0.5 | max delta to stop early |
| `timeResolution` | 0.5 | minutes between samples |

### Force Computation (`threadLayout.ts:207-266`)

Per iteration, for each agent pair (a, b):

```
if same_location:
  f = (attraction + interaction_bonus_if_active) × distance × direction
else:
  f = -repulsion × (minSeparation / max(distance, minSeparation)) × direction

if distance < minSeparation:
  push = (minSeparation - distance) / 2  (separation enforcement)
```

Plus per-agent:
```
lane_spring = laneSpringStrength × (baseLane - currentY)
inertia_force = inertia × (prevY - currentY)
```

Applied with 0.1 damping factor (`threadLayout.ts:260`).
