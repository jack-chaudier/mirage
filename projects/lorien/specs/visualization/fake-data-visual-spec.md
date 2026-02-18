# Fake Data Visual Specification

**Status:** FINAL (references canonical events.md, agents.md, tension-pipeline.md)
**Author:** viz-architect
**Dependencies:** specs/schema/events.md (#1), specs/schema/agents.md (#2), specs/simulation/dinner-party-config.md (#8)
**Consumers:** Phase 1 coding agent (builds the renderer to match this visual description)
**Cross-references:** doc3.md Decisions 13 (3-tier fake data), 15 (precomputed hover), 16 (scene segmentation)

---

## 1. Purpose

This document describes **what the user SEES** when the Phase 1 renderer loads the 70-event fake dinner party dataset. It is the "concept art" for the visualization — a concrete target the coding agent builds toward. Every visual element described here maps to a data structure defined in renderer-architecture.md and an interaction defined in interaction-model.md.

---

## 2. The Dinner Party: Characters and Setup

Six characters. One evening (~120 minutes sim time). Five locations. From agents.md Section 9:

| Thread | Character | Color | Role | Core Secret |
|--------|-----------|-------|------|-------------|
| 1 | James Thorne | `#E69F00` (warm orange) | Host, wealthy businessman | Knows nothing — maximum irony target |
| 2 | Elena Thorne | `#56B4E9` (sky blue) | Thorne's wife | Affair with Marcus; guilt-ridden |
| 3 | Marcus Webb | `#009E73` (teal green) | Business partner | Affair + embezzlement; most secrets |
| 4 | Lydia Cross | `#F0E442` (yellow) | Thorne's assistant | Suspects both secrets; too afraid to speak |
| 5 | Diana Forrest | `#0072B2` (deep blue) | Elena's friend | Knows about affair; owes Marcus debt |
| 6 | Victor Hale | `#D55E00` (vermillion) | Journalist | Investigating Marcus; doesn't know about affair |

**Locations:** dining_table (privacy 0.1, capacity 6), kitchen (privacy 0.5, capacity 3), balcony (privacy 0.7, capacity 3), foyer (privacy 0.6, capacity 4), bathroom (privacy 0.9, capacity 1).

---

## 3. The 70-Event Dataset: Three Tiers (Decision 13)

### Tier 1: Story-Critical Events (~20 events)

These form the causal spine of the narrative. High tension, clear cause-and-effect chains.

| # | Time | Type | Characters | Location | Description | Tension |
|---|------|------|-----------|----------|-------------|---------|
| 1 | 0.0 | CHAT | Thorne → All | dining_table | Thorne welcomes guests, proposes a toast | 0.05 |
| 5 | 5.0 | OBSERVE | Lydia → (internal) | dining_table | Lydia notices Marcus checking his phone nervously | 0.15 |
| 12 | 15.0 | SOCIAL_MOVE | Elena → (move) | dining_table→kitchen | Elena excuses herself to "check on dessert" | 0.12 |
| 14 | 18.0 | CONFIDE | Elena → Marcus | kitchen | Elena whispers fear that Thorne suspects something | 0.36 |
| 16 | 20.0 | OBSERVE | Lydia → (internal) | kitchen | Lydia overhears Elena and Marcus whispering | 0.35 |
| 22 | 30.0 | CONFIDE | Elena → Diana | bathroom | Elena confides about the affair to Diana | 0.40 |
| 27 | 38.0 | INTERNAL | Diana → (internal) | dining_table | Diana realizes her debt to Marcus traps her | 0.32 |
| 31 | 42.0 | LIE | Marcus → Thorne | foyer | Marcus deflects Thorne's question about finances | 0.45 |
| 35 | 48.0 | OBSERVE | Victor → (internal) | dining_table | Victor notices the financial discrepancy hints | 0.38 |
| 38 | 52.0 | SOCIAL_MOVE | Thorne → (move) | dining_table→balcony | Thorne steps out to "get air" (actually upset) | 0.28 |
| 41 | 55.0 | CONFLICT | Victor → Marcus | dining_table | Victor pointedly asks about the firm's accounts | 0.55 |
| 44 | 60.0 | SOCIAL_MOVE | Marcus → (move) | dining_table→balcony | Marcus follows Thorne to the balcony | 0.42 |
| 47 | 65.0 | CONFLICT | Thorne → Marcus | balcony | Thorne confronts Marcus about the ledger | 0.72 |
| 50 | 72.0 | REVEAL | Lydia → Thorne | balcony | Lydia finally tells Thorne what she overheard | 0.68 |
| 53 | 78.0 | OBSERVE | Elena → (internal) | dining_table | Elena sees Lydia on the balcony with Thorne, panics | 0.58 |
| 56 | 85.0 | LIE | Marcus → All | dining_table | Marcus tries to publicly deny everything | 0.62 |
| 60 | 92.0 | CATASTROPHE | Elena → All | dining_table | Elena breaks down, confesses the affair | 0.95 |
| 63 | 98.0 | CONFLICT | Thorne → Marcus | dining_table | Thorne turns on Marcus — double betrayal | 0.88 |
| 66 | 105.0 | SOCIAL_MOVE | Thorne → (move) | dining_table→foyer | Thorne storms toward the front door | 0.70 |
| 68 | 110.0 | INTERNAL | Diana → (internal) | dining_table | Diana realizes she's free of Marcus's leverage | 0.35 |

### Tier 2: Texture Events (~40 events)

Low-tension social filler that makes the evening feel real. Small talk, pouring wine, passing dishes, going to the bathroom. These fill the gaps between story-critical events.

Examples:
- CHAT: Victor and Diana discuss art (t=8)
- PHYSICAL: Thorne pours wine for guests (t=3, t=25, t=50 — alcohol_level rises)
- CHAT: Lydia and Victor small talk about journalism (t=10)
- SOCIAL_MOVE: Diana goes to bathroom (t=20)
- PHYSICAL: Marcus refills his wine glass (t=35)
- CHAT: General table conversation about travel (t=45)
- PHYSICAL: Elena clears plates (t=55)

Tension range: 0.02 - 0.18

### Tier 3: Ambiguity Events (~10 events)

Deliberately messy events that test renderer robustness:
- Simultaneous conversations (2 CHAT events at same tick_id, different pairs)
- A misheard observation (Lydia thinks she hears her name — OBSERVE with low confidence)
- Contradictory signals (Marcus smiles while angry — emotion delta contradicts dialogue)
- An event with no causal links to anything before (Victor gets a text from outside)
- An event at the boundary of two scenes (character transitions mid-conversation)

Tension range: 0.05 - 0.30

---

## 4. What the User SEES: Full Canvas Description

### 4.1 Initial Load (Threads Zoom Level, Default Weights)

The canvas fills the screen (assume 1400x800px viewport). The X-axis spans 0-120 minutes. The Y-axis distributes 6 character threads.

**Overall impression:** Six colored threads flowing left to right, like a braided river. The threads cluster in the center during the first half (everyone at the dining table), diverge when characters leave, and reconverge dramatically before the climax. The right third of the canvas glows warm (amber/red tension heatmap) while the left third is cool/dim.

#### Thread Layout (top to bottom, initial order)

```
Y=80px   ── Thorne (orange) ─────────────────────────────────────
Y=180px  ── Elena (sky blue) ─────────────────────────────────────
Y=280px  ── Marcus (teal) ──────────────────────────────────────
Y=380px  ── Lydia (yellow) ─────────────────────────────────────
Y=480px  ── Diana (deep blue) ──────────────────────────────────
Y=580px  ── Victor (vermillion) ────────────────────────────────
```

#### Phase-by-phase visual description

**Minutes 0-15: "The Gathering" (Scene 1)**
- All 6 threads run in a tight cluster (Y spread ~250px, compressed from the ~500px full spread)
- Thread thickness: thin (2-3px). Low tension events.
- Event nodes: small dots (4-5px radius), mostly grey-blue (CHAT type)
- A few gold dots (OBSERVE: Lydia's observation at t=5)
- Background: cool blue-grey tint. Scene boundary at t=0 labeled "Arrival / Dining Table"
- **The cluster communicates "everyone together, nothing happening yet"**

**Minutes 15-30: "The Kitchen Meetings" (Scene 2)**
- Elena's sky blue thread dips downward away from the cluster (she moves to the kitchen)
- Marcus's teal thread follows her 3 minutes later
- The two threads converge at the bottom of the canvas (~Y=450) for t=18 (the confide event)
- Lydia's yellow thread subtly bends toward them (she goes to the kitchen to observe)
- A gold-colored dot on Lydia's thread at t=20 (OBSERVE: overhearing) gets a faint glow — this is the first event with non-trivial tension (0.35)
- Diana's thread briefly dips away at t=20 (bathroom trip, then returns)
- Elena's thread moves further down at t=22 (bathroom confide with Diana) — the two threads converge briefly at the very bottom
- Thorne's and Victor's threads remain at the top in a loose pair (still at the dining table)
- **The visual shows "secrets being shared in private spaces" — threads pulling away from the main group**

**Minutes 30-52: "Escalating Suspicion" (Scene 3)**
- Threads reconverge toward center as characters return to the dining table
- Marcus's teal thread shows a slight thickening (rising tension from the LIE at t=42)
- A brown dot on Marcus's thread (LIE event at t=42) has a moderate amber glow
- Victor's vermillion thread begins to thicken (his investigation is heating up)
- At t=48, Victor's thread shows a red dot (CONFLICT: asking about accounts) — this is the first openly confrontational event, tension 0.55
- At t=52, Thorne's orange thread dips away from the cluster (moves to balcony)
- **The canvas is warming up: event nodes are getting larger, glow halos are appearing, thread thickness is increasing**

**Minutes 52-78: "The Balcony Confrontation" (Scene 4)**
- The most visually dramatic pre-catastrophe segment
- Thorne's orange thread moves sharply upward or downward (away from the dining table group) — he's on the balcony
- Marcus's teal thread follows him (t=60) — the two threads converge at the balcony position, now isolated from others
- **At t=65: the Thorne-Marcus CONFLICT event** — a large red node (radius ~9px) with a hot amber glow (tension 0.72). This is the visual peak before the catastrophe.
- The other 4 threads remain at the dining table, loosely clustered, with scattered texture events
- At t=72, Lydia's yellow thread makes a bold move — it arcs from the dining table cluster toward the balcony, converging with Thorne and Marcus. Her REVEAL event (telling Thorne what she overheard) is a gold dot with amber glow (tension 0.68)
- Elena's sky blue thread shows an OBSERVE event at t=78 — she sees Lydia on the balcony. A purple dot (INTERNAL) with growing glow
- **The visual shows "the crisis is brewing on the balcony while everyone else waits"**

**Minutes 78-95: "The Unraveling" (Scene 5)**
- All threads begin converging back to the dining table
- Marcus's denial (t=85) is a brown dot (LIE) but large and glowing — tension 0.62
- The tension heatmap in the background is now solid amber-red across the full canvas width
- Thread thickness for Elena, Marcus, and Thorne is at maximum (5-6px)
- Event node frequency increases — events are happening faster

**Minute 92: THE CATASTROPHE**
- Elena's sky blue thread gets a massive node: **bright red circle, radius 10px, with a pulsing red glow that extends 15px outward**
- The CATASTROPHE event is visually unique:
  - Double-width glow ring
  - All other threads show a brief "shockwave" — a subtle radial opacity pulse outward from the catastrophe node
  - Thread thickness of ALL characters spikes momentarily
  - The background heatmap reaches its hottest point directly behind this node (deep red)
- Tension: 0.95. This is the visual climax.

**Minutes 95-120: "The Aftermath" (Scene 6)**
- Thorne's orange thread makes a sharp move away (storming to the foyer, t=105)
- The CONFLICT between Thorne and Marcus at t=98 is a large red node (tension 0.88) — still very hot
- After t=105, tension begins to drop. Thread thickness decreases.
- The background heatmap cools from red back toward amber
- Diana's deep blue INTERNAL event at t=110 (realization about freedom) is a small purple dot with a cool blue glow — a moment of quiet resolution amid the chaos
- The final few events are scattered texture events (people gathering belongings, awkward silence)
- **The visual shows "the party disintegrating" — threads spreading apart, cooling colors, thinning lines**

### 4.2 The Tension Heatmap (Background Layer)

A Gaussian-blurred color field behind all threads. Interpolated from per-event tension values.

**Color scale:**
- 0.0-0.2: transparent (no visible tint)
- 0.2-0.4: faint cool blue (`#1a2a4a` at 15% opacity)
- 0.4-0.6: warm amber (`#cc8844` at 20% opacity)
- 0.6-0.8: hot orange-red (`#cc4422` at 25% opacity)
- 0.8-1.0: intense red (`#cc2222` at 35% opacity)

**What the user sees:** The left third of the canvas (minutes 0-40) is mostly transparent with faint cool patches. The center (40-80) transitions to warm amber. The right third (80-120) is a visible warm glow that peaks at a bright red hotspot around the catastrophe at t=92.

The heatmap is wide (Gaussian blur sigma = 30px in x, 60px in y) — it's an atmospheric backdrop, not precise per-event coloring.

### 4.3 Scene Boundaries

Thin vertical dashed lines at scene transitions. Subtle background tint alternates between scenes (odd scenes slightly darker).

```
Scene 1 "Arrival"       |  Scene 2 "Kitchen"    |  Scene 3 "Suspicion"
t=0 ─────────────── t=15 ──────────────── t=30 ──────────────── t=52

Scene 4 "Balcony"       |  Scene 5 "Unraveling"  |  Scene 6 "Aftermath"
t=52 ─────────────── t=78 ──────────────── t=95 ──────────────── t=120
```

At "Threads" zoom level, scene labels appear at the top of each scene region. At "Cloud" zoom, the scene structure is visible as the primary organizational element.

---

## 5. Hover Interactions: What the User SEES

### 5.1 Hovering the Catastrophe Event (evt_0060, t=92)

The user moves their mouse over the bright red CATASTROPHE dot on Elena's thread.

**Immediate visual response (< 16ms):**

1. All events NOT in the BFS-3 causal cone dim to 15% opacity
2. Backward cone highlights (blue tint):
   - evt_0053 (Elena OBSERVE: sees Lydia on balcony) — the direct cause
   - evt_0050 (Lydia REVEAL to Thorne) — caused Elena's panic
   - evt_0047 (Thorne-Marcus CONFLICT on balcony) — caused Lydia's reveal
   - evt_0044 (Marcus moves to balcony)
   - evt_0031 (Marcus LIE to Thorne) — the lie that started unraveling
   - ~3 more events in the causal chain
   These appear as dots with blue glow at 70% opacity, connected by faint blue lines along their threads
3. Forward cone highlights (amber tint):
   - evt_0063 (Thorne-Marcus CONFLICT: double betrayal) — direct consequence
   - evt_0066 (Thorne storms out) — consequence of consequence
   - evt_0068 (Diana's realization) — ripple effect
   These appear with amber glow at 85% opacity
4. The hovered catastrophe event: full brightness, white ring, 1.3x scale
5. Thread segments connecting causal cone events glow with their respective colors

**Tooltip appears:**
```
┌──────────────────────────────────────┐
│ [!] CATASTROPHE                       │
│ Elena breaks down and confesses the   │
│ affair with Marcus                    │
│ Time: 92m | Dining Table              │
│ All present: Thorne, Marcus, Lydia,   │
│   Diana, Victor                       │
│ Tension: ████████████████░ 0.95      │
│ [turning_point]                       │
└──────────────────────────────────────┘
```

**What this reveals:** The user can instantly trace the causal chain backward — from the catastrophe through Lydia's reveal, through the balcony confrontation, back to Marcus's lie. The forward cone shows the immediate fallout. The "shape of the story" becomes visible: a web of causation that converges on this single explosive moment.

### 5.2 Hovering a Texture Event (evt_0020, t=25, PHYSICAL: Thorne pours wine)

1. Very small causal cone (maybe 1-2 events forward, 0-1 backward)
2. Most of the canvas dims, but the dimming is less dramatic because fewer events are highlighted
3. The texture event itself glows only slightly (tension 0.05, minimal glow)
4. Tooltip shows simple content: "Thorne refills wine glasses. Time: 25m | Dining Table | Tension: 0.05"
5. **The user quickly learns: texture events have thin causal connections. Story-critical events have rich ones. This teaches the visual language.**

---

## 6. Crystallization: What the User SEES

### 6.1 Clicking Marcus's Thread (Crystallizing the Villain's Arc)

The user clicks on any event on Marcus's teal thread.

**Ripple (0-50ms):** A teal circle expands from the click point.

**Solidify (50-150ms):** Marcus's teal thread brightens to full saturation, thickens to 1.5x, gains a subtle drop shadow. All of Marcus's events become fully opaque with labels appearing (at Threads zoom):
- t=0: (arrival)
- t=18: "Whispers with Elena in kitchen" (CONFIDE, gold dot)
- t=42: "Tells Thorne finances are fine" (LIE, brown dot with amber glow)
- t=60: "Follows Thorne to balcony" (SOCIAL_MOVE)
- t=65: "Confronted by Thorne about ledger" (CONFLICT, large red dot)
- t=85: "Publicly denies everything" (LIE, brown dot with glow)
- t=92: (involved in CATASTROPHE — appears as a shared event with Elena)
- t=98: "Thorne turns on Marcus — double betrayal" (CONFLICT)

**Relate (50-200ms):** Characters who interact with Marcus:
- Elena: 45% opacity (many shared events — affair partner)
- Thorne: 45% opacity (business partner, confrontations)
- Victor: 45% opacity (interrogator)
- Lydia: 30% opacity (few direct interactions)
- Diana: 20% opacity (minimal contact with Marcus directly)

**Annotate (150-250ms):** Beat labels appear on Marcus's key events:
- t=0: `[SETUP]`
- t=42: `[ESCALATION]`
- t=65: `[ESCALATION]`
- t=85: `[ESCALATION]`
- t=92: `[TURNING POINT]` (shared with Elena's catastrophe)
- t=98: `[CONSEQUENCE]`

**What the user sees:** Marcus's journey crystallizes — a thread of deception that escalates through three LIE/CONFLICT events, building toward the catastrophe. The rising glow intensity along his thread tells the tension story visually. His arc is classic tragedy: overcommitment to deception leading to exposure and ruin.

### 6.2 Clicking Elena's Thread (Crystallizing the Tragic Figure)

Elena's sky blue thread solidifies. Her arc shows:
- The kitchen confide with Marcus (early intimacy)
- The bathroom confide with Diana (seeking support)
- The observe event where she sees Lydia on the balcony (growing panic)
- The catastrophe (breakdown)

**Key visual contrast with Marcus:** Elena's thread has fewer events but they carry more emotional weight. Her tension curve rises more steeply — long stretches of low-tension existence punctuated by high-tension private moments. Marcus's arc shows steady escalation; Elena's shows suppression followed by sudden collapse.

---

## 7. Zoom Levels: What the User SEES

### 7.1 Cloud Level (Fully Zoomed Out)

The entire 120-minute timeline fits in the viewport. Individual event nodes are invisible.

**What the user sees:**
- 6 thin colored lines flowing left to right (1px each)
- The tension heatmap is the dominant visual: a cool-to-warm gradient across time
- A clear bright red hotspot at t=92 (the catastrophe)
- Scene boundary labels at the top: "Arrival", "Kitchen", "Suspicion", "Balcony", "Unraveling", "Aftermath"
- The thread bundle is tight on the left, loosens in the middle (characters dispersing), and reconverges before the hotspot

**This communicates:** The overall shape of the evening — a slow burn that explodes near the end. The user can immediately see "the interesting stuff is on the right" and zoom in there.

### 7.2 Threads Level (Default)

As described in Section 4. The full description above is at Threads zoom level.

### 7.3 Detail Level (Zoomed In to a Section)

Zoomed into the t=60-100 range (the most dramatic stretch).

**What the user sees:**
- Thread paths are thick (4-6px) with visible tension-driven thickness variation
- Every event node has a text label (truncated to 40 chars): "Thorne confronts Marcus about the ledge..."
- Thin directed arrows between causally linked events (grey, 50% opacity)
- The causal arrow from evt_0047→evt_0050→evt_0053→evt_0060 forms a visible chain across threads
- Scene annotations: "Balcony Confrontation — Thorne, Marcus, Lydia — CONFLICT" with a list of participants
- Individual delta summaries visible on hover (reason_display text)
- Beat type badges visible on labeled events: colored chips reading "ESCALATION", "TURNING POINT", etc.

**This communicates:** The blow-by-blow narrative — exactly what happened, who did what to whom, and why. A writer could read the labels in sequence and reconstruct the scene.

---

## 8. Filter Controls: What the User SEES

### 8.1 Hiding a Character (Toggle Lydia OFF)

Lydia's yellow thread fades to 5% opacity. Her event nodes vanish. The remaining 5 threads maintain their Y positions — no layout reflow. The causal cone, if hovering, excludes Lydia's events (they still exist in the data but aren't highlighted).

**Effect on the narrative view:** Removing Lydia removes the "silent observer" thread. The story now focuses on the actors (Marcus, Elena, Thorne) and the reactor (Victor). The balcony scene loses Lydia's REVEAL event from the causal chain — hovering the catastrophe now shows a shorter backward cone.

### 8.2 Switching to Thriller Preset

The user clicks "Thriller" in the tension preset panel. All 8 sliders animate to the thriller weights (danger=2.5, time_pressure=2.0, etc.).

**What changes:**
- The texture events (CHAT, PHYSICAL) become even dimmer (their sub-metrics are low on danger/time_pressure)
- The CONFLICT events glow brighter (danger weight amplified from 1.0 to 2.5)
- The overall heatmap shifts: the early portion becomes nearly invisible, the balcony confrontation (t=65) brightens significantly, the catastrophe remains hot
- Marcus's LIE events become less visually prominent (moral_cost weight drops from 1.0 to 0.5)
- Victor's CONFLICT (t=55) gets a bigger glow (his direct questioning is "dangerous" in the thriller lens)

**What the user sees:** The same dinner party, but the "thriller" view emphasizes physical confrontation and danger. The more introspective/relational beats (Elena's confides, Diana's internal realization) fade into the background. The story becomes "Victor hunts Marcus, Thorne finds out, confrontation explodes."

### 8.3 Switching to Mystery Preset

Information_gap weight = 2.5, irony_density = 2.0.

**What changes:**
- Lydia's OBSERVE events glow brighter (high information_gap — she knows things others don't)
- Victor's events glow brighter (he's investigating — high truth-seeking)
- The early part of the evening (t=0-30) becomes more prominent because the information asymmetry is HIGHEST when no one knows anything
- Elena's confide events glow strongly (information transfer events shift the gap)
- The catastrophe still glows hot but the pre-catastrophe LIE events (Marcus lying) become the most dramatic — they WIDEN the gap

**What the user sees:** A completely different narrative shape from the same data. The mystery view says "the real tension was in who knew what and when."

---

## 9. The Region Select: What the User SEES

### 9.1 Selecting the Balcony Confrontation (t=52 to t=78)

The user shift-drags from t=52 to t=78.

**During drag:** An amber semi-transparent rectangle extends across the full canvas height. A floating label reads: "52m - 78m | 14 events | 2 scenes"

**After release:** The rectangle solidifies. The BeatSheetPanel in the sidebar shows:

```
Beat Sheet: Balcony Confrontation
──────────────────────────────────

Scene 4: "Balcony" (t=52-78)
  Participants: Thorne, Marcus, Lydia
  Dominant theme: loyalty_betrayal

  [ESCALATION] t=55 — Victor asks about accounts (tension 0.55)
  [ESCALATION] t=65 — Thorne confronts Marcus (tension 0.72)
  [COMPLICATION] t=72 — Lydia reveals what she knows (tension 0.68)
  [ESCALATION] t=78 — Elena sees the balcony gathering (tension 0.58)

  Suggested protagonist: Thorne (highest tension variance)
  Tension arc: 0.28 → 0.42 → 0.72 → 0.68 → 0.58

  [Export as JSON] [Copy to Clipboard]
```

---

## 10. Visual Design Tokens

### Thread Style

| State | Thickness | Opacity | Dash | Shadow |
|-------|-----------|---------|------|--------|
| Default | 2-4px (tension-driven) | 0.6 | Solid | None |
| Crystallized | 3-6px (1.5x) | 1.0 | Solid | 2px blur, black 30% |
| Related to crystallized | 2-4px | 0.45 | Solid | None |
| Unrelated to crystallized | 1px | 0.12 | Dotted | None |
| Hover causal cone | 2-4px | 0.7-0.85 | Solid | None |
| Hidden (filtered out) | 1px | 0.05 | Dotted | None |

### Event Node Style

| State | Radius | Border | Glow |
|-------|--------|--------|------|
| Default | 4-8px (significance-driven) | 1px, event type color | Tension glow: 0-12px |
| Hovered | radius * 1.3 | 2px white ring | Max glow |
| Crystallized | radius * 1.2 | 1.5px, thread color | Tension glow |
| In causal cone (backward) | radius | 1px | Blue glow, 6px |
| In causal cone (forward) | radius | 1px | Amber glow, 6px |
| Dimmed | radius | 1px, 15% opacity | None |
| CATASTROPHE (special) | 10px fixed | 2px red ring | Pulsing red, 15px |

### Glow Color Scale

| Tension Range | Glow Color | Intensity |
|--------------|------------|-----------|
| 0.0 - 0.2 | None | 0% |
| 0.2 - 0.4 | `#4488cc` (cool blue) | 20% |
| 0.4 - 0.6 | `#cc8844` (warm amber) | 40% |
| 0.6 - 0.8 | `#cc6633` (orange) | 60% |
| 0.8 - 1.0 | `#cc4444` (red) | 80% |

---

## 11. Visual Landmarks the Coding Agent Must Hit

These are pass/fail criteria. If the rendered output doesn't match these, something is wrong.

1. **The catastrophe at t=92 MUST be the most visually prominent single element on the canvas.** Largest node, hottest glow, brightest color. It should be immediately identifiable from any zoom level.

2. **The thread convergence at t=0-15 MUST be visually distinct from the divergence at t=15-30.** The "everyone together" phase should look different from the "people splitting off" phase.

3. **Marcus's teal thread MUST show increasing thickness from left to right** as his tension accumulates through escalating lies and confrontations.

4. **Hovering the catastrophe MUST illuminate a clear backward chain** that stretches back at least 30 minutes of sim time (to the kitchen meetings). The chain should cross at least 3 different character threads.

5. **The thriller and relationship-drama presets MUST produce visibly different heatmaps** from the same data. The thriller view should make the CONFLICT events stand out; the drama view should make the CONFIDE and LIE events stand out.

6. **Texture events (Tier 2) MUST be present but NOT dominant.** At threads zoom, they should appear as small, dim dots between the larger, glowing story-critical events. They provide visual rhythm but don't distract.

7. **All 6 character threads MUST remain distinguishable at all times.** Even when 6 threads cluster at the dining table, the colors must be separable and the minimum Y-separation (20px) must hold.

8. **Scene boundaries MUST be visible without obscuring events.** Thin dashed vertical lines with subtle background tinting, not heavy borders.

---

## 12. NOT in Scope

- **Actual fake data JSON generation.** This spec describes what the output looks like. The fake data JSON file itself will be authored separately (either hand-crafted or generated by a script).
- **Animation/playback.** The renderer shows all events simultaneously. No time-based playback in Phase 1.
- **3D tension surface view.** 2D only in Phase 1.
- **Prose generation from selected regions.** The beat sheet panel shows structured data, not LLM-generated prose.
