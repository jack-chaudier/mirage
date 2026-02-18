# Interaction Model Specification

**Status:** FINAL (aligned with canonical events.md, agents.md, tension-pipeline.md)
**Author:** viz-architect
**Dependencies:** specs/schema/events.md (#1), specs/visualization/renderer-architecture.md (#9)
**Consumers:** Phase 1 coding agent, specs/visualization/fake-data-visual-spec.md (#12)
**Cross-references:** doc3.md Decisions 12, 15, 16; doc2.md "Wavefunction Collapse" interface

---

## Overview

Every user interaction in NarrativeField follows a consistent pattern: **trigger** (user action) → **data lookup** (what information is needed) → **visual response** (what changes on screen) → **performance target** (how fast). This document specifies all interactions for the Phase 1 renderer.

---

## 1. Hover: Causal Cone Highlight

The most frequent interaction. Must feel instant.

### Trigger

Mouse enters the hit region of an event node on the canvas.

### Data Lookup

1. Read `hoveredEventId` from hit canvas (pixel color decode → event index → O(1))
2. Fetch causal neighborhood from `CausalIndex.get(hoveredEventId)` → `{ backward: Set<string>, forward: Set<string> }` — O(1) precomputed lookup (Decision 15)
3. Fetch event details for tooltip: `EventStore.get(hoveredEventId)` — O(1)

### Visual Response

**Phase 1: Dim non-connected events (0-16ms)**
- All events NOT in the causal cone set their opacity to `0.15`
- All thread segments NOT connecting causal cone events set opacity to `0.15`
- Transition: immediate (no easing — hover must feel instant)

**Phase 2: Highlight backward cone (same frame)**
- Events in `backward` set: opacity `0.7`, add blue-tinted glow (`#4488cc`, radius 6px)
- Thread segments connecting backward events: opacity `0.7`, slight blue tint
- Visual metaphor: "what led to this moment" — cool color = past

**Phase 3: Highlight forward cone (same frame)**
- Events in `forward` set: opacity `0.85`, add amber-tinted glow (`#cc8844`, radius 6px)
- Thread segments connecting forward events: opacity `0.85`, slight amber tint
- Visual metaphor: "what follows from this" — warm color = future

**Phase 4: Highlight hovered event (same frame)**
- Hovered event: opacity `1.0`, full glow at event's tension color, radius `+4px`
- White ring border (`#ffffff`, 2px)
- Scale up to `1.3x` base radius

**Phase 5: Show tooltip (same frame)**
- HTML tooltip positioned 12px above the hovered event node
- Content:
  ```
  ┌─────────────────────────────┐
  │ [EventType icon] Description│
  │ Time: 19:45 | Kitchen       │
  │ Participants: Thorne, Lyra  │
  │ Tension: ████░░░░ 0.72     │
  │ [Beat type badge if present]│
  └─────────────────────────────┘
  ```
- Tooltip follows cursor if mouse moves within the same event's hit region
- Tooltip dismisses instantly on mouse leave

**Phase 6: Streamline animation (deferred, runs over 300ms)**
- Animate faint directional particles along causal link edges
- Backward: particles flow toward the hovered event (converging)
- Forward: particles flow away from the hovered event (diverging)
- Particle color matches cone tinting (blue backward, amber forward)
- This is a polish feature — render without it first, add when core hover works

### Performance Target

- **Total hover response: < 16ms** (must complete within single frame at 60fps)
- Hit detection: < 1ms (pixel read from offscreen canvas)
- Causal index lookup: < 0.1ms (precomputed Map.get)
- Canvas redraw: < 12ms (200 events, opacity changes only — no layout recomputation)

### Hover Exit

When mouse leaves all event hit regions:
- All events restore to their base opacity (determined by crystallization state or default 0.6)
- Tooltip hides immediately
- Streamline particles fade over 150ms
- Transition: immediate for opacity, 150ms fade for particles

### Fallback

If hit canvas decode fails (e.g., event nodes overlap at low zoom):
- Return the event with the highest `significance` score among overlapping candidates
- At "cloud" zoom level, hover is disabled for individual events (show density tooltip instead)

---

## 2. Click/Select: Arc Crystallization

The signature interaction. Clicking an event crystallizes its character's arc.

### Trigger

Mouse click (left button) on an event node.

### Data Lookup

1. Identify clicked event: same hit canvas mechanism as hover
2. Fetch the event's `source_agent` → this becomes the "crystallized" character
3. Fetch the character's full timeline: `ThreadIndex.get(source_agent)` → ordered `event_id[]`
4. Fetch causal neighborhoods for all events in the timeline (already precomputed)

### Visual Response — The Crystallization Sequence (350ms total)

**Frame 0-50ms: "Ripple" from click point**
- A circular ripple animation expands outward from the clicked event (radius 0 → 200px, opacity 0.5 → 0)
- Ripple color: the character's signature color
- Purpose: satisfying click feedback, draws attention

**Frame 50-150ms: "Solidify" the selected arc**
- The clicked character's thread transitions:
  - Opacity: `0.6 → 1.0`
  - Thickness: `base → base * 1.5`
  - Color saturation: `60% → 100%`
  - Add subtle drop shadow for depth
- All events on this thread:
  - Opacity: `0.6 → 1.0`
  - Radius: `base → base * 1.2`
  - Labels appear (event description text) if zoom level is "threads" or "detail"
- Easing: `ease-out`

**Frame 50-200ms: "Relate" connected characters**
- Characters who share events with the crystallized character:
  - Opacity: `0.6 → 0.45`
  - Their shared events get a subtle highlight ring in the crystallized character's color
  - Thread segments where they interact with the crystallized character: opacity `0.5`
- Characters with NO interaction with the crystallized character:
  - Opacity: `0.6 → 0.12`
  - Threads become thin dotted lines
- Easing: `ease-in-out`

**Frame 150-250ms: "Annotate" the arc**
- Beat type labels appear on the crystallized arc's key events (if `beat_type` is set):
  - Setup, Complication, Escalation, Turning Point, Consequence
  - Small badges above the event nodes, styled per beat type
- Scene boundaries along this character's path become more prominent
- Tension glow on the crystallized arc events intensifies proportionally

**Frame 250-350ms: "Settle"**
- All transitions complete, final state holds
- Side panel updates: EventDetailPanel shows clicked event, SceneListPanel highlights scenes containing this character

### Click on Empty Space (Decrystallization)

Clicking on canvas background (no event hit):
- Reverse the crystallization: all threads restore to default opacity `0.6`
- Labels hide, beat badges hide
- Duration: 200ms, easing `ease-in`
- `selectedArcAgentId` set to `null`

### Click on Different Character's Event

If a crystallized arc exists and user clicks an event from a different character:
- Decrystallize current arc (150ms)
- Crystallize new arc (350ms)
- Total transition: 500ms, overlapping animations

### Performance Target

- Click detection: < 1ms
- Crystallization animation: 350ms total (multi-phase)
- Side panel update: < 50ms (React state update + render)

### Fallback

If clicked event has no `source_agent` (should not happen per schema, but defensive):
- Show event detail in side panel without crystallizing an arc
- Log warning to console

---

## 3. Zoom Levels: Cloud → Threads → Detail

### Trigger

- Mouse wheel scroll (pinch-to-zoom on trackpad)
- Zoom control buttons in toolbar
- Double-click to zoom in (centered on click position)
- Keyboard: `+` / `-` keys

### Zoom Mechanics

Zoom is continuous (smooth scroll), but the visual presentation snaps between three discrete levels at thresholds. The thresholds and their visual configurations:

#### Cloud Level (scale 0.1 - 0.4)

**What the user sees:** A density overview of the entire narrative.

| Element | Appearance |
|---------|-----------|
| Threads | Thin lines (1px), character color at 40% opacity |
| Event nodes | Hidden — replaced by density heatmap |
| Tension heatmap | Visible — Gaussian-blurred color field (blue→amber→red) |
| Scene boundaries | Visible as subtle vertical dividers with scene type labels |
| Labels | None |
| Hover | Disabled for individual events. Hovering a region shows "N events, avg tension X" |

**Transition INTO cloud (from threads):** Scale crosses below 0.4
- Event nodes fade out (opacity 1.0 → 0.0 over 200ms)
- Tension heatmap fades in (opacity 0.0 → 1.0 over 200ms)
- Threads thin from variable width to 1px (200ms)

#### Threads Level (scale 0.4 - 1.5)

**What the user sees:** Character threads as distinct colored paths with event nodes as dots.

| Element | Appearance |
|---------|-----------|
| Threads | Variable thickness (2-5px), character color at 70% opacity, tension-driven width |
| Event nodes | Visible dots (radius 3-8px), colored by event type |
| Tension heatmap | Hidden — tension shown via thread thickness + event glow |
| Scene boundaries | Visible as background tint bands |
| Labels | Hidden (appear only on hover/crystallize) |
| Hover | Full causal cone highlight |

**Default zoom level.** This is where most interaction happens.

#### Detail Level (scale 1.5 - 5.0)

**What the user sees:** Full detail — event nodes with labels, causal link arrows, all annotations.

| Element | Appearance |
|---------|-----------|
| Threads | Variable thickness, full opacity |
| Event nodes | Large dots (radius 5-10px) with description labels |
| Causal link arrows | Visible — thin directed arrows between causally linked events |
| Scene boundaries | Full annotation: scene type label, participant list |
| Labels | Event description text (truncated to 40 chars), participant names |
| Hover | Full causal cone + tooltip with complete detail |

**Transition INTO detail (from threads):** Scale crosses above 1.5
- Labels fade in (opacity 0.0 → 1.0 over 200ms)
- Causal link arrows fade in (opacity 0.0 → 0.5 over 200ms)
- Event node radius increases smoothly

### Pan

- Click-and-drag on empty canvas space: pans the viewport
- Middle mouse button drag: always pans (regardless of what's under cursor)
- Arrow keys: pan in 50px increments

### Performance Target

- Zoom/pan: canvas redraws at 60fps during continuous scroll
- Level transition: detail threshold swap within single frame, visual fade over 200ms
- Max canvas redraw during zoom: < 12ms

---

## 4. Region Select: Drag-to-Export

### Trigger

Hold `Shift` + click-and-drag on the canvas to define a time range selection rectangle.

### Visual Response During Drag

1. **Selection rectangle** appears: semi-transparent overlay (`rgba(255, 180, 0, 0.15)`) with dashed amber border (`#E69F00`)
2. Rectangle is constrained to full canvas height (selects a time range, not a spatial region)
3. As the user drags, event count and time range update in a floating label:
   ```
   ┌────────────────────────┐
   │ 19:30 — 20:15          │
   │ 23 events | 4 scenes   │
   └────────────────────────┘
   ```
4. Events within the time range get a subtle amber tint
5. Threads passing through the range get amber edge highlights

### Visual Response After Release

1. Selection rectangle becomes solid (dashed → solid border, slightly more opaque)
2. BeatSheetPanel in the side panel activates, showing:
   - Time range of selection
   - Events within the range, grouped by scene
   - For each event: beat_type (if assigned), description, tension
   - "Export Beat Sheet" button
3. Character threads within the selected range are ranked by tension variance (most dramatic arc first)

### Data Lookup

1. Filter `events` where `sim_time >= timeStart && sim_time <= timeEnd`
2. Filter `scenes` overlapping the time range
3. Identify which agents have events in the range → populate `regionSelection.agentIds`
4. Compute per-agent tension variance within the range for ranking

### Deselection

- Press `Escape` or click outside the selection rectangle
- Selection clears, BeatSheetPanel deactivates
- 150ms fade-out transition

### Export Action

When user clicks "Export Beat Sheet":
- Generate JSON structure:
  ```json
  {
    "time_range": [19.5, 20.25],
    "events": [ ... ],
    "scenes": [ ... ],
    "suggested_protagonist": "agent_id_with_highest_tension_variance",
    "tension_arc": [ ... ]
  }
  ```
- Download as `.json` file or copy to clipboard

### Performance Target

- Drag responsiveness: 60fps during drag (update rectangle + count only, no layout recomputation)
- Beat sheet generation on release: < 100ms (filtering + grouping)

### Fallback

- If selection contains 0 events: show "No events in this range" message, disable export
- If selection is too narrow (< 5px drag): treat as click, not region select

---

## 5. Filter Controls

### 5.1 Character Toggles

**Location:** SidePanel → CharacterFilterList

**UI:** List of character rows, each with:
- Color swatch (character's signature color)
- Character name
- Event count badge
- Toggle switch (on/off)

**Trigger:** Click toggle switch, or click character name/swatch

**Visual Response:**
- Toggle OFF: character's thread fades to `opacity 0.05` (near invisible but maintains layout space), all their event nodes hide. 150ms transition.
- Toggle ON: thread and events restore to default opacity. 150ms transition.
- When a character is toggled off during crystallization of that character: decrystallize first, then hide.

**Data Lookup:** Refilter visible events from store. No layout recomputation — Y positions stay stable, hidden characters just become transparent.

**Performance Target:** < 50ms per toggle (opacity change + redraw)

### 5.2 Tension Weight Sliders

**Location:** SidePanel → TensionSliderPanel

**UI:** 8 horizontal sliders (one per tension sub-metric), each with:
- Metric label (e.g., "Danger", "Time Pressure")
- Slider range: 0.0 to 3.0, step 0.1, default 1.0
- Current value display
- Preset buttons at top: "Default", "Thriller", "Drama", "Mystery"

**Trigger:** Drag slider handle, or click preset button

**Visual Response:**
- Tension values recompute for all events (client-side weighted sum)
- Thread thickness, event glow intensity, and tension heatmap update in real-time
- Canvas redraws continuously during slider drag
- Preset button click: all sliders animate to preset values (200ms), then canvas updates

**Data Lookup:** `recomputeAllTension(events, newWeights)` — iterates all events, applies new weights to sub-metric vectors. O(n) where n = number of events.

**Performance Target:**
- Slider drag: < 16ms per frame (recompute tension for 200 events: ~0.1ms, redraw: ~10ms)
- Preset switch: sliders animate over 200ms, canvas updates each frame

### 5.3 Event Type Filter

**Location:** ToolbarPanel (dropdown or chip selector)

**UI:** Multi-select chips for each EventType. All active by default.

**Trigger:** Click chip to toggle event type visibility

**Visual Response:**
- Filtered-out event types: nodes disappear (150ms fade), thread segments through those events thin
- Remaining events maintain their positions — no layout shift

**Performance Target:** < 50ms per filter change

### 5.4 Minimum Tension Threshold

**Location:** ToolbarPanel (slider)

**UI:** Single horizontal slider, range 0.0 to 1.0

**Trigger:** Drag slider

**Visual Response:**
- Events with tension below threshold fade to `opacity 0.1`
- Useful for "show me only the dramatic moments" — quickly strips texture/noise events

**Performance Target:** < 16ms per frame during drag

---

## 6. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Switch to Cloud zoom |
| `2` | Switch to Threads zoom |
| `3` | Switch to Detail zoom |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| Arrow keys | Pan viewport |
| `Escape` | Decrystallize / deselect region / close panels |
| `Tab` | Cycle through characters (crystallize next) |
| `Shift+Tab` | Cycle through characters (crystallize previous) |
| `Space` | Toggle sidebar visibility |
| `F` | Fit all events in viewport (reset zoom/pan) |

---

## 7. Interaction State Machine

The renderer has a finite set of interaction states. This prevents conflicting interactions.

```
                    ┌──────────┐
                    │   IDLE   │
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
    ┌───────────┐  ┌──────────┐  ┌──────────────┐
    │  HOVERING │  │SELECTING │  │  PANNING     │
    │           │  │ (region) │  │              │
    └─────┬─────┘  └────┬─────┘  └──────────────┘
          │              │
          ▼              ▼
    ┌──────────┐   ┌───────────┐
    │CRYSTALLI-│   │ REGION    │
    │  ZED     │   │ SELECTED  │
    └──────────┘   └───────────┘
```

**State transitions:**

| From | Trigger | To |
|------|---------|-----|
| IDLE | Mouse enters event | HOVERING |
| IDLE | Shift+drag start | SELECTING |
| IDLE | Drag on empty space | PANNING |
| HOVERING | Mouse leaves event | IDLE |
| HOVERING | Click on event | CRYSTALLIZED |
| HOVERING | Shift+drag start | SELECTING |
| CRYSTALLIZED | Click on empty space | IDLE |
| CRYSTALLIZED | Click on different character | CRYSTALLIZED (new character) |
| CRYSTALLIZED | Mouse enters event | CRYSTALLIZED + HOVERING (overlay) |
| CRYSTALLIZED | Escape | IDLE |
| SELECTING | Mouse up | REGION_SELECTED |
| REGION_SELECTED | Escape | IDLE |
| REGION_SELECTED | Click outside region | IDLE |
| REGION_SELECTED | Shift+drag new region | SELECTING |
| PANNING | Mouse up | IDLE |

Note: CRYSTALLIZED and HOVERING can coexist — hovering within a crystallized view shows the causal cone relative to the crystallized arc.

---

## 8. Tooltip Content by Zoom Level

| Zoom Level | Tooltip Content |
|------------|----------------|
| **Cloud** | "23 events in this region, avg tension 0.65, dominant: conflict" (regional summary) |
| **Threads** | Event type icon, description (40 chars), time, location, tension bar |
| **Detail** | Full: description, time, location, all participants, tension + irony + significance, beat type, first 3 deltas with reason_display |

---

## 9. NOT in Scope

- **Drag-to-resimulate:** Requires simulation backend. Phase 2+.
- **Right-click context menu:** No context menu in Phase 1. Future: "Simulate alternatives", "Find similar events", "View counterfactual impact".
- **Touch/mobile support:** Phase 1 is desktop-only. Touch gestures (pinch zoom, two-finger pan) would be Phase 3+.
- **Multi-select events:** Only single event selection + region selection. Multi-select with Ctrl+click is Phase 3+.
- **Timeline scrubbing:** No playback/animation of the simulation over time. All events are shown simultaneously. Playback is a Phase 3+ feature.
- **Annotation/note-taking:** Users cannot add notes to events in Phase 1.

---

## 10. Edge Cases

| Scenario | Handling |
|----------|----------|
| Two events at identical (x, y) position | Hit canvas returns the one with higher significance. In detail zoom, offset nodes by 3px vertically with a visual cluster indicator. |
| Hover on event with 0 causal links | Show event with full opacity, no cone highlight (no connected events to show). Tooltip displays normally. |
| Crystallize character with only 1 event | Thread shows as a single point with a short stub line. Side panel shows the single event. |
| All 6 characters at same location (dinner table) | Threads converge but maintain distinct Y positions within a narrow band. Thread colors must remain distinguishable. See thread-layout.md for convergence behavior. |
| 0 events visible after filtering | Show "No events match current filters" message centered on canvas. Disable hover/click interactions. |
| Region select captures a time range with 200+ events | BeatSheetPanel shows summary with "Showing top 20 by significance. Full list: [expand]" |
| Rapid hover switching (mouse moves quickly across many events) | Debounce tooltip rendering at 50ms. Causal cone highlight updates every frame (no debounce — it must be instant). |
