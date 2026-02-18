# UX Review: NarrativeField Visualization

**Date:** 2026-02-09
**Reviewer:** UX Review Agent
**Build:** codex/portfolio-packaging branch
**Dataset:** dinner_party (70 events, 6 agents, 6 scenes)

---

## Summary

Both Threads and Topology views render correctly with fake data. Core interactions (hover tooltips, click selection, arc crystallization, character/event filtering, tension presets, zoom controls, view mode switching) all function as expected. State is preserved across view mode toggles. Several visual polish improvements were applied during this review.

---

## Findings

### P0 (Critical) -- None

No broken functionality preventing usage.

### P1 (High) -- None

No significant usability issues found. All core interactions work correctly.

### P2 (Medium)

#### P2-1: Scene category labels overlapped in Topology View
- **View:** Topology
- **Description:** At the top of the topology canvas, scene category labels (e.g., "CATASTROPHE", "CONFRONTATION") could overlap when scenes are close together in time.
- **Status:** FIXED. Added overlap detection logic that skips labels too close to the previous one, and reduced max label length from 22 to 18 characters.
- **File:** `src/canvas/layers/TensionTerrainLayer.ts:279-290`

#### P2-2: Event annotation labels had low contrast against terrain
- **View:** Topology
- **Description:** Peak annotation labels (italic text showing event descriptions near tension peaks) used `rgba(255,255,255,0.60)` fill with `shadowBlur: 6`. Over bright pink/red terrain blobs, these could be hard to read.
- **Status:** FIXED. Increased label opacity to 0.78, shadow color to `rgba(0,0,0,0.85)`, and shadow blur to 8.
- **File:** `src/canvas/layers/AnnotationLayer.ts:98-102`

#### P2-3: Topology event markers too small for easy clicking
- **View:** Topology
- **Description:** Diamond markers used base radius 4.25px. On a dense topology with many overlapping terrain blobs, small markers were difficult to target precisely.
- **Status:** FIXED. Increased base radius from 4.25 to 5.0, turning points from 6.5 to 7.0, catastrophe events from 7.25 to 8.0. Also increased marker outline stroke from `rgba(255,255,255,0.22)` / 1px to `rgba(255,255,255,0.35)` / 1.25px for better visibility against terrain.
- **File:** `src/canvas/layers/TopologyEventLayer.ts:70-73, 105-106`

#### P2-4: Tooltip shadow too subtle in Topology View
- **View:** Topology
- **Description:** The white tooltip popup had a light shadow (`rgba(0,0,0,0.14)`) that didn't stand out strongly enough against the very dark topology background.
- **Status:** FIXED. Added view-mode-aware tooltip styling: topology mode uses stronger shadow (`rgba(0,0,0,0.45)`) and a subtle white border.
- **File:** `src/components/CanvasRenderer.tsx:580-586`

### P3 (Low)

#### P3-1: Zoom buttons lack visual active state
- **View:** Both
- **Description:** The Cloud/Threads/Detail zoom buttons used `aria-pressed` for accessibility but had no visible styling difference for the active state.
- **Status:** FIXED. Added bold font weight and subtle background highlight for the active zoom button.
- **File:** `src/components/ControlPanel.tsx:187-213`

#### P3-2: Slider styling uses browser defaults
- **View:** Both
- **Description:** Range input sliders use the browser's default styling. They are functional but look different across browsers and lack visual consistency with the rest of the UI.
- **Status:** Recommendation only. Consider adding custom CSS for slider track/thumb styling in a future pass, using `appearance: none` with custom `::-webkit-slider-*` and `::-moz-range-*` pseudo-elements.

#### P3-3: Missing favicon
- **View:** Both
- **Description:** Browser console shows a 404 for `/favicon.ico`. Purely cosmetic.
- **Status:** Recommendation only. Add a simple favicon.

---

## Functional Testing Results

| Feature | Threads | Topology | Status |
|---------|---------|----------|--------|
| Renders with fake data | Yes | Yes | Pass |
| Tooltip on hover | Yes | Yes | Pass |
| Click event selection | Yes | Yes | Pass |
| Detail panel shows metrics | Yes | Yes | Pass |
| Arc crystallization | Yes | Yes | Pass |
| Character toggle (checkbox) | Yes | Yes | Pass |
| Genre preset (Thriller/Drama/Mystery) | Yes | Yes | Pass |
| Tension slider adjustment | Yes | Yes | Pass |
| Reset button | Yes | Yes | Pass |
| Zoom levels (Cloud/Threads/Detail) | Yes | Yes | Pass |
| Scale slider | Yes | Yes | Pass |
| View mode toggle preserves state | -- | -- | Pass |
| Scene labels (bottom bar) | Yes | N/A | Pass |
| Scene labels (top, category) | N/A | Yes | Pass |
| Tension terrain heatmap | N/A | Yes | Pass |
| Diamond/star markers | N/A | Yes | Pass |
| Character arc curves | N/A | Yes | Pass |
| Pan (drag) | Yes | Yes | Pass |
| Keyboard shortcuts (Escape, 1/2/3, Tab) | Yes | Yes | Pass |

---

## Files Modified

1. `src/canvas/layers/AnnotationLayer.ts` -- Increased peak label contrast
2. `src/canvas/layers/TopologyEventLayer.ts` -- Larger markers, stronger outlines
3. `src/canvas/layers/TensionTerrainLayer.ts` -- Scene label overlap prevention
4. `src/components/CanvasRenderer.tsx` -- View-mode-aware tooltip shadow
5. `src/components/ControlPanel.tsx` -- Active zoom button styling
6. `vite.config.ts` -- Added `host.docker.internal` to allowedHosts for testing
