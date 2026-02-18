# NarrativeField Visualization

## Tech Stack
- React 18+ with TypeScript (strict mode)
- Zustand for state management (~1KB, no context provider needed)
- HTML5 Canvas API for rendering (NOT SVG — see renderer-architecture.md Section 1)
- D3.js (d3-scale, d3-force, d3-shape only — tree-shaken, ~30KB)
- Vite for bundling

## Authoritative Specs
- `specs/visualization/renderer-architecture.md` — store, components, canvas layers, hit detection
- `specs/visualization/interaction-model.md` — hover, click, zoom, region select, state machine
- `specs/visualization/thread-layout.md` — spring-force Y-axis algorithm
- `specs/visualization/fake-data-visual-spec.md` — 70-event target visual
- `specs/integration/data-flow.md` Section 3.3 — NarrativeFieldPayload shape

## Key Constraints
- Canvas, not SVG (200 events with hover redraws → SVG DOM is the bottleneck)
- Hit detection via offscreen color-coded canvas (unique RGB per event → O(1) mouse lookup)
- Precomputed BFS-3 causal neighborhoods for <16ms hover
- 4 canvas layers: Background (scene bands) → Thread (splines) → EventNode (dots) → Highlight (hover/selection)
- Client-side tension recomputation: frontend receives 8 sub-metric vectors, applies TensionWeights via sliders
- Performance: initial render <100ms, hover <16ms, click crystallization <50ms, layout <500ms

## Color Palette (Wong palette, colorblind-safe)
- James Thorne: #E69F00 (warm orange)
- Elena Thorne: #56B4E9 (sky blue)
- Marcus Webb: #009E73 (teal green)
- Lydia Cross: #F0E442 (yellow)
- Diana Forrest: #0072B2 (deep blue)
- Victor Hale: #D55E00 (vermillion)

## Event Type Colors
Define in constants/colors.ts mapping EventType → color. Use muted palette for nodes, saturated for highlights.

## Workflow

- Update /CHANGELOG.md when making visualization changes
