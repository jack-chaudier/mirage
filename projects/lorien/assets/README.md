# Visualization Assets

This directory contains portfolio-facing visualization assets.

Current artifact:

- `checkpoint-overview-2026-02.svg` — generated checkpoint graphic (determinism + catastrophe distribution + location-share summary), sourced from `data/checkpoints/2026-02-checkpoint-science.json`.

Planned screenshot captures:

- `threads-view-seed42.png` — Threads View with the seed 42 `.nf-viz` payload loaded.
- `topology-view-seed42.png` — Topology View with the same seed 42 payload.

Capture guidance:

1. Start the visualization app (`cd src/visualization && npm run dev`).
2. Load a seed 42 payload (`.nf-viz.json`) via the Control Panel file picker.
3. Capture one screenshot in Threads View and one in Topology View at desktop resolution.
4. Keep images under ~2 MB each for repository friendliness.
