# Thread Layout Specification

**Status:** FINAL (aligned with canonical events.md, agents.md, tension-pipeline.md)
**Author:** viz-architect
**Dependencies:** specs/schema/events.md (#1), specs/visualization/renderer-architecture.md (#9)
**Consumers:** Phase 1 coding agent, specs/visualization/fake-data-visual-spec.md (#12)
**Cross-references:** doc3.md Decisions 1, 12, 16; doc2.md "Pinned Manifold"

---

## 1. Problem Statement

Character threads need Y-axis positions that communicate narrative structure:
- Characters in the same scene should be **close together** (convergence = interaction)
- Characters in different locations should be **far apart** (divergence = separation)
- Threads should **not jump erratically** (smooth motion = readability)
- Threads should **never fully overlap** (distinct lanes = distinguishability)

The X-axis is pinned to `sim_time` (Decision 12). The Y-axis is computed by a spring-force simulation that balances attraction (shared scenes), repulsion (different locations), and inertia (smoothness).

---

## 2. Conceptual Model

At each time step, each character has a Y position. The layout engine computes Y positions for all characters at all event timestamps, then interpolates smooth spline paths between them.

Think of each character as a horizontal bead on a vertical rail at each time step. Springs pull beads together when characters interact, push them apart when they separate, and resist sudden movement.

---

## 3. Input

```typescript
interface ThreadLayoutInput {
  events: Event[];              // all events, sorted by sim_time
  agents: string[];             // all agent IDs (6 for dinner party)
  scenes: Scene[];              // scene groupings
  canvasHeight: number;         // available vertical pixels
  parameters: LayoutParameters; // tunable force parameters
}

interface LayoutParameters {
  // Force strengths
  attractionStrength: number;    // pull between co-located agents (default: 0.3)
  repulsionStrength: number;     // push between separated agents (default: 0.2)
  interactionBonus: number;      // extra pull for direct interaction (default: 0.5)
  laneSpringStrength: number;    // pull toward assigned base lane (default: 0.1)
  inertia: number;               // resistance to Y-position change (default: 0.7)

  // Layout constraints
  minSeparation: number;         // minimum Y-distance between any two agents (pixels, default: 20)
  lanePadding: number;           // top/bottom margin (pixels, default: 40)

  // Simulation
  iterations: number;            // force iterations per time step (default: 50)
  convergenceThreshold: number;  // stop early if max delta < this (default: 0.5)
  timeResolution: number;        // compute Y positions at this time interval (default: 0.5 sim-minutes)
}
```

---

## 4. Output

```typescript
interface ThreadLayoutOutput {
  /** Y position of each agent at each time sample */
  positions: Map<string, number[]>;  // agent_id -> Y values at each time sample

  /** Time samples corresponding to position arrays */
  timeSamples: number[];             // ascending sim_time values

  /** Spline control points for smooth rendering */
  threadPaths: ThreadPath[];         // per-agent spline paths
}

interface ThreadPath {
  agentId: string;
  agentName: string;
  color: string;
  /** Control points for cubic Bezier spline: [x, y][] */
  controlPoints: [number, number][];
  /** Per-segment thickness (between consecutive control points) */
  thickness: number[];
}
```

---

## 5. Algorithm

### Phase 1: Initialize Base Lanes

Assign each agent a "home" Y position evenly distributed across the canvas height. This is their default position when no forces apply. Lane assignment determines the visual ordering of characters.

```typescript
function initializeLanes(agents: string[], canvasHeight: number, padding: number): Map<string, number> {
  const lanes = new Map<string, number>();
  const usableHeight = canvasHeight - 2 * padding;
  const spacing = usableHeight / (agents.length - 1 || 1);

  for (let i = 0; i < agents.length; i++) {
    lanes.set(agents[i], padding + i * spacing);
  }
  return lanes;
}
```

**Lane ordering heuristic:** Order agents by their first interaction. Characters who interact earliest are placed in adjacent lanes. This minimizes initial thread crossings. Specifically: build an interaction graph from events, compute a DFS ordering that keeps frequently-interacting pairs adjacent.

For the dinner party MVP with 6 agents, manual ordering in the character config may be preferred.

### Phase 2: Compute Time Samples

Sample time at regular intervals across the full sim_time range.

```typescript
function computeTimeSamples(events: Event[], resolution: number): number[] {
  const minTime = events[0].sim_time;
  const maxTime = events[events.length - 1].sim_time;
  const samples: number[] = [];
  for (let t = minTime; t <= maxTime; t += resolution) {
    samples.push(t);
  }
  return samples;
}
```

### Phase 3: For Each Time Sample, Compute Forces

At each time sample `t`, determine which agents are in which locations, who is interacting, and apply forces:

```typescript
function computePositionsAtTime(
  t: number,
  agents: string[],
  prevPositions: Map<string, number>,  // Y positions at previous time sample
  baseLanes: Map<string, number>,
  locationAtTime: (agentId: string, t: number) => string,
  interactingAtTime: (agentA: string, agentB: string, t: number) => boolean,
  params: LayoutParameters,
  canvasHeight: number
): Map<string, number> {

  // Start from previous positions (inertia)
  const positions = new Map<string, number>();
  for (const agent of agents) {
    positions.set(agent, prevPositions.get(agent) ?? baseLanes.get(agent)!);
  }

  // Iterative force relaxation
  for (let iter = 0; iter < params.iterations; iter++) {
    const forces = new Map<string, number>();
    for (const agent of agents) forces.set(agent, 0);

    for (let i = 0; i < agents.length; i++) {
      for (let j = i + 1; j < agents.length; j++) {
        const a = agents[i];
        const b = agents[j];
        const ya = positions.get(a)!;
        const yb = positions.get(b)!;
        const dy = yb - ya;
        const dist = Math.abs(dy);
        const direction = dy > 0 ? 1 : -1;  // positive = b is below a

        const sameLocation = locationAtTime(a, t) === locationAtTime(b, t);
        const interacting = interactingAtTime(a, b, t);

        if (sameLocation) {
          // ATTRACTION: pull together
          let strength = params.attractionStrength;
          if (interacting) strength += params.interactionBonus;

          // Attraction force proportional to distance
          const force = strength * dist * direction;
          forces.set(a, forces.get(a)! + force);
          forces.set(b, forces.get(b)! - force);
        } else {
          // REPULSION: push apart
          // Repulsion force inversely proportional to distance (capped)
          const cappedDist = Math.max(dist, params.minSeparation);
          const force = -params.repulsionStrength * (params.minSeparation / cappedDist) * direction;
          forces.set(a, forces.get(a)! + force);
          forces.set(b, forces.get(b)! - force);
        }

        // MINIMUM SEPARATION enforcement (hard constraint)
        if (dist < params.minSeparation) {
          const push = (params.minSeparation - dist) / 2;
          forces.set(a, forces.get(a)! - push * direction);
          forces.set(b, forces.get(b)! + push * direction);
        }
      }
    }

    // LANE SPRING: pull toward base lane (prevents total convergence)
    for (const agent of agents) {
      const y = positions.get(agent)!;
      const baseLane = baseLanes.get(agent)!;
      const laneForce = params.laneSpringStrength * (baseLane - y);
      forces.set(agent, forces.get(agent)! + laneForce);
    }

    // INERTIA: blend toward previous position
    for (const agent of agents) {
      const y = positions.get(agent)!;
      const prev = prevPositions.get(agent) ?? baseLanes.get(agent)!;
      const inertiaForce = params.inertia * (prev - y);
      forces.set(agent, forces.get(agent)! + inertiaForce);
    }

    // Apply forces
    let maxDelta = 0;
    for (const agent of agents) {
      const y = positions.get(agent)!;
      const force = forces.get(agent)!;
      const newY = Math.max(
        params.lanePadding,
        Math.min(canvasHeight - params.lanePadding, y + force * 0.1)  // damping factor
      );
      maxDelta = Math.max(maxDelta, Math.abs(newY - y));
      positions.set(agent, newY);
    }

    // Early termination if converged
    if (maxDelta < params.convergenceThreshold) break;
  }

  return positions;
}
```

### Phase 4: Determine Location and Interaction State

Helper functions that read from the event log to determine who is where and who is interacting at a given time.

```typescript
function buildLocationLookup(events: Event[]): (agentId: string, t: number) => string {
  // For each agent, build a sorted list of (time, location) from SOCIAL_MOVE events
  // and from initial positions. At time t, binary-search for the most recent location.
  // ...
}

function buildInteractionLookup(events: Event[]): (a: string, b: string, t: number) => boolean {
  // Two agents are "interacting" at time t if there exists an event within
  // [t - window, t + window] where both are participants.
  // Window size: 1 sim-minute (2 time samples at default resolution).
  // ...
}
```

### Phase 5: Generate Spline Control Points

Convert the per-time-sample positions into smooth cubic Bezier splines for rendering.

```typescript
function generateSplines(
  positions: Map<string, number[]>,
  timeSamples: number[],
  xMapper: (simTime: number) => number,  // sim_time → canvas X
  tensionAtTime: (agentId: string, t: number) => number
): ThreadPath[] {
  const paths: ThreadPath[] = [];

  for (const [agentId, yValues] of positions) {
    const controlPoints: [number, number][] = [];
    const thickness: number[] = [];

    for (let i = 0; i < timeSamples.length; i++) {
      const x = xMapper(timeSamples[i]);
      const y = yValues[i];
      controlPoints.push([x, y]);

      if (i < timeSamples.length - 1) {
        // Thickness between this point and next, driven by tension
        const avgTension = (tensionAtTime(agentId, timeSamples[i]) +
                           tensionAtTime(agentId, timeSamples[i + 1])) / 2;
        thickness.push(2 + avgTension * 4);  // range: 2px to 6px
      }
    }

    paths.push({
      agentId,
      agentName: agentId,  // resolved from agent metadata
      color: CHARACTER_COLORS[agents.indexOf(agentId) % CHARACTER_COLORS.length],
      controlPoints,
      thickness,
    });
  }

  return paths;
}
```

The canvas renderer draws these as smooth cubic Bezier curves using `ctx.bezierCurveTo()`, with monotone x-interpolation to prevent horizontal looping.

---

## 6. Worked Examples

### Example 1: Two Characters Meet at the Balcony

**Setup:**
- 6 characters, canvas height = 600px, padding = 40px
- Base lanes: Agent A = 152px, Agent B = 256px, Agent C = 360px, Agent D = 464px, Agent E = 568px (skip F for brevity — similar behavior)
- At time t=10: A and B are both at "balcony", C/D/E are at "dining_table"

**Forces on A and B at t=10:**
- Attraction (same location): `0.3 * |256 - 152| * 1 = 0.3 * 104 = 31.2` pulling A down, B up
- Interaction bonus (they share an event at t=10): `0.5 * 104 = 52.0` additional pull
- Lane spring on A: `0.1 * (152 - current_y)` pulling A back toward 152
- Lane spring on B: `0.1 * (256 - current_y)` pulling B back toward 256
- Repulsion from C/D/E (different location): pushes A and B away from the C/D/E cluster

**Result after convergence:**
- A moves from 152 → ~185 (down toward B)
- B moves from 256 → ~223 (up toward A)
- They converge to ~40px apart (limited by minSeparation=20 and lane spring)
- C/D/E cluster tighter at their end of the canvas

**Visual effect:** The A and B threads visibly pull together at t=10, signaling their meeting. The C/D/E threads remain in a loose cluster at the dining table.

### Example 2: Character Leaves the Dinner Table

**Setup:**
- At time t=5: All 6 characters at "dining_table" (convergence case)
- At time t=8: Character D moves to "kitchen"
- At time t=12: D returns to "dining_table"

**At t=5 (all together):**
- All 6 have strong attraction forces pulling them toward each other
- Lane springs prevent full collapse — they cluster within a ~200px band centered on the canvas
- Minimum separation ensures 20px between each pair
- Positions: A=205, B=230, C=255, D=280, E=305, F=330 (roughly)

**At t=8 (D in kitchen):**
- D has repulsion against A/B/C/E/F (all at dining_table, D at kitchen)
- D is pushed away from the cluster
- D's lane spring (original lane 360) also pulls D toward the bottom
- D moves from ~280 → ~430 (pushed below the main cluster)
- Remaining 5 at the dining table re-cluster slightly tighter (one fewer member)

**At t=12 (D returns):**
- D regains attraction to the dining table group
- Inertia slows D's return (doesn't snap back instantly)
- Over 2-3 time samples, D eases back from ~430 → ~310 (near original position)
- The "dip and return" of D's thread visually shows the departure

**Visual effect:** D's thread dips downward (away from the group) from t=8 to t=12, clearly showing the kitchen detour.

### Example 3: The All-Together Edge Case

**Setup:**
- All 6 characters at "dining_table" for the entire evening
- No one ever leaves

**Problem:** Without intervention, strong attraction + no repulsion → all threads converge to a single line. Unreadable.

**Solution — Lane Spring as Primary Force:**
- When all agents are co-located, attraction pulls them together
- Lane springs pull them toward their base lanes
- The equilibrium is determined by the ratio of `attractionStrength` to `laneSpringStrength`
- With defaults (0.3 vs 0.1), threads compress to ~60% of their full spread

**Positions:**
- Full spread (no attraction): A=152, B=256, C=360, D=464, E=568 (F omitted). Range = 416px.
- Equilibrium with all-together attraction: threads compress toward center. Each thread moves ~40% toward the center of mass (360px).
  - A: 152 + 0.4*(360-152) = 235
  - B: 256 + 0.4*(360-256) = 298
  - C: 360 + 0.4*(360-360) = 360
  - D: 464 + 0.4*(360-464) = 422
  - E: 568 + 0.4*(360-568) = 485
- Range compressed to ~250px (60% of original)
- Still visually distinct — minimum 20px separation maintained

**When interactions happen within the all-together scene:**
- Two characters having a direct conversation get the interaction bonus (0.5)
- Their threads pull closer together within the cluster
- This creates visible "pairings" within the dinner table grouping

**Visual effect:** Threads run as a moderately-tight bundle with visible internal structure showing who talks to whom.

---

## 7. Parameter Tuning Guide

| Parameter | Low Value Effect | High Value Effect | Suggested Default |
|-----------|-----------------|-------------------|-------------------|
| `attractionStrength` | Characters barely converge when co-located | Characters collapse into single line | 0.3 |
| `repulsionStrength` | Separated characters don't spread much | Characters in different rooms fly to canvas edges | 0.2 |
| `interactionBonus` | Direct conversations don't visually stand out | Conversations create dramatic convergence | 0.5 |
| `laneSpringStrength` | Threads wander freely (beautiful but chaotic) | Threads stay rigid in lanes (subway map) | 0.1 |
| `inertia` | Threads can jump between time samples | Threads resist all movement (too smooth) | 0.7 |
| `minSeparation` | Threads can overlap | Threads maintain large gaps even when co-located | 20px |

**Recommendation:** Expose these as debug sliders during development. Hardcode the tuned values for production.

---

## 8. Performance Analysis

For the dinner party MVP (6 agents, ~100 time samples at 0.5 sim-minute resolution):

| Step | Computation | Time |
|------|------------|------|
| Initialize lanes | 6 assignments | < 0.1ms |
| Per time sample: force computation | 15 pairs * 50 iterations = 750 force calculations | ~0.5ms |
| Total across all samples | 100 * 0.5ms | ~50ms |
| Spline generation | 6 agents * 100 control points | ~2ms |
| **Total layout** | | **~55ms** |

Well within the 500ms budget. Even at 200 events with 200 time samples, total would be ~110ms.

---

## 9. Smoothing: Monotone Cubic Interpolation

To draw smooth threads without horizontal looping (where X-values go backwards), use monotone cubic Hermite interpolation (Fritsch-Carlson method). This guarantees:
- The curve passes through all control points
- X values are strictly monotonically increasing
- No visual "loops" or "S-curves" between adjacent time samples

D3 provides this via `d3.curveMonotoneX`.

---

## 10. Edge Cases

| Scenario | Handling |
|----------|----------|
| Agent appears mid-simulation (joins the party) | Thread starts at the time of first event, from their base lane position. Fade-in over 2 time samples. |
| Agent leaves permanently (storms out) | Thread ends at last event with a visual terminus (tapered end, small "x" marker). |
| Two agents swap locations simultaneously | Both threads transition smoothly in opposite directions. No special handling needed — forces naturally resolve. |
| Agent has no events in a time range | Thread continues at last known position with no forces applied (inertia holds). Thread becomes slightly transparent (dashed line) to indicate absence from narrative action. |
| Canvas resize | Recompute all positions (scale base lanes to new height). Animate transition over 200ms. |

---

## 11. NOT in Scope

- **3D displacement (Z-axis tension):** The thread layout is 2D only. Tension is encoded in thread thickness and glow, not vertical displacement. 3D mode is a future feature.
- **Thread crossing minimization:** No explicit crossing reduction algorithm. The force simulation naturally reduces crossings through inertia and lane springs. If crossings become problematic, add a gentle crossing penalty force in Phase 2+.
- **Animated simulation playback:** The layout engine runs once on the full event log. Animated "growing" of threads over time is a Phase 3+ feature.
- **Agent grouping by faction:** For MVP, each agent is independent. Faction-based grouping (agents in the same faction share a super-lane) is a Phase 7 (MVP 1.5 "The Bridge") feature.
