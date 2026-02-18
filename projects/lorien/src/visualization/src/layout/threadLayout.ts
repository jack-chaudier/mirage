import type { AgentManifest, Event, Scene, ThreadPath } from '../types';
import { DeltaKind } from '../types';

export interface LayoutParameters {
  attractionStrength: number;
  repulsionStrength: number;
  interactionBonus: number;
  laneSpringStrength: number;
  inertia: number;

  minSeparation: number;
  lanePadding: number;

  iterations: number;
  convergenceThreshold: number;
  timeResolution: number;
}

export interface ThreadLayoutInput {
  events: Event[];
  agents: Array<string | Pick<AgentManifest, 'id' | 'initial_location'>>;
  scenes: Scene[];
  canvasHeight: number;
  parameters?: Partial<LayoutParameters>;
}

export interface ThreadLayoutOutput {
  positions: Map<string, number[]>;
  timeSamples: number[];
  threadPaths: ThreadPath[];
}

function normalizeAgents(
  agents: Array<string | Pick<AgentManifest, 'id' | 'initial_location'>>
): { ids: string[]; initialLocations: Map<string, string> } {
  const ids: string[] = [];
  const initialLocations = new Map<string, string>();

  for (const entry of agents) {
    if (typeof entry === 'string') {
      ids.push(entry);
      if (!initialLocations.has(entry)) initialLocations.set(entry, 'dining_table');
      continue;
    }
    ids.push(entry.id);
    initialLocations.set(entry.id, entry.initial_location || 'dining_table');
  }

  return { ids, initialLocations };
}

export const DEFAULT_LAYOUT_PARAMETERS: LayoutParameters = {
  attractionStrength: 0.3,
  repulsionStrength: 0.2,
  interactionBonus: 0.5,
  laneSpringStrength: 0.1,
  inertia: 0.7,

  minSeparation: 20,
  lanePadding: 40,

  iterations: 50,
  convergenceThreshold: 0.5,
  timeResolution: 0.5
};

export function initializeLanes(
  agents: string[],
  canvasHeight: number,
  padding: number
): Map<string, number> {
  const lanes = new Map<string, number>();
  const usableHeight = Math.max(1, canvasHeight - 2 * padding);
  const spacing = usableHeight / (agents.length - 1 || 1);

  for (let i = 0; i < agents.length; i += 1) {
    lanes.set(agents[i]!, padding + i * spacing);
  }
  return lanes;
}

export function computeTimeSamples(events: Event[], resolution: number): number[] {
  if (events.length === 0) return [];
  const minTime = events[0]!.sim_time;
  const maxTime = events[events.length - 1]!.sim_time;
  const samples: number[] = [];
  for (let t = minTime; t <= maxTime + 1e-9; t += resolution) {
    samples.push(Number(t.toFixed(6)));
  }
  return samples;
}

function buildLocationLookup(
  agents: string[],
  events: Event[],
  initialLocations: Map<string, string>
): (agentId: string, t: number) => string {
  type Entry = { time: number; location: string };
  const timelines = new Map<string, Entry[]>();

  for (const agentId of agents) {
    timelines.set(agentId, [
      { time: Number.NEGATIVE_INFINITY, location: initialLocations.get(agentId) ?? 'dining_table' }
    ]);
  }

  for (const e of events) {
    for (const d of e.deltas) {
      if (d.kind !== DeltaKind.AGENT_LOCATION) continue;
      const agentId = d.agent;
      if (!timelines.has(agentId)) timelines.set(agentId, []);
      timelines.get(agentId)!.push({ time: e.sim_time, location: String(d.value) });
    }
  }

  // Ensure sorted timelines for binary search.
  for (const [agentId, entries] of timelines) {
    entries.sort((a, b) => a.time - b.time);
    timelines.set(agentId, entries);
  }

  function locationAtTime(agentId: string, t: number): string {
    const entries = timelines.get(agentId);
    if (!entries || entries.length === 0) return 'dining_table';

    // Binary search for rightmost entry with entry.time <= t.
    let lo = 0;
    let hi = entries.length - 1;
    let best = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const entry = entries[mid]!;
      if (entry.time <= t) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return entries[best]!.location;
  }

  return locationAtTime;
}

function buildInteractionLookup(
  events: Event[],
  windowMinutes = 1.0
): (a: string, b: string, t: number) => boolean {
  // Build (a|b) -> sorted list of interaction times.
  const pairTimes = new Map<string, number[]>();

  for (const e of events) {
    const participants = new Set([e.source_agent, ...e.target_agents]);
    const list = Array.from(participants);
    for (let i = 0; i < list.length; i += 1) {
      for (let j = i + 1; j < list.length; j += 1) {
        const a = list[i]!;
        const b = list[j]!;
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        const arr = pairTimes.get(key);
        if (arr) arr.push(e.sim_time);
        else pairTimes.set(key, [e.sim_time]);
      }
    }
  }

  for (const [k, arr] of pairTimes) {
    arr.sort((x, y) => x - y);
    pairTimes.set(k, arr);
  }

  function hasInteractionNear(key: string, t: number): boolean {
    const arr = pairTimes.get(key);
    if (!arr || arr.length === 0) return false;

    // Binary search for insertion index.
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid]! < t) lo = mid + 1;
      else hi = mid;
    }
    const candidates = [arr[lo - 1], arr[lo]];
    return candidates.some((x) => x !== undefined && Math.abs(x - t) <= windowMinutes);
  }

  return (a: string, b: string, t: number) => {
    if (a === b) return false;
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    return hasInteractionNear(key, t);
  };
}

function computePositionsAtTime(args: {
  t: number;
  agents: string[];
  prevPositions: Map<string, number>;
  baseLanes: Map<string, number>;
  locationAtTime: (agentId: string, t: number) => string;
  interactingAtTime: (a: string, b: string, t: number) => boolean;
  params: LayoutParameters;
  canvasHeight: number;
}): Map<string, number> {
  const {
    t,
    agents,
    prevPositions,
    baseLanes,
    locationAtTime,
    interactingAtTime,
    params,
    canvasHeight
  } = args;

  const positions = new Map<string, number>();
  for (const agent of agents) {
    positions.set(agent, prevPositions.get(agent) ?? baseLanes.get(agent)!);
  }

  const clampY = (y: number) =>
    Math.max(params.lanePadding, Math.min(canvasHeight - params.lanePadding, y));

  // Iterative force relaxation.
  for (let iter = 0; iter < params.iterations; iter += 1) {
    const forces = new Map<string, number>();
    for (const agent of agents) forces.set(agent, 0);

    for (let i = 0; i < agents.length; i += 1) {
      for (let j = i + 1; j < agents.length; j += 1) {
        const a = agents[i]!;
        const b = agents[j]!;
        const ya = positions.get(a)!;
        const yb = positions.get(b)!;
        const dy = yb - ya;
        const dist = Math.abs(dy);
        const direction = dy > 0 ? 1 : -1;

        const sameLocation = locationAtTime(a, t) === locationAtTime(b, t);
        const interacting = interactingAtTime(a, b, t);

        if (sameLocation) {
          let strength = params.attractionStrength;
          if (interacting) strength += params.interactionBonus;
          const f = strength * dist * direction;
          forces.set(a, forces.get(a)! + f);
          forces.set(b, forces.get(b)! - f);
        } else {
          const cappedDist = Math.max(dist, params.minSeparation);
          const f = -params.repulsionStrength * (params.minSeparation / cappedDist) * direction;
          forces.set(a, forces.get(a)! + f);
          forces.set(b, forces.get(b)! - f);
        }

        if (dist < params.minSeparation) {
          const push = (params.minSeparation - dist) / 2;
          forces.set(a, forces.get(a)! - push * direction);
          forces.set(b, forces.get(b)! + push * direction);
        }
      }
    }

    // Lane spring + inertia.
    for (const agent of agents) {
      const y = positions.get(agent)!;
      const base = baseLanes.get(agent)!;
      forces.set(agent, forces.get(agent)! + params.laneSpringStrength * (base - y));

      const prev = prevPositions.get(agent) ?? base;
      forces.set(agent, forces.get(agent)! + params.inertia * (prev - y));
    }

    // Apply.
    let maxDelta = 0;
    for (const agent of agents) {
      const y = positions.get(agent)!;
      const f = forces.get(agent)!;
      const next = clampY(y + f * 0.1); // damping
      maxDelta = Math.max(maxDelta, Math.abs(next - y));
      positions.set(agent, next);
    }

    if (maxDelta < params.convergenceThreshold) break;
  }

  return positions;
}

function tensionAtTimeFromEvents(events: Event[], agentId: string, t: number): number {
  const window = 0.75; // minutes
  let maxT = 0;
  for (const e of events) {
    const participants = e.source_agent === agentId || e.target_agents.includes(agentId);
    if (!participants) continue;
    if (Math.abs(e.sim_time - t) > window) continue;
    maxT = Math.max(maxT, e.metrics.tension);
  }
  return maxT;
}

export function computeThreadLayout(input: ThreadLayoutInput): ThreadLayoutOutput {
  const params: LayoutParameters = { ...DEFAULT_LAYOUT_PARAMETERS, ...input.parameters };
  const events = [...input.events].sort((a, b) => a.sim_time - b.sim_time);
  const normalized = normalizeAgents(input.agents);
  const agents = normalized.ids;

  const lanes = initializeLanes(agents, input.canvasHeight, params.lanePadding);
  const timeSamples = computeTimeSamples(events, params.timeResolution);

  // Thread layout should start from each agent's scenario-defined initial location.
  const initialLocations = normalized.initialLocations;

  const locationAtTime = buildLocationLookup(agents, events, initialLocations);
  const interactingAtTime = buildInteractionLookup(events, 1.0);

  const positionsByAgent = new Map<string, number[]>();
  for (const a of agents) positionsByAgent.set(a, []);

  let prevPositions = new Map<string, number>();
  for (const a of agents) prevPositions.set(a, lanes.get(a)!);

  for (const t of timeSamples) {
    const posAtT = computePositionsAtTime({
      t,
      agents,
      prevPositions,
      baseLanes: lanes,
      locationAtTime,
      interactingAtTime,
      params,
      canvasHeight: input.canvasHeight
    });

    for (const a of agents) {
      positionsByAgent.get(a)!.push(posAtT.get(a)!);
    }
    prevPositions = posAtT;
  }

  // ThreadPath controlPoints use sim_time as X; renderer maps to pixels.
  const threadPaths: ThreadPath[] = [];
  for (const agentId of agents) {
    const yValues = positionsByAgent.get(agentId)!;
    const controlPoints: [number, number][] = [];
    const thickness: number[] = [];

    for (let i = 0; i < timeSamples.length; i += 1) {
      controlPoints.push([timeSamples[i]!, yValues[i]!]);
      if (i < timeSamples.length - 1) {
        const avgTension =
          (tensionAtTimeFromEvents(events, agentId, timeSamples[i]!) +
            tensionAtTimeFromEvents(events, agentId, timeSamples[i + 1]!)) /
          2;
        thickness.push(2 + avgTension * 4);
      }
    }

    threadPaths.push({
      agentId,
      agentName: agentId,
      color: '#999999',
      controlPoints,
      thickness
    });
  }

  return { positions: positionsByAgent, timeSamples, threadPaths };
}
