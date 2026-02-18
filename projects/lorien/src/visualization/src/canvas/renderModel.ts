import { CHARACTER_COLORS, EVENT_TYPE_COLORS } from '../constants/colors';
import { ZoomLevel } from '../types';
import { EventType } from '../types';
import type { AgentManifest, Event, RenderedEvent, ThreadPath, ViewportState } from '../types';

export function createTimeToX(args: { viewport: ViewportState; width: number }): (t: number) => number {
  const { viewport, width } = args;
  const minT = viewport.x;
  const maxT = viewport.x + viewport.width;
  const span = Math.max(1e-6, maxT - minT);
  return (t: number) => ((t - minT) / span) * width;
}

export function interpolateY(args: {
  agentId: string;
  t: number;
  timeSamples: number[];
  positions: Map<string, number[]>;
}): number {
  const { agentId, t, timeSamples, positions } = args;
  const ys = positions.get(agentId);
  if (!ys || ys.length === 0 || timeSamples.length === 0) return 0;

  if (t <= timeSamples[0]!) return ys[0]!;
  if (t >= timeSamples[timeSamples.length - 1]!) return ys[ys.length - 1]!;

  // Find i where timeSamples[i] <= t < timeSamples[i+1]
  let lo = 0;
  let hi = timeSamples.length - 2;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (timeSamples[mid]! <= t && t < timeSamples[mid + 1]!) {
      lo = mid;
      break;
    }
    if (timeSamples[mid]! < t) lo = mid + 1;
    else hi = mid - 1;
  }
  const i = Math.max(0, Math.min(timeSamples.length - 2, lo));
  const t0 = timeSamples[i]!;
  const t1 = timeSamples[i + 1]!;
  const y0 = ys[i]!;
  const y1 = ys[i + 1]!;
  const alpha = (t - t0) / Math.max(1e-6, t1 - t0);
  return y0 + (y1 - y0) * alpha;
}

export type InteractionLink = {
  t: number;
  a: string;
  b: string;
  yA: number;
  yB: number;
};

export function computeInteractionLinks(args: {
  events: Event[];
  timeSamples: number[];
  positions: Map<string, number[]>;
  tMin: number;
  tMax: number;
  maxDyPx?: number;
  tensionById?: Map<string, number>;
}): InteractionLink[] {
  const { events, timeSamples, positions, tMin, tMax, maxDyPx, tensionById } = args;
  const dyMax = maxDyPx ?? 15;

  const best = new Map<
    string,
    {
      score: number;
      link: InteractionLink;
    }
  >();

  for (const e of events) {
    if (e.sim_time < tMin || e.sim_time > tMax) continue;

    const participants = new Set<string>([e.source_agent, ...e.target_agents]);
    const list = Array.from(participants);
    if (list.length < 2) continue;

    const score = tensionById?.get(e.id) ?? e.metrics.tension ?? 0;
    const roundedT = Math.round(e.sim_time * 10) / 10;

    for (let i = 0; i < list.length; i += 1) {
      for (let j = i + 1; j < list.length; j += 1) {
        const a0 = list[i]!;
        const b0 = list[j]!;
        const a = a0 < b0 ? a0 : b0;
        const b = a0 < b0 ? b0 : a0;

        const yA = interpolateY({ agentId: a, t: e.sim_time, timeSamples, positions });
        const yB = interpolateY({ agentId: b, t: e.sim_time, timeSamples, positions });

        if (Math.abs(yA - yB) > dyMax) continue;

        const key = `${roundedT}|${a}|${b}`;
        const prev = best.get(key);
        if (!prev || score > prev.score) {
          best.set(key, { score, link: { t: e.sim_time, a, b, yA, yB } });
        }
      }
    }
  }

  const out = Array.from(best.values()).map((v) => v.link);
  out.sort((x, y) => x.t - y.t);
  return out;
}

function tensionToGlowHex(tension: number): string {
  if (tension < 0.2) return 'rgba(0,0,0,0)';
  if (tension < 0.4) return '#4488cc';
  if (tension < 0.6) return '#cc8844';
  if (tension < 0.8) return '#cc6633';
  return '#cc4444';
}

export function buildThreadPathsWithMetadata(args: {
  basePaths: ThreadPath[];
  agents: AgentManifest[];
}): ThreadPath[] {
  const nameById = new Map(args.agents.map((a) => [a.id, a.name] as const));
  return args.basePaths.map((p) => ({
    ...p,
    agentName: nameById.get(p.agentId) ?? p.agentId,
    color: CHARACTER_COLORS[p.agentId] ?? p.color ?? '#999999'
  }));
}

export function buildRenderedEvents(args: {
  events: Event[];
  computedTension: Map<string, number>;
  timeSamples: number[];
  positions: Map<string, number[]>;
  timeToX: (t: number) => number;
  zoomLevel: ZoomLevel;
}): RenderedEvent[] {
  const { events, computedTension, timeSamples, positions, timeToX, zoomLevel } = args;

  const baseRadius =
    zoomLevel === ZoomLevel.CLOUD ? 4 : zoomLevel === ZoomLevel.THREADS ? 8 : 12;

  return events.map((e) => {
    const tension = computedTension.get(e.id) ?? e.metrics.tension ?? 0;
    const x = timeToX(e.sim_time);
    const y = interpolateY({ agentId: e.source_agent, t: e.sim_time, timeSamples, positions });

    const radius = baseRadius + (e.metrics.significance ?? 0) * 2;
    const color = EVENT_TYPE_COLORS[e.type];

    const zoomVisibility =
      (e.type === EventType.CHAT || e.type === EventType.PHYSICAL) && tension < 0.12
        ? ZoomLevel.DETAIL
        : ZoomLevel.THREADS;

    return {
      eventId: e.id,
      simTime: e.sim_time,
      description: e.description,
      eventType: e.type,
      beatType: e.beat_type ?? null,
      x,
      y,
      radius,
      color,
      opacity: Math.min(1, 0.25 + tension * 0.85),
      glowIntensity: tension,
      glowColor: tensionToGlowHex(tension),
      threadId: e.source_agent,
      zoomVisibility
    };
  });
}
