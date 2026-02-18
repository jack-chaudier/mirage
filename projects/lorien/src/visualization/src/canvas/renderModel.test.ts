import { describe, expect, test } from 'vitest';

import {
  createTimeToX,
  interpolateY,
  buildRenderedEvents,
  buildThreadPathsWithMetadata,
  computeInteractionLinks
} from './renderModel';
import { EventType, ZoomLevel, type Event, type AgentManifest, type ThreadPath } from '../types';

function makeTestEvent(overrides: Partial<Event> & { id: string; sim_time: number }): Event {
  return {
    id: overrides.id,
    sim_time: overrides.sim_time,
    tick_id: overrides.tick_id ?? Math.floor(overrides.sim_time),
    order_in_tick: overrides.order_in_tick ?? 0,
    type: overrides.type ?? EventType.CHAT,
    source_agent: overrides.source_agent ?? 'a',
    target_agents: overrides.target_agents ?? [],
    location_id: overrides.location_id ?? 'loc',
    causal_links: overrides.causal_links ?? [],
    deltas: overrides.deltas ?? [],
    description: overrides.description ?? overrides.id,
    metrics: overrides.metrics ?? {
      tension: 0.5,
      irony: 0,
      significance: 0,
      thematic_shift: {},
      tension_components: {
        danger: 0, time_pressure: 0, goal_frustration: 0,
        relationship_volatility: 0, information_gap: 0,
        resource_scarcity: 0, moral_cost: 0, irony_density: 0
      },
      irony_collapse: null
    }
  };
}

describe('createTimeToX', () => {
  test('maps sim_time to pixel x correctly', () => {
    const timeToX = createTimeToX({
      viewport: { x: 0, y: 0, width: 10, height: 100, pixelWidth: 0, pixelHeight: 0, scale: 1 },
      width: 1000
    });
    expect(timeToX(0)).toBe(0);
    expect(timeToX(10)).toBe(1000);
    expect(timeToX(5)).toBe(500);
  });

  test('handles non-zero viewport offset', () => {
    const timeToX = createTimeToX({
      viewport: { x: 5, y: 0, width: 5, height: 100, pixelWidth: 0, pixelHeight: 0, scale: 1 },
      width: 500
    });
    expect(timeToX(5)).toBe(0);
    expect(timeToX(10)).toBe(500);
    expect(timeToX(7.5)).toBe(250);
  });
});

describe('interpolateY', () => {
  const timeSamples = [0, 1, 2, 3];
  const positions = new Map<string, number[]>([
    ['a', [100, 200, 300, 400]]
  ]);

  test('returns first y when t is before first sample', () => {
    expect(interpolateY({ agentId: 'a', t: -1, timeSamples, positions })).toBe(100);
  });

  test('returns last y when t is after last sample', () => {
    expect(interpolateY({ agentId: 'a', t: 10, timeSamples, positions })).toBe(400);
  });

  test('interpolates between samples', () => {
    const result = interpolateY({ agentId: 'a', t: 0.5, timeSamples, positions });
    expect(result).toBe(150);
  });

  test('returns 0 for unknown agent', () => {
    expect(interpolateY({ agentId: 'unknown', t: 1, timeSamples, positions })).toBe(0);
  });
});

describe('buildRenderedEvents', () => {
  test('produces correct RenderedEvent objects with radius by zoom level', () => {
    const evt = makeTestEvent({ id: 'e1', sim_time: 1, source_agent: 'a' });
    const timeSamples = [0, 1, 2];
    const positions = new Map([['a', [50, 100, 150]]]);
    const timeToX = (t: number) => t * 100;
    const tension = new Map([['e1', 0.7]]);

    const cloudResult = buildRenderedEvents({
      events: [evt], computedTension: tension,
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.CLOUD
    });
    expect(cloudResult[0]!.radius).toBe(4); // baseRadius=4 + significance(0)*2

    const threadResult = buildRenderedEvents({
      events: [evt], computedTension: tension,
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.THREADS
    });
    expect(threadResult[0]!.radius).toBe(8);

    const detailResult = buildRenderedEvents({
      events: [evt], computedTension: tension,
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.DETAIL
    });
    expect(detailResult[0]!.radius).toBe(12);
  });

  test('opacity scales with tension', () => {
    const evt = makeTestEvent({ id: 'e1', sim_time: 1 });
    const timeSamples = [0, 1];
    const positions = new Map([['a', [50, 100]]]);
    const timeToX = (t: number) => t * 100;

    const lowTension = buildRenderedEvents({
      events: [evt], computedTension: new Map([['e1', 0]]),
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.THREADS
    });
    const highTension = buildRenderedEvents({
      events: [evt], computedTension: new Map([['e1', 1.0]]),
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.THREADS
    });

    expect(lowTension[0]!.opacity).toBeLessThan(highTension[0]!.opacity);
  });

  test('glowColor changes based on tension thresholds', () => {
    const evt = makeTestEvent({ id: 'e1', sim_time: 1 });
    const timeSamples = [0, 1];
    const positions = new Map([['a', [50, 100]]]);
    const timeToX = (t: number) => t * 100;

    const low = buildRenderedEvents({
      events: [evt], computedTension: new Map([['e1', 0.1]]),
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.THREADS
    });
    expect(low[0]!.glowColor).toBe('rgba(0,0,0,0)');

    const high = buildRenderedEvents({
      events: [evt], computedTension: new Map([['e1', 0.9]]),
      timeSamples, positions, timeToX, zoomLevel: ZoomLevel.THREADS
    });
    expect(high[0]!.glowColor).toBe('#cc4444');
  });
});

describe('buildThreadPathsWithMetadata', () => {
  test('maps agent names and colors correctly', () => {
    const basePaths: ThreadPath[] = [
      { agentId: 'thorne', agentName: 'thorne', color: '#999999', controlPoints: [[0, 50]], thickness: [] },
      { agentId: 'unknown_id', agentName: 'unknown_id', color: '#999999', controlPoints: [[0, 100]], thickness: [] }
    ];
    const agents: AgentManifest[] = [
      { id: 'thorne', name: 'Thorne', initial_location: 'loc', goal_summary: '', primary_flaw: '' }
    ];

    const result = buildThreadPathsWithMetadata({ basePaths, agents });

    expect(result[0]!.agentName).toBe('Thorne');
    // thorne has a defined color in CHARACTER_COLORS
    expect(result[0]!.color).toBe('#E69F00');

    // unknown agent falls back to agentId for name
    expect(result[1]!.agentName).toBe('unknown_id');
  });
});

describe('computeInteractionLinks', () => {
  test('emits links only when threads are within dy threshold', () => {
    const e1 = makeTestEvent({ id: 'e1', sim_time: 1, source_agent: 'a', target_agents: ['b'], type: EventType.CONFLICT });
    const e2 = makeTestEvent({ id: 'e2', sim_time: 1, source_agent: 'a', target_agents: ['c'], type: EventType.CONFLICT });

    const timeSamples = [0, 1, 2];
    const positions = new Map<string, number[]>([
      ['a', [100, 100, 100]],
      ['b', [110, 110, 110]], // within 15px
      ['c', [250, 250, 250]] // far
    ]);

    const links = computeInteractionLinks({
      events: [e1, e2],
      timeSamples,
      positions,
      tMin: 0,
      tMax: 2,
      maxDyPx: 15
    });

    expect(links.length).toBe(1);
    expect(links[0]!.a).toBe('a');
    expect(links[0]!.b).toBe('b');
  });

  test('includes pair links among all participants in an event', () => {
    const e = makeTestEvent({
      id: 'e1',
      sim_time: 1,
      source_agent: 'a',
      target_agents: ['b', 'c'],
      type: EventType.CONFLICT
    });

    const timeSamples = [0, 1, 2];
    const positions = new Map<string, number[]>([
      ['a', [100, 100, 100]],
      ['b', [108, 108, 108]],
      ['c', [112, 112, 112]]
    ]);

    const links = computeInteractionLinks({
      events: [e],
      timeSamples,
      positions,
      tMin: 0,
      tMax: 2,
      maxDyPx: 15
    });

    const keys = new Set(links.map((l) => `${l.a}|${l.b}`));
    expect(keys.has('a|b')).toBe(true);
    expect(keys.has('a|c')).toBe(true);
    expect(keys.has('b|c')).toBe(true);
    expect(links.length).toBe(3);
  });

  test('deduplicates by rounded time bucket, keeping highest tension', () => {
    const e1 = makeTestEvent({ id: 'e1', sim_time: 1.01, source_agent: 'a', target_agents: ['b'], type: EventType.CONFLICT });
    const e2 = makeTestEvent({ id: 'e2', sim_time: 1.04, source_agent: 'a', target_agents: ['b'], type: EventType.CONFLICT });

    const timeSamples = [0, 1, 2];
    const positions = new Map<string, number[]>([
      ['a', [100, 100, 100]],
      ['b', [110, 110, 110]]
    ]);

    const tension = new Map<string, number>([
      ['e1', 0.3],
      ['e2', 0.9]
    ]);

    const links = computeInteractionLinks({
      events: [e1, e2],
      timeSamples,
      positions,
      tMin: 0,
      tMax: 2,
      maxDyPx: 15,
      tensionById: tension
    });

    expect(links.length).toBe(1);
    expect(links[0]!.t).toBeCloseTo(1.04);
  });
});
