import { describe, expect, test } from 'vitest';

import { buildTopologyEventLabels } from './topologyEventLabels';
import { EventType, ZoomLevel, type Event, type RenderedEvent } from '../types';

function makeEvent(overrides: Partial<Event> & { id: string; type: EventType; description: string }): Event {
  return {
    id: overrides.id,
    sim_time: overrides.sim_time ?? 1,
    tick_id: overrides.tick_id ?? 1,
    order_in_tick: overrides.order_in_tick ?? 0,
    type: overrides.type,
    source_agent: overrides.source_agent ?? 'a',
    target_agents: overrides.target_agents ?? [],
    location_id: overrides.location_id ?? 'loc',
    causal_links: overrides.causal_links ?? [],
    deltas: overrides.deltas ?? [],
    description: overrides.description,
    dialogue: overrides.dialogue ?? null,
    content_metadata: overrides.content_metadata ?? null,
    beat_type: overrides.beat_type ?? null,
    metrics: overrides.metrics ?? {
      tension: 0.5,
      irony: 0,
      significance: 0,
      thematic_shift: {},
      tension_components: {
        danger: 0,
        time_pressure: 0,
        goal_frustration: 0,
        relationship_volatility: 0,
        information_gap: 0,
        resource_scarcity: 0,
        moral_cost: 0,
        irony_density: 0
      },
      irony_collapse: null
    }
  };
}

function makeRenderedEvent(overrides: Partial<RenderedEvent> & { eventId: string; x: number; y: number }): RenderedEvent {
  return {
    eventId: overrides.eventId,
    simTime: overrides.simTime ?? 1,
    description: overrides.description ?? overrides.eventId,
    eventType: overrides.eventType ?? EventType.REVEAL,
    beatType: overrides.beatType ?? null,
    x: overrides.x,
    y: overrides.y,
    radius: overrides.radius ?? 8,
    color: overrides.color ?? '#fff',
    opacity: overrides.opacity ?? 1,
    glowIntensity: overrides.glowIntensity ?? 0.5,
    glowColor: overrides.glowColor ?? 'rgba(0,0,0,0)',
    threadId: overrides.threadId ?? 'a',
    zoomVisibility: overrides.zoomVisibility ?? ZoomLevel.THREADS
  };
}

function fakeMeasure(text: string): number {
  return Array.from(text).length * 7;
}

describe('buildTopologyEventLabels', () => {
  test('dedup prefers Tier 1 over Tier 2 even if Tier 2 has higher tension', () => {
    const events: Event[] = [
      makeEvent({ id: 'e1', type: EventType.CATASTROPHE, description: 'Confront Marcus Webb' }),
      makeEvent({ id: 'e2', type: EventType.CONFLICT, description: 'Confront Marcus Webb', target_agents: ['b'] })
    ];
    const renderedById = new Map<string, RenderedEvent>([
      ['e1', makeRenderedEvent({ eventId: 'e1', x: 100, y: 80, eventType: EventType.CATASTROPHE })],
      ['e2', makeRenderedEvent({ eventId: 'e2', x: 120, y: 90, eventType: EventType.CONFLICT })]
    ]);
    const computedTension = new Map<string, number>([
      ['e1', 0.2],
      ['e2', 0.95]
    ]);
    const agentNameById = new Map<string, string>([
      ['a', 'Diana'],
      ['b', 'Marcus']
    ]);

    const labels = buildTopologyEventLabels({
      width: 600,
      height: 300,
      events,
      renderedById,
      computedTension,
      agentNameById,
      zoomLevel: ZoomLevel.THREADS,
      viewportScale: 1.0,
      measureTextWidth: fakeMeasure
    });

    expect(labels.length).toBe(1);
    expect(labels[0]!.eventId).toBe('e1');
    expect(labels[0]!.tier).toBe(1);
  });

  test('skips labels when collisions cannot be resolved', () => {
    const events: Event[] = [
      makeEvent({ id: 'e1', type: EventType.CATASTROPHE, description: 'Catastrophe One' }),
      makeEvent({ id: 'e2', type: EventType.CATASTROPHE, description: 'Catastrophe Two' })
    ];
    const renderedById = new Map<string, RenderedEvent>([
      ['e1', makeRenderedEvent({ eventId: 'e1', x: 100, y: 20, eventType: EventType.CATASTROPHE })],
      ['e2', makeRenderedEvent({ eventId: 'e2', x: 102, y: 22, eventType: EventType.CATASTROPHE })]
    ]);
    const computedTension = new Map<string, number>([
      ['e1', 0.9],
      ['e2', 0.8]
    ]);
    const agentNameById = new Map<string, string>([['a', 'Victor']]);

    // Very small canvas forces clamping to a single rect; wide labels then collide no matter the side.
    const labels = buildTopologyEventLabels({
      width: 190,
      height: 26,
      events,
      renderedById,
      computedTension,
      agentNameById,
      zoomLevel: ZoomLevel.THREADS,
      viewportScale: 1.0,
      measureTextWidth: fakeMeasure
    });

    expect(labels.length).toBe(1);
    expect(labels[0]!.eventId).toBe('e1');
  });

  test('uses alternate placements to avoid overlap when possible', () => {
    const events: Event[] = [
      makeEvent({ id: 'e1', type: EventType.CATASTROPHE, description: 'Major disaster at table' }),
      makeEvent({ id: 'e2', type: EventType.CATASTROPHE, description: 'Major disaster in kitchen' })
    ];
    const renderedById = new Map<string, RenderedEvent>([
      ['e1', makeRenderedEvent({ eventId: 'e1', x: 240, y: 100, eventType: EventType.CATASTROPHE })],
      ['e2', makeRenderedEvent({ eventId: 'e2', x: 248, y: 95, eventType: EventType.CATASTROPHE })]
    ]);
    const computedTension = new Map<string, number>([
      ['e1', 0.9],
      ['e2', 0.85]
    ]);
    const agentNameById = new Map<string, string>([['a', 'Diana']]);

    const labels = buildTopologyEventLabels({
      width: 800,
      height: 400,
      events,
      renderedById,
      computedTension,
      agentNameById,
      zoomLevel: ZoomLevel.THREADS,
      viewportScale: 1.0,
      measureTextWidth: fakeMeasure
    });

    expect(labels.length).toBe(2);
    const a = labels.find((l) => l.eventId === 'e1')!;
    const b = labels.find((l) => l.eventId === 'e2')!;
    const overlap = a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
    expect(overlap).toBe(false);
  });

  test('truncates long labels to max width', () => {
    const long = 'A very long reveal description that should be truncated to fit the maximum label width.';
    const events: Event[] = [makeEvent({ id: 'e1', type: EventType.REVEAL, description: long })];
    const renderedById = new Map<string, RenderedEvent>([
      ['e1', makeRenderedEvent({ eventId: 'e1', x: 200, y: 140, eventType: EventType.REVEAL })]
    ]);
    const computedTension = new Map<string, number>([['e1', 0.7]]);
    const agentNameById = new Map<string, string>([['a', 'Diana']]);

    const labels = buildTopologyEventLabels({
      width: 800,
      height: 400,
      events,
      renderedById,
      computedTension,
      agentNameById,
      zoomLevel: ZoomLevel.THREADS,
      viewportScale: 1.0,
      measureTextWidth: fakeMeasure
    });

    expect(labels.length).toBe(1);
    expect(labels[0]!.w).toBeLessThanOrEqual(180);
    expect(labels[0]!.text.endsWith('...')).toBe(true);
  });
});
