import { describe, expect, test } from 'vitest';

import { computeThreadLayout, initializeLanes } from './threadLayout';
import { DeltaKind, DeltaOp, EventType, type Event } from '../types';

function makeEvent(args: {
  id: string;
  sim_time: number;
  type: EventType;
  source_agent: string;
  target_agents?: string[];
  location_id: string;
  causal_links?: string[];
  deltas?: Event['deltas'];
}): Event {
  return {
    id: args.id,
    sim_time: args.sim_time,
    tick_id: Math.floor(args.sim_time),
    order_in_tick: 0,
    type: args.type,
    source_agent: args.source_agent,
    target_agents: args.target_agents ?? [],
    location_id: args.location_id,
    causal_links: args.causal_links ?? [],
    deltas: args.deltas ?? [],
    description: args.id,
    metrics: {
      tension: 0.2,
      irony: 0,
      significance: 0,
      thematic_shift: {},
      tension_components: {
        danger: 0.2,
        time_pressure: 0.2,
        goal_frustration: 0.2,
        relationship_volatility: 0.2,
        information_gap: 0.2,
        resource_scarcity: 0.2,
        moral_cost: 0.2,
        irony_density: 0.2
      },
      irony_collapse: null
    }
  };
}

describe('computeThreadLayout', () => {
  test('produces thread paths with smooth sampled positions and no crossings', () => {
    const agents = ['a', 'b', 'c', 'd', 'e', 'f'];

    const events: Event[] = [];
    for (let i = 1; i <= 20; i += 1) {
      const id = `evt_${String(i).padStart(4, '0')}`;
      events.push(
        makeEvent({
          id,
          sim_time: i * 0.5,
          type: EventType.CHAT,
          source_agent: agents[i % agents.length]!,
          target_agents: [agents[(i + 1) % agents.length]!],
          location_id: 'dining_table',
          causal_links: i === 1 ? [] : [events[events.length - 1]!.id]
        })
      );
    }

    // Move agent d to kitchen at t=3.0, then back at t=5.0
    events.push(
      makeEvent({
        id: 'evt_0099',
        sim_time: 3.0,
        type: EventType.SOCIAL_MOVE,
        source_agent: 'd',
        target_agents: [],
        location_id: 'kitchen',
        causal_links: [events[events.length - 1]!.id],
        deltas: [
          {
            kind: DeltaKind.AGENT_LOCATION,
            agent: 'd',
            attribute: '',
            op: DeltaOp.SET,
            value: 'kitchen',
            reason_code: 'MOVE',
            reason_display: 'Moves to kitchen'
          }
        ]
      })
    );
    events.push(
      makeEvent({
        id: 'evt_0100',
        sim_time: 5.0,
        type: EventType.SOCIAL_MOVE,
        source_agent: 'd',
        target_agents: [],
        location_id: 'dining_table',
        causal_links: ['evt_0099'],
        deltas: [
          {
            kind: DeltaKind.AGENT_LOCATION,
            agent: 'd',
            attribute: '',
            op: DeltaOp.SET,
            value: 'dining_table',
            reason_code: 'MOVE',
            reason_display: 'Returns to table'
          }
        ]
      })
    );

    events.sort((a, b) => a.sim_time - b.sim_time);

    const canvasHeight = 600;
    const out = computeThreadLayout({ events, agents, scenes: [], canvasHeight });

    expect(out.timeSamples.length).toBeGreaterThan(0);
    expect(out.positions.size).toBe(agents.length);
    expect(out.threadPaths).toHaveLength(agents.length);

    // Position arrays align with timeSamples
    for (const a of agents) {
      expect(out.positions.get(a)!.length).toBe(out.timeSamples.length);
      expect(out.threadPaths.find((p) => p.agentId === a)!.controlPoints).toHaveLength(out.timeSamples.length);
    }

    // No crossings at any time sample: agent order stays monotone in Y.
    for (let i = 0; i < out.timeSamples.length; i += 1) {
      for (let j = 0; j < agents.length - 1; j += 1) {
        const ya = out.positions.get(agents[j]!)![i]!;
        const yb = out.positions.get(agents[j + 1]!)![i]!;
        expect(ya).toBeLessThan(yb);
      }
    }

    // When everyone is co-located, the bundle compresses vs base lane spread.
    const lanes = initializeLanes(agents, canvasHeight, 40);
    const baseYs = agents.map((a) => lanes.get(a)!);
    const baseSpread = Math.max(...baseYs) - Math.min(...baseYs);

    const sample0Ys = agents.map((a) => out.positions.get(a)![0]!);
    const spread0 = Math.max(...sample0Ys) - Math.min(...sample0Ys);
    expect(spread0).toBeLessThan(baseSpread);
  });

  test('single event produces valid layout', () => {
    const events = [
      makeEvent({
        id: 'evt_single',
        sim_time: 1.0,
        type: EventType.CHAT,
        source_agent: 'a',
        target_agents: ['b'],
        location_id: 'dining_table'
      })
    ];
    const agents = ['a', 'b'];
    const out = computeThreadLayout({ events, agents, scenes: [], canvasHeight: 400 });

    expect(out.timeSamples.length).toBeGreaterThan(0);
    expect(out.positions.size).toBe(2);
    expect(out.threadPaths).toHaveLength(2);
  });

  test('empty events produces empty time samples', () => {
    const out = computeThreadLayout({ events: [], agents: ['a', 'b'], scenes: [], canvasHeight: 400 });
    expect(out.timeSamples).toHaveLength(0);
    expect(out.positions.get('a')).toEqual([]);
    expect(out.positions.get('b')).toEqual([]);
  });

  test('single agent has one entry in positions map', () => {
    const events = [
      makeEvent({
        id: 'evt_solo',
        sim_time: 0,
        type: EventType.INTERNAL,
        source_agent: 'solo',
        target_agents: [],
        location_id: 'dining_table'
      })
    ];
    const out = computeThreadLayout({ events, agents: ['solo'], scenes: [], canvasHeight: 400 });

    expect(out.positions.size).toBe(1);
    expect(out.positions.has('solo')).toBe(true);
    expect(out.threadPaths).toHaveLength(1);
  });

  test('uses manifest initial_location when provided', () => {
    const events = [
      makeEvent({
        id: 'evt_initial',
        sim_time: 0.0,
        type: EventType.CHAT,
        source_agent: 'a',
        target_agents: [],
        location_id: 'dining_table'
      })
    ];

    const outDefault = computeThreadLayout({ events, agents: ['a', 'b'], scenes: [], canvasHeight: 400 });
    const outManifest = computeThreadLayout({
      events,
      agents: [
        { id: 'a', initial_location: 'dining_table' },
        { id: 'b', initial_location: 'balcony' }
      ],
      scenes: [],
      canvasHeight: 400
    });

    const defaultDistance = Math.abs(outDefault.positions.get('a')![0]! - outDefault.positions.get('b')![0]!);
    const manifestDistance = Math.abs(
      outManifest.positions.get('a')![0]! - outManifest.positions.get('b')![0]!
    );

    expect(manifestDistance).toBeGreaterThan(defaultDistance);
  });

  test('ignores non-location deltas for lane movement', () => {
    const events = [
      makeEvent({
        id: 'evt_0001',
        sim_time: 0.0,
        type: EventType.CHAT,
        source_agent: 'a',
        target_agents: ['b'],
        location_id: 'dining_table',
        deltas: [
          {
            kind: 'future_delta_kind',
            agent: 'a',
            attribute: '',
            op: DeltaOp.SET,
            value: 'noop',
            reason_code: 'FUTURE',
            reason_display: 'future'
          }
        ]
      }),
      makeEvent({
        id: 'evt_0002',
        sim_time: 1.0,
        type: EventType.CHAT,
        source_agent: 'b',
        target_agents: ['a'],
        location_id: 'dining_table',
        causal_links: ['evt_0001']
      })
    ];

    const out = computeThreadLayout({ events, agents: ['a', 'b'], scenes: [], canvasHeight: 400 });
    expect(out.positions.get('a')?.length).toBeGreaterThan(0);
    expect(out.positions.get('b')?.length).toBeGreaterThan(0);
  });
});
