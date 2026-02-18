import { describe, expect, test } from 'vitest';

import { buildCausalIndex } from './causalIndex';
import { EventType, type Event } from '../types';

function makeEvent(id: string, causal_links: string[]): Event {
  return {
    id,
    sim_time: 0,
    tick_id: 0,
    order_in_tick: 0,
    type: EventType.CHAT,
    source_agent: 'a',
    target_agents: [],
    location_id: 'loc',
    causal_links,
    deltas: [],
    description: id,
    metrics: {
      tension: 0,
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

describe('buildCausalIndex', () => {
  test('precomputes BFS-3 backward and forward cones', () => {
    // evt_0001 -> evt_0002 -> evt_0003 -> evt_0004 -> evt_0005
    const events: Event[] = [
      makeEvent('evt_0001', []),
      makeEvent('evt_0002', ['evt_0001']),
      makeEvent('evt_0003', ['evt_0002']),
      makeEvent('evt_0004', ['evt_0003']),
      makeEvent('evt_0005', ['evt_0004'])
    ];

    const index = buildCausalIndex(events, 3);

    const cone3 = index.get('evt_0003');
    expect(cone3).toBeTruthy();
    expect(Array.from(cone3!.backward).sort()).toEqual(['evt_0001', 'evt_0002']);
    expect(Array.from(cone3!.forward).sort()).toEqual(['evt_0004', 'evt_0005']);

    const cone2 = index.get('evt_0002');
    expect(Array.from(cone2!.backward).sort()).toEqual(['evt_0001']);
    expect(Array.from(cone2!.forward).sort()).toEqual(['evt_0003', 'evt_0004', 'evt_0005']);
  });
});

