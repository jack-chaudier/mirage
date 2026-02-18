import { describe, expect, test } from 'vitest';

import { parseNarrativeFieldPayload, validateNarrativeFieldPayloadCli } from './loader';
import { DeltaKind, EventType } from '../types';

function makeMinimalPayloadObj() {
  return {
    format_version: '1.0.0',
    metadata: {
      simulation_id: 'sim_test',
      scenario: 'dinner_party',
      total_ticks: 0,
      total_sim_time: 0,
      agent_count: 0,
      event_count: 0,
      snapshot_interval: 20,
      timestamp: '2026-02-07T00:00:00Z'
    },
    agents: [],
    locations: [],
    secrets: [],
    events: [],
    scenes: [],
    belief_snapshots: []
  };
}

describe('parseNarrativeFieldPayload', () => {
  test('invalid JSON string returns errors', () => {
    const result = parseNarrativeFieldPayload('not json {{');
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0]).toContain('Invalid JSON');
    }
  });

  test('missing required top-level fields returns errors', () => {
    const result = parseNarrativeFieldPayload(JSON.stringify({ metadata: {} }));
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errors.some((e) => e.includes('agents'))).toBe(true);
      expect(result.errors.some((e) => e.includes('events'))).toBe(true);
    }
  });

  test('empty events array with valid structure succeeds', () => {
    const payload = makeMinimalPayloadObj();
    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.payload.events).toHaveLength(0);
    }
  });

  test('event with invalid EventType returns error mentioning valid types', () => {
    const payload = makeMinimalPayloadObj();
    payload.events = [
      {
        id: 'evt_001',
        sim_time: 0,
        tick_id: 0,
        order_in_tick: 0,
        type: 'INVALID_TYPE',
        source_agent: 'a',
        target_agents: [],
        location_id: 'loc',
        causal_links: [],
        deltas: [],
        description: 'test',
        metrics: {
          tension: 0,
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
      }
    ] as unknown as typeof payload.events;

    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(false);
    if (!result.success) {
      const typeError = result.errors.find((e) => e.includes('EventType'));
      expect(typeError).toBeDefined();
      // Should mention at least one valid type
      expect(typeError).toContain(EventType.CHAT);
    }
  });

  test('missing simulation_id/timestamp falls back to deterministic_id/unknown', () => {
    const payload = makeMinimalPayloadObj();
    payload.metadata = {
      deterministic_id: 'dinner_party_seed_51',
      scenario: 'dinner_party',
      total_ticks: 2,
      total_sim_time: 3,
      agent_count: 0,
      event_count: 0,
      snapshot_interval: 20
    } as unknown as typeof payload.metadata;

    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.payload.metadata.simulation_id).toBe('dinner_party_seed_51');
      expect(result.payload.metadata.timestamp).toBe('unknown');
    }
  });

  test('location_memory delta kind is accepted', () => {
    const payload = makeMinimalPayloadObj();
    payload.metadata.event_count = 1;
    payload.events = [
      {
        id: 'evt_0001',
        sim_time: 0,
        tick_id: 0,
        order_in_tick: 0,
        type: EventType.CATASTROPHE,
        source_agent: 'a',
        target_agents: [],
        location_id: 'dining_table',
        causal_links: [],
        deltas: [
          {
            kind: DeltaKind.LOCATION_MEMORY,
            agent: 'a',
            attribute: 'dining_table',
            op: 'set',
            value: 0.7,
            reason_code: 'LOCATION_TENSION_STAIN',
            reason_display: 'Tension residue updated'
          }
        ],
        description: 'catastrophe',
        metrics: {
          tension: 0.9,
          irony: 0.1,
          significance: 0.3,
          thematic_shift: {},
          tension_components: {
            danger: 0.9,
            time_pressure: 0.1,
            goal_frustration: 0.1,
            relationship_volatility: 0.1,
            information_gap: 0.1,
            resource_scarcity: 0.1,
            moral_cost: 0.1,
            irony_density: 0.1
          },
          irony_collapse: null
        }
      }
    ] as unknown as typeof payload.events;

    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(true);
  });

  test('unknown delta kind is preserved (warn-and-pass)', () => {
    const payload = makeMinimalPayloadObj();
    payload.metadata.event_count = 1;
    payload.events = [
      {
        id: 'evt_0001',
        sim_time: 0,
        tick_id: 0,
        order_in_tick: 0,
        type: EventType.CHAT,
        source_agent: 'a',
        target_agents: [],
        location_id: 'dining_table',
        causal_links: [],
        deltas: [
          {
            kind: 'future_delta_kind',
            agent: 'a',
            attribute: '',
            op: 'set',
            value: 'x',
            reason_code: 'FUTURE',
            reason_display: 'future'
          }
        ],
        description: 'chat',
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
      }
    ] as unknown as typeof payload.events;

    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.payload.events[0]?.deltas[0]?.kind).toBe('future_delta_kind');
    }
  });

  test('CLI validator allows non-secret belief keys (claims)', () => {
    const payload = makeMinimalPayloadObj() as unknown as {
      metadata: Record<string, unknown>;
      agents: Array<Record<string, unknown>>;
      belief_snapshots: Array<Record<string, unknown>>;
    };
    payload.agents = [
      {
        id: 'thorne',
        name: 'Thorne',
        initial_location: 'dining_table',
        goal_summary: '',
        primary_flaw: ''
      }
    ];
    payload.metadata.agent_count = 1;
    payload.belief_snapshots = [
      {
        tick_id: 0,
        sim_time: 0,
        beliefs: {
          thorne: {
            claim_thorne_health: 'unknown'
          }
        },
        agent_irony: {},
        scene_irony: 0
      }
    ] as unknown as typeof payload.belief_snapshots;

    const result = parseNarrativeFieldPayload(JSON.stringify(payload));
    expect(result.success).toBe(true);
    if (result.success) {
      expect(validateNarrativeFieldPayloadCli(result.payload)).toEqual([]);
    }
  });
});
