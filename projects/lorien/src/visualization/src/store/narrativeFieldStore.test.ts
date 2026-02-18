import { beforeEach, describe, expect, test } from 'vitest';

import { useNarrativeFieldStore } from './narrativeFieldStore';
import { EventType, type NarrativeFieldPayload, type TensionWeights } from '../types';

const initialState = useNarrativeFieldStore.getState();

function makePayload(): NarrativeFieldPayload {
  return {
    format_version: '1.0.0',
    metadata: {
      simulation_id: 'sim_test',
      scenario: 'dinner_party',
      total_ticks: 1,
      total_sim_time: 1,
      agent_count: 2,
      event_count: 2,
      snapshot_interval: 20,
      timestamp: '2026-02-07T00:00:00Z'
    },
    agents: [
      {
        id: 'a',
        name: 'Agent A',
        initial_location: 'loc',
        goal_summary: '',
        primary_flaw: ''
      },
      {
        id: 'b',
        name: 'Agent B',
        initial_location: 'loc',
        goal_summary: '',
        primary_flaw: ''
      }
    ],
    locations: [],
    secrets: [],
    events: [
      {
        id: 'evt_0001',
        sim_time: 0,
        tick_id: 0,
        order_in_tick: 0,
        type: EventType.CHAT,
        source_agent: 'a',
        target_agents: ['b'],
        location_id: 'loc',
        causal_links: [],
        deltas: [],
        description: 'hello',
        metrics: {
          tension: 0,
          irony: 0,
          significance: 0,
          thematic_shift: {},
          tension_components: {
            danger: 1,
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
      },
      {
        id: 'evt_0002',
        sim_time: 1,
        tick_id: 1,
        order_in_tick: 0,
        type: EventType.OBSERVE,
        source_agent: 'b',
        target_agents: [],
        location_id: 'loc',
        causal_links: ['evt_0001'],
        deltas: [],
        description: 'observe',
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
    ],
    scenes: [],
    belief_snapshots: []
  };
}

beforeEach(() => {
  useNarrativeFieldStore.setState(initialState, true);
});

describe('NarrativeFieldStore', () => {
  test('loadEventLog handles empty events/scenes without crashing', () => {
    const payload = makePayload();
    payload.events = [];
    payload.scenes = [];
    payload.metadata.event_count = 0;
    payload.metadata.total_sim_time = 0;

    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const state = useNarrativeFieldStore.getState();
    expect(state.loadErrors).toBeNull();
    expect(state.events).toHaveLength(0);
    expect(state.scenes).toHaveLength(0);
    expect(state.timeDomain[0]).toBe(0);
    expect(state.timeDomain[1]).toBeGreaterThan(0);
  });

  test('loadEventLog parses JSON and builds indices', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const state = useNarrativeFieldStore.getState();
    expect(state.events).toHaveLength(2);

    const cone = state.causalIndex.get('evt_0002');
    expect(cone).toBeTruthy();
    expect(Array.from(cone!.backward)).toContain('evt_0001');

    expect(state.threadIndex.agents().sort()).toEqual(['a', 'b']);
    expect(state.computedTension.has('evt_0001')).toBe(true);
  });

  test('setTensionWeights recomputes computedTension', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const weights: TensionWeights = {
      danger: 1,
      time_pressure: 0,
      goal_frustration: 0,
      relationship_volatility: 0,
      information_gap: 0,
      resource_scarcity: 0,
      moral_cost: 0,
      irony_density: 0
    };

    useNarrativeFieldStore.getState().setTensionWeights(weights);

    const tension = useNarrativeFieldStore.getState().computedTension.get('evt_0001');
    expect(tension).toBe(1);
  });

  test('applyGenrePreset sets weights to spec values', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    useNarrativeFieldStore.getState().applyGenrePreset('thriller');

    const weights = useNarrativeFieldStore.getState().tensionWeights;
    expect(weights.danger).toBeCloseTo(2.5);
    expect(weights.time_pressure).toBeCloseTo(2.0);
    expect(weights.relationship_volatility).toBeCloseTo(0.5);
  });

  test('toggleEventTypeFilter removes/adds type', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const before = useNarrativeFieldStore.getState().activeFilters.eventTypeFilter;
    expect(before.has(EventType.CHAT)).toBe(true);

    useNarrativeFieldStore.getState().toggleEventTypeFilter(EventType.CHAT);
    const after = useNarrativeFieldStore.getState().activeFilters.eventTypeFilter;
    expect(after.has(EventType.CHAT)).toBe(false);

    useNarrativeFieldStore.getState().toggleEventTypeFilter(EventType.CHAT);
    const after2 = useNarrativeFieldStore.getState().activeFilters.eventTypeFilter;
    expect(after2.has(EventType.CHAT)).toBe(true);
  });

  test('setViewMode changes viewMode', () => {
    expect(useNarrativeFieldStore.getState().viewMode).toBe('threads');
    useNarrativeFieldStore.getState().setViewMode('topology');
    expect(useNarrativeFieldStore.getState().viewMode).toBe('topology');
    useNarrativeFieldStore.getState().setViewMode('threads');
    expect(useNarrativeFieldStore.getState().viewMode).toBe('threads');
  });

  test('selectArc sets selectedArcAgentId and can clear it', () => {
    expect(useNarrativeFieldStore.getState().selectedArcAgentId).toBeNull();
    useNarrativeFieldStore.getState().selectArc('a');
    expect(useNarrativeFieldStore.getState().selectedArcAgentId).toBe('a');
    useNarrativeFieldStore.getState().selectArc(null);
    expect(useNarrativeFieldStore.getState().selectedArcAgentId).toBeNull();
  });

  test('toggleCharacterFilter removes/adds agent from visibleAgents', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const before = useNarrativeFieldStore.getState().activeFilters.visibleAgents;
    expect(before.has('a')).toBe(true);

    useNarrativeFieldStore.getState().toggleCharacterFilter('a');
    expect(useNarrativeFieldStore.getState().activeFilters.visibleAgents.has('a')).toBe(false);

    useNarrativeFieldStore.getState().toggleCharacterFilter('a');
    expect(useNarrativeFieldStore.getState().activeFilters.visibleAgents.has('a')).toBe(true);
  });

  test('setViewportScale transitions zoomLevel correctly', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    useNarrativeFieldStore.getState().setViewportScale(0.3);
    expect(useNarrativeFieldStore.getState().zoomLevel).toBe('cloud');

    useNarrativeFieldStore.getState().setViewportScale(1.0);
    expect(useNarrativeFieldStore.getState().zoomLevel).toBe('threads');

    useNarrativeFieldStore.getState().setViewportScale(2.0);
    expect(useNarrativeFieldStore.getState().zoomLevel).toBe('detail');
  });

  test('pan updates viewport.x and viewport.y', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    const before = useNarrativeFieldStore.getState().viewport;
    useNarrativeFieldStore.getState().pan(0.1, 5);
    const after = useNarrativeFieldStore.getState().viewport;
    expect(after.y).toBe(before.y + 5);
  });

  test('fitAll resets viewport to full time domain with scale=1', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    // Zoom in first to change state
    useNarrativeFieldStore.getState().setViewportScale(3.0);
    expect(useNarrativeFieldStore.getState().zoomLevel).toBe('detail');

    useNarrativeFieldStore.getState().fitAll();
    const state = useNarrativeFieldStore.getState();
    expect(state.viewport.scale).toBe(1);
    expect(state.viewport.x).toBe(state.timeDomain[0]);
    expect(state.zoomLevel).toBe('threads');
  });

  test('topology viewport clamps to world bounds (x + y)', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));

    // Simulate a real canvas height so worldBounds.yMax is meaningful.
    useNarrativeFieldStore.getState().setViewport({ pixelHeight: 400, height: 400, y: 0 });

    useNarrativeFieldStore.getState().setViewMode('topology');
    useNarrativeFieldStore.getState().setViewportScale(2.0);

    const wb = useNarrativeFieldStore.getState().worldBounds!;
    const vp = useNarrativeFieldStore.getState().viewport;

    // Hard clamp at min.
    useNarrativeFieldStore.getState().setViewport({ x: -999, y: -999 });
    const afterMin = useNarrativeFieldStore.getState().viewport;
    expect(afterMin.x).toBeCloseTo(wb.xMin);
    expect(afterMin.y).toBeCloseTo(wb.yMin);

    // Hard clamp at max (accounting for span).
    useNarrativeFieldStore.getState().setViewport({ x: 999, y: 999 });
    const afterMax = useNarrativeFieldStore.getState().viewport;
    expect(afterMax.x).toBeCloseTo(wb.xMax - vp.width);
    expect(afterMax.y).toBeCloseTo(wb.yMax - vp.height);
  });

  test('topology zoom updates viewport.width and viewport.height proportionally', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));
    useNarrativeFieldStore.getState().setViewport({ pixelHeight: 500, height: 500, y: 0 });
    useNarrativeFieldStore.getState().setViewMode('topology');

    const wb = useNarrativeFieldStore.getState().worldBounds!;
    useNarrativeFieldStore.getState().setViewportScale(2.0);

    const vp = useNarrativeFieldStore.getState().viewport;
    expect(vp.width).toBeCloseTo((wb.xMax - wb.xMin) / 2);
    expect(vp.height).toBeCloseTo((wb.yMax - wb.yMin) / 2);
  });

  test('fitWorld sets viewport to the full world bounds (topology mode)', () => {
    const payload = makePayload();
    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));
    useNarrativeFieldStore.getState().setViewport({ pixelHeight: 360, height: 360, y: 0 });
    useNarrativeFieldStore.getState().setViewMode('topology');

    useNarrativeFieldStore.getState().fitWorld();

    const wb = useNarrativeFieldStore.getState().worldBounds!;
    const vp = useNarrativeFieldStore.getState().viewport;
    expect(vp.scale).toBeCloseTo(1);
    expect(vp.x).toBeCloseTo(wb.xMin);
    expect(vp.width).toBeCloseTo(wb.xMax - wb.xMin);
    expect(vp.y).toBeCloseTo(wb.yMin);
    expect(vp.height).toBeCloseTo(wb.yMax - wb.yMin);
  });

  test('setRegionSelection stores and clears region', () => {
    expect(useNarrativeFieldStore.getState().regionSelection).toBeNull();

    const region = { timeStart: 0, timeEnd: 1, agentIds: ['a'] };
    useNarrativeFieldStore.getState().setRegionSelection(region);
    expect(useNarrativeFieldStore.getState().regionSelection).toEqual(region);

    useNarrativeFieldStore.getState().setRegionSelection(null);
    expect(useNarrativeFieldStore.getState().regionSelection).toBeNull();
  });

  test('loadEventLog with invalid JSON sets loadErrors', () => {
    useNarrativeFieldStore.getState().loadEventLog('not valid json {{{');
    const state = useNarrativeFieldStore.getState();
    expect(state.loadErrors).not.toBeNull();
    expect(state.loadErrors!.length).toBeGreaterThan(0);
    expect(state.loadErrors![0]).toContain('Invalid JSON');
  });

  test('loadEventLog supports metadata fallback and location_memory deltas', () => {
    const payload = makePayload() as unknown as Record<string, unknown>;
    const metadata = payload.metadata as Record<string, unknown>;
    delete metadata.simulation_id;
    delete metadata.timestamp;
    metadata.deterministic_id = 'dinner_party_seed_51';

    const events = payload.events as Array<Record<string, unknown>>;
    const first = events[0]!;
    first.deltas = [
      {
        kind: 'location_memory',
        agent: 'a',
        attribute: 'loc',
        op: 'set',
        value: 0.4,
        reason_code: 'LOCATION_TENSION_STAIN',
        reason_display: 'updated'
      }
    ];

    useNarrativeFieldStore.getState().loadEventLog(JSON.stringify(payload));
    const state = useNarrativeFieldStore.getState();
    expect(state.loadErrors).toBeNull();
    expect(state.metadata?.simulation_id).toBe('dinner_party_seed_51');
    expect(state.metadata?.timestamp).toBe('unknown');
    expect(state.events[0]?.deltas[0]?.kind).toBe('location_memory');
  });
});
