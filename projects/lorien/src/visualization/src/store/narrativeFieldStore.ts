import { useMemo } from 'react';
import { create } from 'zustand';
import { useShallow } from 'zustand/react/shallow';

import { buildCausalIndex } from '../data/causalIndex';
import { parseNarrativeFieldPayload } from '../data/loader';
import { recomputeTensionMap } from '../data/tensionComputer';
import { DEFAULT_TENSION_WEIGHTS, GENRE_PRESETS, type GenrePresetId } from '../constants/tensionPresets';
import {
  type AgentManifest,
  type AxisConfig,
  type CausalIndex,
  type Event,
  EventType,
  type FilterState,
  type LocationDefinition,
  type NarrativeFieldPayload,
  type RegionSelection,
  type RenderedEvent,
  type Scene,
  type SecretDefinitionPayload,
  type ThreadIndex,
  type ThreadPath,
  type TensionWeights,
  type ViewportState,
  type ViewMode,
  ZoomLevel
} from '../types';

let fitWorldRafId: number | null = null;

export type WorldBounds = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

export interface NarrativeFieldStore {
  // --- Data Layer ---
  metadata: NarrativeFieldPayload['metadata'] | null;
  agents: AgentManifest[];
  locations: LocationDefinition[];
  secrets: SecretDefinitionPayload[];
  events: Event[];
  scenes: Scene[];
  // Pre-staged for the planned belief matrix heatmap; currently stored but not rendered.
  beliefSnapshots: NarrativeFieldPayload['belief_snapshots'];
  causalIndex: CausalIndex;
  threadIndex: ThreadIndex;
  timeDomain: [number, number];

  // --- Metrics Layer ---
  tensionWeights: TensionWeights;
  computedTension: Map<string, number>;

  // --- View State ---
  viewMode: ViewMode;
  viewport: ViewportState;
  worldBounds: WorldBounds | null;
  zoomLevel: ZoomLevel;
  axisConfig: AxisConfig;

  // --- Interaction State ---
  hoveredEventId: string | null;
  selectedEventId: string | null;
  selectedArcAgentId: string | null;
  activeFilters: FilterState;

  // --- Selection State ---
  regionSelection: RegionSelection | null;

  // --- Layout Cache ---
  renderedEvents: RenderedEvent[];
  threadPaths: ThreadPath[];

  // --- Error State ---
  loadErrors: string[] | null;

  // --- Actions ---
  loadEventLog: (json: string) => void;
  clearLoadErrors: () => void;
  setViewMode: (mode: ViewMode) => void;
  setTensionWeights: (weights: Partial<TensionWeights>) => void;
  applyGenrePreset: (preset: GenrePresetId) => void;
  resetTensionWeights: () => void;
  setHoveredEvent: (id: string | null) => void;
  selectEvent: (id: string | null) => void;
  selectArc: (agentId: string | null) => void;
  setZoom: (level: ZoomLevel) => void;
  setViewport: (viewport: Partial<ViewportState>) => void;
  setViewportScale: (scale: number, anchorTime?: number, anchorY?: number) => void;
  pan: (dx: number, dy: number) => void;
  fitAll: () => void;
  fitWorld: (opts?: { animate?: boolean }) => void;
  toggleCharacterFilter: (agentId: string) => void;
  toggleEventTypeFilter: (type: EventType) => void;
  setRegionSelection: (region: RegionSelection | null) => void;
  setAxisConfig: (config: AxisConfig) => void;
}

function makeDefaultAxisConfig(): AxisConfig {
  return {
    mode: 'sim_time',
    label: 'Sim Time (min)',
    mapper: (e) => e.sim_time
  };
}

function makeDefaultViewport(): ViewportState {
  return {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
    pixelWidth: 0,
    pixelHeight: 0,
    scale: 1
  };
}

function makeEmptyCausalIndex(): CausalIndex {
  return {
    get() {
      return undefined;
    }
  };
}

function buildThreadIndex(events: Event[]): ThreadIndex {
  const map = new Map<string, string[]>();
  const agentIds = new Set<string>();

  for (const e of events) {
    const participants = [e.source_agent, ...e.target_agents.filter((t) => t !== e.source_agent)];
    for (const agentId of participants) {
      agentIds.add(agentId);
      const list = map.get(agentId);
      if (list) list.push(e.id);
      else map.set(agentId, [e.id]);
    }
  }

  const agentList = Array.from(agentIds).sort();

  return {
    get(agentId: string) {
      return map.get(agentId);
    },
    agents() {
      return agentList;
    }
  };
}

function makeDefaultFilters(agentIds: string[], timeDomain: [number, number]): FilterState {
  return {
    visibleAgents: new Set(agentIds),
    eventTypeFilter: new Set(Object.values(EventType) as EventType[]),
    minTension: 0,
    timeRange: timeDomain
  };
}

export const useNarrativeFieldStore = create<NarrativeFieldStore>((set, get) => ({
  metadata: null,
  agents: [],
  locations: [],
  secrets: [],
  events: [],
  scenes: [],
  beliefSnapshots: [],
  causalIndex: makeEmptyCausalIndex(),
  threadIndex: {
    get() {
      return undefined;
    },
    agents() {
      return [];
    }
  },
  timeDomain: [0, 1],

  tensionWeights: DEFAULT_TENSION_WEIGHTS,
  computedTension: new Map(),

  viewMode: 'threads',
  viewport: makeDefaultViewport(),
  worldBounds: null,
  zoomLevel: ZoomLevel.THREADS,
  axisConfig: makeDefaultAxisConfig(),

  hoveredEventId: null,
  selectedEventId: null,
  selectedArcAgentId: null,
  activeFilters: makeDefaultFilters([], [0, 1]),

  regionSelection: null,

  renderedEvents: [],
  threadPaths: [],
  loadErrors: null,

  loadEventLog(json: string) {
    const parsed = parseNarrativeFieldPayload(json);
    if (!parsed.success) {
      set({ loadErrors: parsed.errors });
      return;
    }
    const payload = parsed.payload;

    const minTime = payload.events.reduce(
      (acc, e) => Math.min(acc, e.sim_time),
      Number.POSITIVE_INFINITY
    );
    const causalIndex = buildCausalIndex(payload.events);
    const threadIndex = buildThreadIndex(payload.events);
    const computedTension = recomputeTensionMap(payload.events, get().tensionWeights);

    const agentIds = payload.agents.map((a) => a.id);

    // Set viewport width to cover the time domain by default.
    const maxTime = payload.events.reduce((acc, e) => Math.max(acc, e.sim_time), 0);
    const timeDomain: [number, number] = [
      Number.isFinite(minTime) ? minTime : 0,
      Math.max(Number.isFinite(minTime) ? minTime + 1e-6 : 1, maxTime)
    ];
    const baseSpan = Math.max(1e-6, timeDomain[1] - timeDomain[0]);
    const padding = baseSpan * 0.05;

    const worldBounds: WorldBounds = {
      xMin: timeDomain[0] - padding,
      xMax: timeDomain[1] + padding,
      yMin: 0,
      yMax: Math.max(1, get().viewport.pixelHeight)
    };

    const viewport = {
      ...get().viewport,
      x: timeDomain[0],
      y: 0,
      width: baseSpan,
      height: Math.max(1, get().viewport.pixelHeight),
      scale: 1
    };

    set({
      loadErrors: null,
      metadata: payload.metadata,
      agents: payload.agents,
      locations: payload.locations,
      secrets: payload.secrets,
      events: payload.events,
      scenes: payload.scenes,
      beliefSnapshots: payload.belief_snapshots,
      causalIndex,
      threadIndex,
      timeDomain,
      computedTension,
      viewport,
      worldBounds,
      activeFilters: makeDefaultFilters(agentIds, timeDomain)
    });
  },

  clearLoadErrors() {
    set({ loadErrors: null });
  },

  setViewMode(mode: ViewMode) {
    const prev = get();
    if (prev.viewMode === mode) return;

    const anchorTime =
      Number.isFinite(prev.viewport.x) && Number.isFinite(prev.viewport.width)
        ? prev.viewport.x + prev.viewport.width * 0.5
        : prev.timeDomain[0];
    const anchorY =
      Number.isFinite(prev.viewport.y) && Number.isFinite(prev.viewport.height)
        ? prev.viewport.y + prev.viewport.height * 0.5
        : 0;

    set({ viewMode: mode });
    // Keep the camera centered while switching how scale maps to spans (threads vs topology).
    get().setViewportScale(prev.viewport.scale, anchorTime, anchorY);
  },

  setTensionWeights(weights: Partial<TensionWeights>) {
    const next = { ...get().tensionWeights, ...weights };
    const computedTension = recomputeTensionMap(get().events, next);
    set({ tensionWeights: next, computedTension });
  },

  applyGenrePreset(preset: GenrePresetId) {
    const next = GENRE_PRESETS[preset];
    const computedTension = recomputeTensionMap(get().events, next);
    set({ tensionWeights: next, computedTension });
  },

  resetTensionWeights() {
    const next = DEFAULT_TENSION_WEIGHTS;
    const computedTension = recomputeTensionMap(get().events, next);
    set({ tensionWeights: next, computedTension });
  },

  setHoveredEvent(id: string | null) {
    set({ hoveredEventId: id });
  },

  selectEvent(id: string | null) {
    set({ selectedEventId: id });
  },

  selectArc(agentId: string | null) {
    set({ selectedArcAgentId: agentId });
  },

  setZoom(level: ZoomLevel) {
    set({ zoomLevel: level });
  },

  setViewport(viewport: Partial<ViewportState>) {
    const prevState = get();
    const mode = prevState.viewMode;

    const next = { ...prevState.viewport, ...viewport };
    const [minT, maxT] = prevState.timeDomain;

    let worldBounds = prevState.worldBounds;
    // Keep world bounds' vertical extent in sync with the current canvas height.
    if (Number.isFinite(next.pixelHeight) && next.pixelHeight > 0) {
      if (!worldBounds) {
        const baseSpan = Math.max(1e-6, maxT - minT);
        const padding = baseSpan * 0.05;
        worldBounds = {
          xMin: minT - padding,
          xMax: maxT + padding,
          yMin: 0,
          yMax: next.pixelHeight
        };
      } else if (Math.abs(worldBounds.yMax - next.pixelHeight) > 1e-6) {
        worldBounds = { ...worldBounds, yMax: next.pixelHeight };
      }
    }

    const xMin = mode === 'topology' && worldBounds ? worldBounds.xMin : minT;
    const xMax = mode === 'topology' && worldBounds ? worldBounds.xMax : maxT;
    const spanX = Math.max(1e-6, xMax - xMin);

    const width = Math.max(1e-6, next.width);
    const clampedWidth = Math.min(width, spanX);
    const maxX = xMax - clampedWidth;
    const x = Math.max(xMin, Math.min(maxX, next.x));

    let y = next.y;
    let clampedHeight = Math.max(1e-6, next.height);
    if (mode === 'topology') {
      const yMin = worldBounds ? worldBounds.yMin : 0;
      const yMax = worldBounds ? worldBounds.yMax : Math.max(1, next.pixelHeight);
      const spanY = Math.max(1e-6, yMax - yMin);

      clampedHeight = Math.min(clampedHeight, spanY);
      const maxY = yMax - clampedHeight;
      y = Math.max(yMin, Math.min(maxY, next.y));
    }

    const scale = next.scale > 0 ? next.scale : prevState.viewport.scale;
    set({ viewport: { ...next, x, y, width: clampedWidth, height: clampedHeight, scale }, worldBounds });
  },

  setViewportScale(scale: number, anchorTime?: number, anchorY?: number) {
    const state = get();
    const prev = state.viewport;
    const mode = state.viewMode;
    const [minT, maxT] = state.timeDomain;

    const worldBounds = state.worldBounds;
    const xMin = mode === 'topology' && worldBounds ? worldBounds.xMin : minT;
    const xMax = mode === 'topology' && worldBounds ? worldBounds.xMax : maxT;
    const baseSpanX = Math.max(1e-6, xMax - xMin);

    const yMin = worldBounds ? worldBounds.yMin : 0;
    const yMax = worldBounds ? worldBounds.yMax : Math.max(1, prev.pixelHeight);
    const baseSpanY = Math.max(1e-6, yMax - yMin);

    const minScale = mode === 'topology' ? 0.5 : 0.1;
    const maxScale = mode === 'topology' ? 4.0 : 5.0;
    const clampedScale = Math.max(minScale, Math.min(maxScale, scale));

    const width = Math.max(1e-6, baseSpanX / clampedScale);
    const height = mode === 'topology' ? Math.max(1e-6, baseSpanY / clampedScale) : prev.height;

    const anchorT =
      anchorTime ??
      (Number.isFinite(prev.x) && Number.isFinite(prev.width) ? prev.x + prev.width * 0.5 : xMin);
    const fracX = prev.width > 0 ? (anchorT - prev.x) / prev.width : 0.5;
    const anchorFracX = Math.max(0, Math.min(1, fracX));

    const anchorY0 =
      mode === 'topology'
        ? anchorY ??
          (Number.isFinite(prev.y) && Number.isFinite(prev.height) ? prev.y + prev.height * 0.5 : yMin)
        : prev.y;
    const fracY = prev.height > 0 ? (anchorY0 - prev.y) / prev.height : 0.5;
    const anchorFracY = Math.max(0, Math.min(1, fracY));

    const maxX = xMax - width;
    const x = Math.max(xMin, Math.min(maxX, anchorT - anchorFracX * width));

    const maxY = yMax - height;
    const y = mode === 'topology' ? Math.max(yMin, Math.min(maxY, anchorY0 - anchorFracY * height)) : prev.y;

    const zoomLevel =
      clampedScale < 0.4
        ? ZoomLevel.CLOUD
        : clampedScale < 1.5
          ? ZoomLevel.THREADS
          : ZoomLevel.DETAIL;

    set({ viewport: { ...prev, x, y, width, height, scale: clampedScale }, zoomLevel });
  },

  pan(dx: number, dy: number) {
    const prev = get().viewport;
    get().setViewport({ x: prev.x + dx, y: prev.y + dy });
  },

  fitAll() {
    const prev = get().viewport;
    const [minT, maxT] = get().timeDomain;
    const baseSpan = Math.max(1e-6, maxT - minT);
    set({ viewport: { ...prev, x: minT, width: baseSpan, scale: 1 }, zoomLevel: ZoomLevel.THREADS });
  },

  fitWorld(opts) {
    const state = get();
    if (state.viewMode !== 'topology') return;
    if (!state.worldBounds) return;

    const animate = opts?.animate ?? false;
    const prev = state.viewport;
    const wb = state.worldBounds;
    const target = {
      x: wb.xMin,
      y: wb.yMin,
      width: Math.max(1e-6, wb.xMax - wb.xMin),
      height: Math.max(1e-6, wb.yMax - wb.yMin),
      scale: 1
    };

    if (!animate) {
      set({ viewport: { ...prev, ...target }, zoomLevel: ZoomLevel.THREADS });
      return;
    }

    if (fitWorldRafId != null) {
      window.cancelAnimationFrame(fitWorldRafId);
      fitWorldRafId = null;
    }

    const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

    const start = { ...prev };
    const startTime = performance.now();
    const durationMs = 300;

    const step = (now: number) => {
      const u = Math.max(0, Math.min(1, (now - startTime) / durationMs));
      const k = easeOutCubic(u);

      const next = {
        ...start,
        x: lerp(start.x, target.x, k),
        y: lerp(start.y, target.y, k),
        width: lerp(start.width, target.width, k),
        height: lerp(start.height, target.height, k),
        scale: lerp(start.scale, target.scale, k)
      };

      // Clamp at each step to keep bounds invariant.
      get().setViewport(next);

      if (u < 1) {
        fitWorldRafId = window.requestAnimationFrame(step);
      } else {
        fitWorldRafId = null;
        // Snap to the exact target at the end (avoids tiny interpolation drift).
        set({ viewport: { ...get().viewport, ...target }, zoomLevel: ZoomLevel.THREADS });
      }
    };

    fitWorldRafId = window.requestAnimationFrame(step);
  },

  toggleCharacterFilter(agentId: string) {
    const current = get().activeFilters.visibleAgents;
    const next = new Set(current);
    if (next.has(agentId)) next.delete(agentId);
    else next.add(agentId);

    set({ activeFilters: { ...get().activeFilters, visibleAgents: next } });
  },

  toggleEventTypeFilter(type: EventType) {
    const current = get().activeFilters.eventTypeFilter;
    const next = new Set(current);
    if (next.has(type)) next.delete(type);
    else next.add(type);
    set({ activeFilters: { ...get().activeFilters, eventTypeFilter: next } });
  },

  setRegionSelection(region: RegionSelection | null) {
    set({ regionSelection: region });
  },

  setAxisConfig(config: AxisConfig) {
    set({ axisConfig: config });
  }
}));

// -----------------------
// Selectors / derived hooks
// -----------------------

export const useVisibleEvents = () =>
  {
    const { events, axisConfig, viewport, activeFilters, computedTension } = useNarrativeFieldStore(
      useShallow((s) => ({
        events: s.events,
        axisConfig: s.axisConfig,
        viewport: s.viewport,
        activeFilters: s.activeFilters,
        computedTension: s.computedTension
      }))
    );

    return useMemo(() => {
      const { mapper } = axisConfig;
      const x0 = viewport.x;
      const x1 = viewport.x + viewport.width;

      return events.filter((e) => {
        if (!activeFilters.visibleAgents.has(e.source_agent)) return false;
        if (!activeFilters.eventTypeFilter.has(e.type)) return false;

        const tension = computedTension.get(e.id) ?? e.metrics.tension ?? 0;
        if (tension < activeFilters.minTension) return false;

        const x = mapper(e);
        return x >= x0 && x <= x1;
      });
    }, [events, axisConfig, viewport.x, viewport.width, activeFilters, computedTension]);
  };

export const useHighlightedEvents = () =>
  useNarrativeFieldStore((s) => (s.hoveredEventId ? s.causalIndex.get(s.hoveredEventId) ?? null : null));

export const useCrystallizedArc = () =>
  useNarrativeFieldStore((s) => (s.selectedArcAgentId ? s.threadIndex.get(s.selectedArcAgentId) ?? null : null));
