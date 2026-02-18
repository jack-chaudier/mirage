import type { BeatType, Event, EventType } from './events';

export interface TensionComponents {
  danger: number;
  time_pressure: number;
  goal_frustration: number;
  relationship_volatility: number;
  information_gap: number;
  resource_scarcity: number;
  moral_cost: number;
  irony_density: number;
}

export interface TensionWeights {
  danger: number;
  time_pressure: number;
  goal_frustration: number;
  relationship_volatility: number;
  information_gap: number;
  resource_scarcity: number;
  moral_cost: number;
  irony_density: number;
}

export interface AxisConfig {
  mode: 'sim_time' | 'scene_index' | 'reveal_order' | 'reader_knowledge' | 'custom';
  label: string;
  mapper: (event: Event) => number;
}

export enum ZoomLevel {
  CLOUD = 'cloud',
  THREADS = 'threads',
  DETAIL = 'detail'
}

export type ViewMode = 'threads' | 'topology';

export interface ViewportState {
  x: number;
  y: number;
  width: number;
  height: number;
  pixelWidth: number;
  pixelHeight: number;
  scale: number;
}

export interface FilterState {
  visibleAgents: Set<string>;
  eventTypeFilter: Set<EventType>;
  minTension: number;
  timeRange: [number, number];
}

export interface RegionSelection {
  timeStart: number;
  timeEnd: number;
  agentIds: string[];
}

export interface RenderedEvent {
  eventId: string;
  simTime: number;
  description: string;
  eventType: EventType;
  beatType?: BeatType | null;
  x: number;
  y: number;
  radius: number;
  color: string;
  opacity: number;
  glowIntensity: number;
  glowColor: string;
  threadId: string;
  zoomVisibility: ZoomLevel;
}

export interface ThreadPath {
  agentId: string;
  agentName: string;
  color: string;
  controlPoints: [number, number][];
  thickness: number[];
}

export interface CausalIndex {
  get(
    eventId: string
  ):
    | {
        backward: Set<string>;
        forward: Set<string>;
      }
    | undefined;
}

export interface ThreadIndex {
  get(agentId: string): string[] | undefined;
  agents(): string[];
}
