import type { TensionComponents } from './renderer';
import type { ThematicAxis } from './thematic';

export enum EventType {
  CHAT = 'chat',
  OBSERVE = 'observe',
  SOCIAL_MOVE = 'social_move',
  REVEAL = 'reveal',
  CONFLICT = 'conflict',
  INTERNAL = 'internal',
  PHYSICAL = 'physical',
  CONFIDE = 'confide',
  LIE = 'lie',
  CATASTROPHE = 'catastrophe'
}

export enum BeatType {
  SETUP = 'setup',
  COMPLICATION = 'complication',
  ESCALATION = 'escalation',
  TURNING_POINT = 'turning_point',
  CONSEQUENCE = 'consequence'
}

export enum DeltaKind {
  AGENT_EMOTION = 'agent_emotion',
  AGENT_RESOURCE = 'agent_resource',
  AGENT_LOCATION = 'agent_location',
  RELATIONSHIP = 'relationship',
  BELIEF = 'belief',
  SECRET_STATE = 'secret_state',
  WORLD_RESOURCE = 'world_resource',
  COMMITMENT = 'commitment',
  PACING = 'pacing',
  LOCATION_MEMORY = 'location_memory',
  ARTIFACT_STATE = 'artifact_state',
  FACTION_STATE = 'faction_state',
  INSTITUTION_STATE = 'institution_state'
}

export enum DeltaOp {
  SET = 'set',
  ADD = 'add'
}

export interface StateDelta {
  // Keep explicit support for known kinds while allowing forward-compatible pass-through.
  kind: DeltaKind | string;
  agent: string;
  agent_b?: string | null;
  attribute: string;
  op: DeltaOp;
  value: number | string | boolean;
  reason_code: string;
  reason_display: string;
}

export interface IronyCollapseInfo {
  detected: boolean;
  drop: number;
  collapsed_beliefs: Array<{ agent: string; secret: string; from: string; to: string }>;
  score: number;
}

export interface EventMetrics {
  tension: number;
  irony: number;
  significance: number;
  thematic_shift: Partial<Record<ThematicAxis, number>>;
  tension_components: TensionComponents;
  // Pre-staged for downstream analytics; currently consumed in Python segmentation,
  // not by visualization components in this MVP.
  irony_collapse: IronyCollapseInfo | null;
}

export interface Event {
  id: string;

  sim_time: number;
  tick_id: number;
  order_in_tick: number;

  type: EventType;

  source_agent: string;
  target_agents: string[];
  location_id: string;

  causal_links: string[];

  deltas: StateDelta[];

  description: string;
  dialogue?: string | null;
  content_metadata?: Record<string, unknown> | null;

  beat_type?: BeatType | null;

  metrics: EventMetrics;
}
