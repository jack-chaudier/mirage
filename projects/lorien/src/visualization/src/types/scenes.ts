import type { AgentState } from './agents';
import type { Event } from './events';
import type { Location, SecretDefinition } from './world';

export enum SceneType {
  CATASTROPHE = 'catastrophe',
  CONFRONTATION = 'confrontation',
  REVELATION = 'revelation',
  BONDING = 'bonding',
  ESCALATION = 'escalation',
  MAINTENANCE = 'maintenance'
}

export interface Scene {
  id: string;
  event_ids: string[];
  location: string;
  participants: string[];
  time_start: number;
  time_end: number;
  tick_start: number;
  tick_end: number;

  tension_arc: number[];
  tension_peak: number;
  tension_mean: number;
  dominant_theme: string;
  scene_type: SceneType | string;
  summary: string;
}

export interface SnapshotState {
  snapshot_id: string;
  tick_id: number;
  sim_time: number;
  event_count: number;

  agents: Record<string, AgentState>;
  secrets: Record<string, SecretDefinition>;
  locations: Record<string, Location>;

  global_tension: number;
  active_scene_id: string;
  belief_matrix: Record<string, Record<string, string>>;
}

export interface EventIndices {
  /**
   * TS-only convenience index computed client-side (not emitted by the Python engine).
   * The visualization builds its own causal index in `src/visualization/src/data/causalIndex.ts`.
   */
  events: Record<string, Event>;
  agent_timeline: Record<string, string[]>;
  location_events: Record<string, string[]>;
  event_participants: Record<string, string[]>;
  secret_events: Record<string, string[]>;
  /**
   * String key encoding of an unordered agent pair, e.g. `"agent1:agent2"` (sorted lexicographically).
   * Python uses tuple keys for this concept, which do not round-trip through JSON.
   */
  pair_interactions: Record<`${string}:${string}`, string[]>;
  forward_links: Record<string, string[]>;
  // Computed client-side; not part of the `.nf-viz` wire payload.
  backward_links: Record<string, string[]>;
}
