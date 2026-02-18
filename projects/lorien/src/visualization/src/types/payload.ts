import type { BeliefState } from './agents';
import type { Event } from './events';
import type { Scene } from './scenes';
import type { Location, SecretDefinition } from './world';

export interface SimulationMetadata {
  simulation_id: string;
  deterministic_id?: string;
  scenario: string;
  total_ticks: number;
  total_sim_time: number;
  agent_count: number;
  event_count: number;
  raw_event_count?: number;
  snapshot_interval: number;
  timestamp: string;
  seed?: number;
  time_scale?: number;
  truncated?: boolean;
  config_hash?: string;
  python_version?: string;
  git_commit?: string | null;
}

export interface AgentManifest {
  id: string;
  name: string;
  initial_location: string;
  goal_summary: string;
  primary_flaw: string;
}

export interface BeliefSnapshot {
  tick_id: number;
  sim_time: number;
  beliefs: Record<string, Record<string, BeliefState>>;
  agent_irony: Record<string, number>;
  scene_irony: number;
}

export type LocationDefinition = Location;

export type SecretDefinitionPayload = SecretDefinition;

export interface NarrativeFieldPayload {
  format_version: string;
  metadata: SimulationMetadata;
  agents: AgentManifest[];
  locations: LocationDefinition[];
  secrets: SecretDefinitionPayload[];
  events: Event[];
  scenes: Scene[];
  belief_snapshots: BeliefSnapshot[];
}
