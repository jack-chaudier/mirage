import type { BeatType } from './events';
import type { AgentManifest, SimulationMetadata } from './payload';
import type { Scene } from './scenes';
import type { SecretDefinitionPayload } from './payload';
import type { Event } from './events';
import type { TensionWeights } from './renderer';

export type StorySelectionType = 'region' | 'arc' | 'query';

export interface StoryContextPayload {
  metadata: SimulationMetadata;
  agents: AgentManifest[];
  scenes: Scene[];
  secrets: SecretDefinitionPayload[];
}

export interface StoryExtractionRequest {
  selection_type: StorySelectionType;
  event_ids: string[];
  protagonist_agent_id?: string;
  tension_weights: TensionWeights;
  genre_preset: string;
  query_text?: string;

  // Optional: allow the renderer to send a minimal context so the backend can be stateless.
  selected_events?: Event[];
  context?: StoryContextPayload;
}

export interface ArcValidation {
  valid: boolean;
  violations: string[];
}

export interface BeatClassification {
  event_id: string;
  beat_type: BeatType;
}

export interface ArcScore {
  composite: number;
  tension_variance: number;
  peak_tension: number;
  tension_shape: number;
  significance: number;
  thematic_coherence: number;
  irony_arc: number;
  protagonist_dominance: number;
}

export interface StoryExtractionResponse {
  validation: ArcValidation;
  beats?: BeatClassification[];
  score?: ArcScore;
  beat_sheet?: unknown;
  suggestions?: string[];

  // Non-canonical extension used by the local API for Phase 4.
  prose?: string | null;
  llm?: { prose?: string | null; error?: string | null } | null;
}

