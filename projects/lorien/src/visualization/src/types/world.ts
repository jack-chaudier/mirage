export interface Location {
  id: string;
  name: string;
  privacy: number;
  capacity: number;
  adjacent: string[];
  overhear_from: string[];
  overhear_probability: number;
  description: string;
}

export interface SecretDefinition {
  id: string;
  description: string;
  truth_value: boolean;
  holder: string[];
  about: string | null;
  content_type: string;
  initial_knowers: string[];
  initial_suspecters: string[];
  dramatic_weight: number;
  reveal_consequences: string;
}

export interface WorldDefinition {
  id: string;
  name: string;
  description: string;
  sim_duration_minutes: number;
  ticks_per_minute: number;

  locations: Record<string, Location>;
  secrets: Record<string, SecretDefinition>;
  seating: Record<string, string[]> | null;

  primary_themes: string[];

  snapshot_interval: number;
  catastrophe_threshold: number;
  composure_gate: number;
  trust_repair_multiplier: number;
}

