export enum FlawType {
  PRIDE = 'pride',
  LOYALTY = 'loyalty',
  TRAUMA = 'trauma',
  AMBITION = 'ambition',
  JEALOUSY = 'jealousy',
  COWARDICE = 'cowardice',
  VANITY = 'vanity',
  GUILT = 'guilt',
  OBSESSION = 'obsession',
  DENIAL = 'denial'
}

export interface CharacterFlaw {
  flaw_type: FlawType;
  strength: number;
  trigger: string;
  effect: string;
  description: string;
}

export interface GoalVector {
  safety: number;
  status: number;
  closeness: Record<string, number>;
  secrecy: number;
  truth_seeking: number;
  autonomy: number;
  loyalty: number;
}

export interface RelationshipState {
  trust: number;
  affection: number;
  obligation: number;
}

export interface PacingState {
  dramatic_budget: number;
  stress: number;
  composure: number;
  commitment: number;
  recovery_timer: number;
  suppression_count: number;
}

export enum BeliefState {
  UNKNOWN = 'unknown',
  SUSPECTS = 'suspects',
  BELIEVES_TRUE = 'believes_true',
  BELIEVES_FALSE = 'believes_false'
}

export interface EmotionalState {
  anger: number;
  fear: number;
  hope: number;
  shame: number;
  affection: number;
  suspicion: number;
}

export interface AgentState {
  id: string;
  name: string;
  location: string;

  goals: GoalVector;
  flaws: CharacterFlaw[];
  pacing: PacingState;
  emotional_state: Record<string, number>;

  relationships: Record<string, RelationshipState>;
  beliefs: Record<string, BeliefState>;

  alcohol_level: number;
  commitments: string[];
}

