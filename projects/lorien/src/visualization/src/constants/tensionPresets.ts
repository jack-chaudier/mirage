import type { TensionWeights } from '../types';

export type GenrePresetId = 'thriller' | 'relationship_drama' | 'mystery';

export const DEFAULT_TENSION_WEIGHTS: TensionWeights = {
  danger: 1.0,
  time_pressure: 1.0,
  goal_frustration: 1.0,
  relationship_volatility: 1.0,
  information_gap: 1.0,
  resource_scarcity: 1.0,
  moral_cost: 1.0,
  irony_density: 1.0
};

// Exact values from specs/metrics/tension-pipeline.md Section 3.3.
export const GENRE_PRESETS: Record<GenrePresetId, TensionWeights> = {
  thriller: {
    danger: 2.5,
    time_pressure: 2.0,
    goal_frustration: 1.0,
    relationship_volatility: 0.5,
    information_gap: 1.5,
    resource_scarcity: 1.5,
    moral_cost: 0.5,
    irony_density: 1.0
  },
  relationship_drama: {
    danger: 0.3,
    time_pressure: 0.5,
    goal_frustration: 1.5,
    relationship_volatility: 2.5,
    information_gap: 1.0,
    resource_scarcity: 0.5,
    moral_cost: 2.0,
    irony_density: 1.5
  },
  mystery: {
    danger: 1.0,
    time_pressure: 1.0,
    goal_frustration: 0.5,
    relationship_volatility: 0.5,
    information_gap: 2.5,
    resource_scarcity: 0.3,
    moral_cost: 1.0,
    irony_density: 2.0
  }
};

