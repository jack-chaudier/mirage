import type { Event, TensionComponents, TensionWeights } from '../types';

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function computeWeightedTension(components: TensionComponents, weights: TensionWeights): number {
  const w = [
    weights.danger,
    weights.time_pressure,
    weights.goal_frustration,
    weights.relationship_volatility,
    weights.information_gap,
    weights.resource_scarcity,
    weights.moral_cost,
    weights.irony_density
  ];

  const c = [
    components.danger,
    components.time_pressure,
    components.goal_frustration,
    components.relationship_volatility,
    components.information_gap,
    components.resource_scarcity,
    components.moral_cost,
    components.irony_density
  ];

  const totalWeight = w.reduce((acc, x) => acc + x, 0);
  if (totalWeight === 0) return 0;

  const value = w.reduce((acc, wi, idx) => acc + wi * c[idx]!, 0) / totalWeight;
  return clamp01(value);
}

export function recomputeTensionMap(events: Event[], weights: TensionWeights): Map<string, number> {
  const map = new Map<string, number>();
  for (const e of events) {
    map.set(e.id, computeWeightedTension(e.metrics.tension_components, weights));
  }
  return map;
}

