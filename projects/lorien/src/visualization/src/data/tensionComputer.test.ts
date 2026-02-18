import { describe, expect, test } from 'vitest';

import { computeWeightedTension } from './tensionComputer';
import type { TensionComponents, TensionWeights } from '../types';

describe('computeWeightedTension', () => {
  test('returns weighted mean in [0,1]', () => {
    const components: TensionComponents = {
      danger: 1,
      time_pressure: 0,
      goal_frustration: 0,
      relationship_volatility: 0,
      information_gap: 0,
      resource_scarcity: 0,
      moral_cost: 0,
      irony_density: 0
    };

    const weights: TensionWeights = {
      danger: 1,
      time_pressure: 1,
      goal_frustration: 1,
      relationship_volatility: 1,
      information_gap: 1,
      resource_scarcity: 1,
      moral_cost: 1,
      irony_density: 1
    };

    expect(computeWeightedTension(components, weights)).toBeCloseTo(1 / 8);
  });

  test('returns 0 if total weight is 0', () => {
    const components: TensionComponents = {
      danger: 1,
      time_pressure: 1,
      goal_frustration: 1,
      relationship_volatility: 1,
      information_gap: 1,
      resource_scarcity: 1,
      moral_cost: 1,
      irony_density: 1
    };

    const weights: TensionWeights = {
      danger: 0,
      time_pressure: 0,
      goal_frustration: 0,
      relationship_volatility: 0,
      information_gap: 0,
      resource_scarcity: 0,
      moral_cost: 0,
      irony_density: 0
    };

    expect(computeWeightedTension(components, weights)).toBe(0);
  });
});

