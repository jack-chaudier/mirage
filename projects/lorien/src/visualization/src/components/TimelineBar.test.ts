import { describe, expect, it } from 'vitest';

import { computeTimelineDomain } from './timelineDomain';

describe('TimelineBar', () => {
  it('computes domain from unsorted events', () => {
    const domain = computeTimelineDomain([
      { sim_time: 12 },
      { sim_time: 3 },
      { sim_time: 9 }
    ]);

    expect(domain.min).toBe(3);
    expect(domain.max).toBe(12);
  });
});
