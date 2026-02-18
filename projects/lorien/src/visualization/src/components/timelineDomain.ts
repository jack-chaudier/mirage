import type { Event } from '../types';

export function computeTimelineDomain(events: Pick<Event, 'sim_time'>[]): { min: number; max: number } {
  if (events.length === 0) return { min: 0, max: 1 };
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const event of events) {
    min = Math.min(min, event.sim_time);
    max = Math.max(max, event.sim_time);
  }
  if (!Number.isFinite(min)) return { min: 0, max: 1 };
  return { min, max: Math.max(min + 1e-6, max) };
}
