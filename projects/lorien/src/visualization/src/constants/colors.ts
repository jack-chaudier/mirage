import { EventType } from '../types';

export const CHARACTER_COLORS: Record<string, string> = {
  thorne: '#E69F00',
  elena: '#56B4E9',
  marcus: '#009E73',
  lydia: '#F0E442',
  diana: '#0072B2',
  victor: '#D55E00'
};

// Muted defaults; highlight layer can use stronger variants.
export const EVENT_TYPE_COLORS: Record<EventType, string> = {
  [EventType.CHAT]: '#9AA0A6',
  [EventType.OBSERVE]: '#7B85B5',
  [EventType.SOCIAL_MOVE]: '#8B8B8B',
  [EventType.REVEAL]: '#C9A227',
  [EventType.CONFLICT]: '#C44536',
  [EventType.INTERNAL]: '#6B6FAE',
  [EventType.PHYSICAL]: '#7A9E9F',
  [EventType.CONFIDE]: '#3A9D9F',
  [EventType.LIE]: '#A0703B',
  [EventType.CATASTROPHE]: '#B00020'
};

export function tensionToHeatColor(tension: number): string {
  // Roughly matches fake-data-visual-spec.md scale.
  if (tension < 0.2) return 'rgba(0,0,0,0)';
  if (tension < 0.4) return 'rgba(26,42,74,0.15)'; // cool blue
  if (tension < 0.6) return 'rgba(204,136,68,0.20)'; // amber
  if (tension < 0.8) return 'rgba(204,68,34,0.25)'; // orange-red
  return 'rgba(204,34,34,0.35)'; // intense red
}

export const HIGHLIGHT_BACKWARD = 'rgba(68,136,204,0.65)'; // blue tint
export const HIGHLIGHT_FORWARD = 'rgba(204,136,68,0.65)'; // amber tint

