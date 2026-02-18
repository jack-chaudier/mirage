import { describe, expect, test } from 'vitest';

// Pure helper functions from TopologyEventLayer.ts are not exported.
// We reimplement them here to document and verify their contracts.

describe('TopologyEventLayer pure function contracts', () => {
  function clamp01(v: number): number {
    return Math.max(0, Math.min(1, v));
  }

  function hash01(s: string): number {
    let h = 2166136261;
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return ((h >>> 0) % 10000) / 10000;
  }

  function hexToRgb(hex: string): [number, number, number] | null {
    const m = hex.trim().match(/^#([0-9a-fA-F]{6})$/);
    if (!m) return null;
    const n = Number.parseInt(m[1]!, 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
  }

  function boostColor(hex: string, mixToWhite = 0.18): string {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    const [r, g, b] = rgb;
    const t = clamp01(mixToWhite);
    const rr = Math.round(r + (255 - r) * t);
    const gg = Math.round(g + (255 - g) * t);
    const bb = Math.round(b + (255 - b) * t);
    return `rgb(${rr},${gg},${bb})`;
  }

  test('hash01 returns value in [0, 1) and is deterministic', () => {
    const v1 = hash01('test_event_id');
    const v2 = hash01('test_event_id');
    expect(v1).toBe(v2);
    expect(v1).toBeGreaterThanOrEqual(0);
    expect(v1).toBeLessThan(1);
  });

  test('hash01 produces different values for different inputs', () => {
    const v1 = hash01('alpha');
    const v2 = hash01('beta');
    expect(v1).not.toBe(v2);
  });

  test('clamp01 edge cases', () => {
    expect(clamp01(-100)).toBe(0);
    expect(clamp01(100)).toBe(1);
    expect(clamp01(0)).toBe(0);
    expect(clamp01(1)).toBe(1);
    expect(clamp01(0.333)).toBeCloseTo(0.333);
  });

  test('boostColor mixes toward white correctly', () => {
    // Black (#000000) boosted by 0.5 should go to rgb(128,128,128) approximately
    const result = boostColor('#000000', 0.5);
    expect(result).toBe('rgb(128,128,128)');

    // White (#ffffff) boosted should stay white
    const white = boostColor('#ffffff', 0.5);
    expect(white).toBe('rgb(255,255,255)');
  });

  test('boostColor returns original hex for invalid input', () => {
    const result = boostColor('not-a-color', 0.5);
    expect(result).toBe('not-a-color');
  });
});
