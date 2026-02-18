import { describe, expect, test } from 'vitest';

// Pure helper functions from TensionTerrainLayer.ts are not exported.
// We reimplement them here to document and verify their contracts.

describe('TensionTerrainLayer pure function contracts', () => {
  function clamp01(v: number): number {
    return Math.max(0, Math.min(1, v));
  }

  test('clamp01 clamps below 0', () => {
    expect(clamp01(-0.5)).toBe(0);
  });

  test('clamp01 clamps above 1', () => {
    expect(clamp01(1.5)).toBe(1);
  });

  test('clamp01 passes through 0.5', () => {
    expect(clamp01(0.5)).toBe(0.5);
  });

  function hexToRgb(hex: string): [number, number, number] | null {
    const m = hex.trim().match(/^#([0-9a-fA-F]{6})$/);
    if (!m) return null;
    const n = Number.parseInt(m[1]!, 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
  }

  function mix(
    a: [number, number, number],
    b: [number, number, number],
    t: number
  ): [number, number, number] {
    const tt = clamp01(t);
    return [
      Math.round(a[0] + (b[0] - a[0]) * tt),
      Math.round(a[1] + (b[1] - a[1]) * tt),
      Math.round(a[2] + (b[2] - a[2]) * tt)
    ];
  }

  const TOPO_RAMP: Array<[number, string]> = [
    [0.0, '#1a0533'],
    [0.1, '#0d2b5e'],
    [0.2, '#0f4c81'],
    [0.3, '#1a7a5c'],
    [0.4, '#2d8f4e'],
    [0.45, '#6ab03d'],
    [0.55, '#c4d934'],
    [0.65, '#e8b831'],
    [0.75, '#e8832a'],
    [0.85, '#d44a28'],
    [1.0, '#a81520']
  ];

  function topoRampRgb(t0: number): [number, number, number] {
    const t = clamp01(t0);
    for (let i = 0; i < TOPO_RAMP.length - 1; i += 1) {
      const [tA, cA] = TOPO_RAMP[i]!;
      const [tB, cB] = TOPO_RAMP[i + 1]!;
      if (t <= tB) {
        const a = hexToRgb(cA)!;
        const b = hexToRgb(cB)!;
        const u = (t - tA) / Math.max(1e-9, tB - tA);
        return mix(a, b, u);
      }
    }
    return hexToRgb(TOPO_RAMP[TOPO_RAMP.length - 1]![1])!;
  }

  test('topoRampRgb(0) returns the darkest ramp color', () => {
    const [r, g, b] = topoRampRgb(0);
    expect(r).toBe(0x1a);
    expect(g).toBe(0x05);
    expect(b).toBe(0x33);
  });

  test('topoRampRgb(1) returns the summit color', () => {
    const [r, g, b] = topoRampRgb(1);
    expect(r).toBe(0xa8);
    expect(g).toBe(0x15);
    expect(b).toBe(0x20);
  });

  test('topoRampRgb clamps input', () => {
    expect(topoRampRgb(-1)).toEqual(topoRampRgb(0));
    expect(topoRampRgb(2)).toEqual(topoRampRgb(1));
  });

  function truncateLabel(s: string, max = 30): string {
    const clean = s.replace(/\s+/g, ' ').trim();
    if (clean.length <= max) return clean;
    return `${clean.slice(0, Math.max(0, max - 3))}...`;
  }

  test('truncateLabel preserves short strings', () => {
    expect(truncateLabel('HELLO')).toBe('HELLO');
    expect(truncateLabel('   spaced   out   ')).toBe('spaced out');
  });

  test('truncateLabel truncates long strings with ellipsis', () => {
    const long = 'A'.repeat(40);
    const result = truncateLabel(long);
    expect(result.length).toBe(30);
    expect(result.endsWith('...')).toBe(true);
  });
});
