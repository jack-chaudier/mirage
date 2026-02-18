import { describe, expect, test } from 'vitest';

// The encode/decode functions are not exported from HitCanvas.ts.
// We reimplement the algorithm here to document and verify the encoding contract.
describe('HitCanvas RGB encoding contract', () => {
  function encodeIndexToRGB(index: number): [number, number, number] {
    const id = index + 1;
    return [(id >> 16) & 0xff, (id >> 8) & 0xff, id & 0xff];
  }

  function decodeRGBToIndex(r: number, g: number, b: number): number | null {
    const id = (r << 16) | (g << 8) | b;
    if (id === 0) return null;
    return id - 1;
  }

  test('roundtrip encode/decode for index 0', () => {
    const [r, g, b] = encodeIndexToRGB(0);
    expect(decodeRGBToIndex(r, g, b)).toBe(0);
  });

  test('roundtrip for large indices', () => {
    const indices = [1, 255, 256, 65535, 65536, 100000];
    for (const idx of indices) {
      const [r, g, b] = encodeIndexToRGB(idx);
      expect(decodeRGBToIndex(r, g, b)).toBe(idx);
    }
  });

  test('decode (0,0,0) returns null (no hit)', () => {
    expect(decodeRGBToIndex(0, 0, 0)).toBeNull();
  });
});
