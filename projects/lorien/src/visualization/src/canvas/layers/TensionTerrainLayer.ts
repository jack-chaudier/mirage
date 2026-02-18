import type { RenderedEvent, Scene, ViewMode, ZoomLevel } from '../../types';

type Rgb = [number, number, number];

// 11-stop topographic color ramp (normalized tension 0.0 → 1.0)
const TOPO_RAMP: Array<[number, string]> = [
  [0.0, '#1a0533'], // deep purple (valley floor)
  [0.1, '#0d2b5e'], // dark navy
  [0.2, '#0f4c81'], // ocean blue
  [0.3, '#1a7a5c'], // deep teal
  [0.4, '#2d8f4e'], // forest green
  [0.45, '#6ab03d'], // lime green
  [0.55, '#c4d934'], // yellow-green
  [0.65, '#e8b831'], // warm yellow
  [0.75, '#e8832a'], // orange
  [0.85, '#d44a28'], // red-orange
  [1.0, '#a81520'] // deep red (summit)
];

const DEFAULT_CELL_PX = 4;
const DEFAULT_SIGMA_FRACTION = 0.15; // influence radius as a fraction of min(width,height)
const COLOR_GAMMA = 1.45; // reserve warm colors for true peaks
const BAND_STEP = 0.05; // discrete elevation bands
const RECOMPUTE_THROTTLE_MS = 100;

const CONTOUR_LEVELS: number[] = (() => {
  const out: number[] = [];
  // 12 levels: 0.10, 0.15, ..., 0.65
  for (let i = 10; i <= 65; i += 5) out.push(i / 100);
  return out;
})();

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function smoothstep(edge0: number, edge1: number, x0: number): number {
  const x = clamp01((x0 - edge0) / Math.max(1e-9, edge1 - edge0));
  return x * x * (3 - 2 * x);
}

function hexToRgb(hex: string): Rgb | null {
  const m = hex.trim().match(/^#([0-9a-fA-F]{6})$/);
  if (!m) return null;
  const n = Number.parseInt(m[1]!, 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

function mixRgb(a: Rgb, b: Rgb, t: number): Rgb {
  const tt = clamp01(t);
  return [
    Math.round(a[0] + (b[0] - a[0]) * tt),
    Math.round(a[1] + (b[1] - a[1]) * tt),
    Math.round(a[2] + (b[2] - a[2]) * tt)
  ];
}

function quantize(v: number, step: number): number {
  if (step <= 0) return v;
  const s = Math.max(1e-9, step);
  return Math.floor(clamp01(v) / s) * s;
}

function topoRampRgb(t0: number): Rgb {
  const t = clamp01(t0);
  // Find surrounding stops
  for (let i = 0; i < TOPO_RAMP.length - 1; i += 1) {
    const [tA, cA] = TOPO_RAMP[i]!;
    const [tB, cB] = TOPO_RAMP[i + 1]!;
    if (t <= tB) {
      const a = hexToRgb(cA)!;
      const b = hexToRgb(cB)!;
      const u = (t - tA) / Math.max(1e-9, tB - tA);
      return mixRgb(a, b, u);
    }
  }
  return hexToRgb(TOPO_RAMP[TOPO_RAMP.length - 1]![1])!;
}

function wendlandC2(r0: number): number {
  // K(r) = max(0, (1-r)^4 * (4r+1)) for r in [0,1]
  const r = r0;
  if (r <= 0) return 1;
  if (r >= 1) return 0;
  const one = 1 - r;
  const one2 = one * one;
  const one4 = one2 * one2;
  return one4 * (4 * r + 1);
}

function kernelInfluenceRadiusPx(glowIntensity: number, sigmaPx: number, threshold: number): number {
  // Solve glowIntensity * K(r) ~= threshold for r in [0,1] to trim work on weak tails.
  if (glowIntensity <= threshold) return 0;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 14; i += 1) {
    const mid = (lo + hi) * 0.5;
    const contribution = glowIntensity * wendlandC2(mid);
    if (contribution > threshold) lo = mid;
    else hi = mid;
  }
  return Math.max(0.05, lo) * sigmaPx;
}

function computeDomainBounds(scenes: Scene[], events: RenderedEvent[]): { minT: number; maxT: number } {
  let minT = Number.POSITIVE_INFINITY;
  let maxT = Number.NEGATIVE_INFINITY;

  for (const s of scenes) {
    if (Number.isFinite(s.time_start)) minT = Math.min(minT, s.time_start);
    if (Number.isFinite(s.time_end)) maxT = Math.max(maxT, s.time_end);
  }

  // Fallback: infer from events.
  if (!Number.isFinite(minT) || !Number.isFinite(maxT) || maxT - minT < 1e-6) {
    minT = Number.POSITIVE_INFINITY;
    maxT = Number.NEGATIVE_INFINITY;
    for (const e of events) {
      if (!Number.isFinite(e.simTime)) continue;
      minT = Math.min(minT, e.simTime);
      maxT = Math.max(maxT, e.simTime);
    }
  }

  if (!Number.isFinite(minT)) minT = 0;
  if (!Number.isFinite(maxT) || maxT - minT < 1e-6) maxT = minT + 1;
  return { minT, maxT };
}

function inferViewportFromTimeToX(args: {
  timeToX: (t: number) => number;
  widthPx: number;
  t1: number;
  t2: number;
}): { x: number; width: number } | null {
  const { timeToX, widthPx, t1, t2 } = args;
  const dt = t2 - t1;
  if (Math.abs(dt) < 1e-9) return null;
  const x1 = timeToX(t1);
  const x2 = timeToX(t2);
  const a = (x2 - x1) / dt;
  if (!Number.isFinite(a) || Math.abs(a) < 1e-9) return null;
  const b = x1 - a * t1;
  const span = widthPx / a;
  const x0 = -b / a;
  return { x: x0, width: span };
}

function hashString32(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function hashF32(v: number): number {
  // Quantize floats to keep cache keys stable across tiny numeric noise.
  const q = Math.round(v * 1000);
  return q >>> 0;
}

function computeTerrainCacheKey(args: {
  width: number;
  height: number;
  cellPx: number;
  sigmaPx: number;
  domainMinT: number;
  domainMaxT: number;
  events: RenderedEvent[];
}): string {
  const { width, height, cellPx, sigmaPx, domainMinT, domainMaxT, events } = args;
  let h = 2166136261;

  const mix = (n: number) => {
    h ^= n;
    h = Math.imul(h, 16777619);
  };

  mix(width | 0);
  mix(height | 0);
  mix(cellPx | 0);
  mix(hashF32(sigmaPx));
  mix(hashF32(domainMinT));
  mix(hashF32(domainMaxT));
  mix(events.length | 0);

  for (const e of events) {
    mix(hashString32(e.eventId));
    mix(hashF32(e.simTime));
    mix(hashF32(e.y));
    mix(hashF32(e.glowIntensity));
  }

  return `terrain:${h >>> 0}`;
}

function createOffscreenCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c;
}

type Any2dContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
type TerrainCanvas = OffscreenCanvas | HTMLCanvasElement;
type TerrainImage = TerrainCanvas | ImageBitmap;

interface FilterableContext2D {
  filter: string;
}

function get2d(c: TerrainCanvas): Any2dContext {
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('Failed to get 2D context for terrain offscreen canvas');
  return ctx as Any2dContext;
}

function asCanvasImageSource(source: TerrainImage): CanvasImageSource {
  return source as unknown as CanvasImageSource;
}

function smoothGrid1(values: Float32Array, cols: number, rows: number): Float32Array {
  // Tiny 3x3-ish blur (separable radius=1) to reduce pixel-stair artifacts in contours/bands.
  const tmp = new Float32Array(values.length);
  for (let y = 0; y < rows; y += 1) {
    const row = y * cols;
    for (let x = 0; x < cols; x += 1) {
      const i = row + x;
      const xl = x > 0 ? x - 1 : x;
      const xr = x + 1 < cols ? x + 1 : x;
      tmp[i] = (values[row + xl]! + values[i]! + values[row + xr]!) / 3;
    }
  }
  const out = new Float32Array(values.length);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const yu = y > 0 ? y - 1 : y;
      const yd = y + 1 < rows ? y + 1 : y;
      const i = y * cols + x;
      out[i] = (tmp[yu * cols + x]! + tmp[i]! + tmp[yd * cols + x]!) / 3;
    }
  }
  return out;
}

function truncateLabel(s: string, max = 30): string {
  const clean = s.replace(/\\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  return `${clean.slice(0, Math.max(0, max - 3))}...`;
}

function measureTextWithTracking(ctx: CanvasRenderingContext2D, text: string, letterSpacingPx: number): number {
  if (letterSpacingPx <= 0) return ctx.measureText(text).width;
  let w = 0;
  for (let i = 0; i < text.length; i += 1) {
    w += ctx.measureText(text[i]!).width;
    if (i < text.length - 1) w += letterSpacingPx;
  }
  return w;
}

function drawTextWithTracking(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, letterSpacingPx: number) {
  if (letterSpacingPx <= 0) {
    ctx.fillText(text, x, y);
    return;
  }
  let cx = x;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]!;
    ctx.fillText(ch, cx, y);
    cx += ctx.measureText(ch).width + letterSpacingPx;
  }
}

function roundedRectPath(ctx: Any2dContext, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.max(0, Math.min(r, w * 0.5, h * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

type TerrainCache = {
  key: string;
  width: number;
  height: number;
  domainMinT: number;
  domainMaxT: number;
  bitmap: ImageBitmap | null;
  canvas: HTMLCanvasElement | null;
  coastGlowBitmap: ImageBitmap | null;
  coastGlowCanvas: HTMLCanvasElement | null;
};

export class TensionTerrainLayer {
  private cache: TerrainCache | null = null;
  private pending:
    | {
        key: string;
        width: number;
        height: number;
        domainMinT: number;
        domainMaxT: number;
        events: RenderedEvent[];
      }
    | null = null;
  private inFlight = false;
  private inFlightKey: string | null = null;
  private recomputeTimer: number | null = null;
  private lastComputeStartMs = 0;
  private onInvalidate: (() => void) | null = null;

  // Reused canvas for dimming everything except the selected arc "corridor".
  private focusOverlay: OffscreenCanvas | HTMLCanvasElement | null = null;
  private focusOverlayCtx: Any2dContext | null = null;
  private focusOverlayW = 0;
  private focusOverlayH = 0;

  // Cached noise tile for the "ocean" backdrop (cheap texture, no per-frame randomness).
  private oceanTile: HTMLCanvasElement | null = null;

  setInvalidateCallback(cb: (() => void) | null) {
    this.onInvalidate = cb;
  }

  dispose() {
    this.onInvalidate = null;
    if (this.recomputeTimer != null) {
      clearTimeout(this.recomputeTimer);
      this.recomputeTimer = null;
    }
    if (this.cache?.bitmap) {
      try {
        this.cache.bitmap.close();
      } catch {
        // ignore
      }
    }
    if (this.cache?.coastGlowBitmap) {
      try {
        this.cache.coastGlowBitmap.close();
      } catch {
        // ignore
      }
    }
    this.cache = null;
    this.pending = null;
    this.inFlight = false;
    this.inFlightKey = null;
    this.focusOverlay = null;
    this.focusOverlayCtx = null;
    this.focusOverlayW = 0;
    this.focusOverlayH = 0;
  }

  private requestRecompute(params: NonNullable<TensionTerrainLayer['pending']>) {
    this.pending = params;

    if (this.inFlight) return;
    if (this.recomputeTimer != null) return;

    const now = performance.now();
    const since = now - this.lastComputeStartMs;
    const delay = Math.max(0, RECOMPUTE_THROTTLE_MS - since);

    this.recomputeTimer = window.setTimeout(() => {
      this.recomputeTimer = null;
      void this.runRecompute();
    }, delay);
  }

  private async runRecompute() {
    if (this.inFlight) return;
    const params = this.pending;
    if (!params) return;

    this.pending = null;
    this.inFlight = true;
    this.inFlightKey = params.key;
    this.lastComputeStartMs = performance.now();

    try {
      const next = await this.computeTerrainBitmap(params);
      // If something else was requested while we were working, keep the new cache
      // but also schedule the next one (trailing update).
      if (this.cache?.bitmap) {
        try {
          this.cache.bitmap.close();
        } catch {
          // ignore
        }
      }
      if (this.cache?.coastGlowBitmap) {
        try {
          this.cache.coastGlowBitmap.close();
        } catch {
          // ignore
        }
      }
      this.cache = next;
    } finally {
      this.inFlight = false;
      this.inFlightKey = null;
    }

    if (this.onInvalidate) this.onInvalidate();

    if (this.pending && !this.inFlight) this.requestRecompute(this.pending);
  }

  private async computeTerrainBitmap(params: NonNullable<TensionTerrainLayer['pending']>): Promise<TerrainCache> {
    const { key, width, height, domainMinT, domainMaxT, events } = params;
    const domainSpan = Math.max(1e-6, domainMaxT - domainMinT);

    const cellPx = DEFAULT_CELL_PX;
    const cols = Math.max(2, Math.ceil(width / cellPx) + 1);
    const rows = Math.max(2, Math.ceil(height / cellPx) + 1);

    const sigmaPx = Math.max(16, Math.min(width, height) * DEFAULT_SIGMA_FRACTION);
    const sigma2 = sigmaPx * sigmaPx;
    const contributionThreshold = 0.001;

    // Scalar field samples at grid nodes (x=i*cellPx, y=j*cellPx).
    // Complexity is O(E * G_local): per event, only nearby grid cells are visited.
    // We trim the effective radius using `contributionThreshold` to keep cost stable
    // as event count grows toward 200+.
    const field = new Float32Array(cols * rows);

    for (const e of events) {
      const t = e.glowIntensity;
      if (!Number.isFinite(t) || t <= 0.0001) continue;
      const ex = ((e.simTime - domainMinT) / domainSpan) * width;
      const ey = e.y;
      if (!Number.isFinite(ex) || !Number.isFinite(ey)) continue;

      const effectiveRadius = Math.min(sigmaPx, kernelInfluenceRadiusPx(t, sigmaPx, contributionThreshold));
      if (effectiveRadius <= 0) continue;
      const effectiveRadius2 = effectiveRadius * effectiveRadius;

      const gx0 = Math.max(0, Math.floor((ex - effectiveRadius) / cellPx));
      const gx1 = Math.min(cols - 1, Math.ceil((ex + effectiveRadius) / cellPx));
      const gy0 = Math.max(0, Math.floor((ey - effectiveRadius) / cellPx));
      const gy1 = Math.min(rows - 1, Math.ceil((ey + effectiveRadius) / cellPx));

      for (let gy = gy0; gy <= gy1; gy += 1) {
        const py = gy * cellPx;
        const dy = py - ey;
        for (let gx = gx0; gx <= gx1; gx += 1) {
          const px = gx * cellPx;
          const dx = px - ex;
          const d2 = dx * dx + dy * dy;
          if (d2 >= sigma2 || d2 >= effectiveRadius2) continue;
          const r = Math.sqrt(d2) / sigmaPx;
          const contribution = t * wendlandC2(r);
          if (contribution < contributionThreshold) continue;
          field[gy * cols + gx] += contribution;
        }
      }
    }

    let minV = Number.POSITIVE_INFINITY;
    let maxV = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < field.length; i += 1) {
      const v = field[i]!;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    if (!Number.isFinite(minV) || !Number.isFinite(maxV) || maxV - minV < 1e-9) {
      minV = 0;
      maxV = 1;
    }

    const invRange = 1 / Math.max(1e-9, maxV - minV);
    const norm = new Float32Array(field.length);
    for (let i = 0; i < field.length; i += 1) norm[i] = (field[i]! - minV) * invRange;

    const smooth = smoothGrid1(norm, cols, rows);

    // --- Render: bands + contour lines ---
    const off = createOffscreenCanvas(width, height);
    const offCtx = get2d(off);

    // Transparent base: the topology view draws an "ocean" behind the island bitmap.
    offCtx.save();
    offCtx.setTransform(1, 0, 0, 1, 0, 0);
    offCtx.clearRect(0, 0, width, height);
    offCtx.restore();

    // Color bands at cell resolution.
    const cellsW = cols - 1;
    const cellsH = rows - 1;
    const band = createOffscreenCanvas(cellsW, cellsH);
    const bandCtx = get2d(band);
    const img = bandCtx.createImageData(cellsW, cellsH);
    const dst = img.data;

    const idx = (x: number, y: number) => y * cols + x;
    let p = 0;
    for (let y = 0; y < cellsH; y += 1) {
      for (let x = 0; x < cellsW; x += 1) {
        const v =
          (smooth[idx(x, y)]! +
            smooth[idx(x + 1, y)]! +
            smooth[idx(x + 1, y + 1)]! +
            smooth[idx(x, y + 1)]!) *
          0.25;
        const banded = quantize(v, BAND_STEP);
        const tColor = Math.pow(clamp01(banded), COLOR_GAMMA);
        const [r, g, b] = topoRampRgb(tColor);
        // Organic coastline: fade to transparent when the field approaches zero.
        const a = smoothstep(0.02, 0.08, v);
        dst[p] = r;
        dst[p + 1] = g;
        dst[p + 2] = b;
        dst[p + 3] = Math.round(a * 255);
        p += 4;
      }
    }

    bandCtx.putImageData(img, 0, 0);

    offCtx.save();
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = 'high';
    offCtx.globalAlpha = 1.0;
    offCtx.drawImage(asCanvasImageSource(band), 0, 0, cellsW, cellsH, 0, 0, width, height);
    offCtx.restore();

    // Marching squares contours (minor + major styling) + contour labels + summit markers.
    type Segment = { ax: number; ay: number; bx: number; by: number };

    const contour = createOffscreenCanvas(width, height);
    const contourCtx = get2d(contour);
    contourCtx.save();
    contourCtx.setTransform(1, 0, 0, 1, 0, 0);
    contourCtx.clearRect(0, 0, width, height);
    contourCtx.restore();

    contourCtx.save();
    contourCtx.lineJoin = 'round';
    contourCtx.lineCap = 'round';

    const interp = (level: number, a: number, b: number) => {
      const d = b - a;
      if (Math.abs(d) < 1e-9) return 0.5;
      return (level - a) / d;
    };

    const traceLevel = (level: number, emit: (ax: number, ay: number, bx: number, by: number) => void) => {
      for (let y = 0; y < rows - 1; y += 1) {
        const y0 = y * cellPx;
        const y1 = (y + 1) * cellPx;
        for (let x = 0; x < cols - 1; x += 1) {
          const x0 = x * cellPx;
          const x1 = (x + 1) * cellPx;

          const v0 = smooth[idx(x, y)]!;
          const v1 = smooth[idx(x + 1, y)]!;
          const v2 = smooth[idx(x + 1, y + 1)]!;
          const v3 = smooth[idx(x, y + 1)]!;

          const i0 = v0 >= level;
          const i1 = v1 >= level;
          const i2 = v2 >= level;
          const i3 = v3 >= level;

          const crossesTop = i0 !== i1;
          const crossesRight = i1 !== i2;
          const crossesBottom = i3 !== i2;
          const crossesLeft = i0 !== i3;

          const pts: Array<{ edge: 'top' | 'right' | 'bottom' | 'left'; x: number; y: number }> = [];

          if (crossesTop) {
            const t = interp(level, v0, v1);
            pts.push({ edge: 'top', x: x0 + t * (x1 - x0), y: y0 });
          }
          if (crossesRight) {
            const t = interp(level, v1, v2);
            pts.push({ edge: 'right', x: x1, y: y0 + t * (y1 - y0) });
          }
          if (crossesBottom) {
            const t = interp(level, v3, v2);
            pts.push({ edge: 'bottom', x: x0 + t * (x1 - x0), y: y1 });
          }
          if (crossesLeft) {
            const t = interp(level, v0, v3);
            pts.push({ edge: 'left', x: x0, y: y0 + t * (y1 - y0) });
          }

          if (pts.length === 2) {
            emit(pts[0]!.x, pts[0]!.y, pts[1]!.x, pts[1]!.y);
          } else if (pts.length === 4) {
            const center = (v0 + v1 + v2 + v3) * 0.25;
            const diagA = i0 && i2 && !i1 && !i3;
            const diagB = i1 && i3 && !i0 && !i2;

            const byEdge = new Map(pts.map((pt) => [pt.edge, pt] as const));
            const top = byEdge.get('top')!;
            const right = byEdge.get('right')!;
            const bottom = byEdge.get('bottom')!;
            const left = byEdge.get('left')!;

            const connectTopRight = () => {
              emit(top.x, top.y, right.x, right.y);
              emit(bottom.x, bottom.y, left.x, left.y);
            };
            const connectTopLeft = () => {
              emit(top.x, top.y, left.x, left.y);
              emit(right.x, right.y, bottom.x, bottom.y);
            };

            if (diagA) {
              if (center >= level) connectTopRight();
              else connectTopLeft();
            } else if (diagB) {
              if (center >= level) connectTopLeft();
              else connectTopRight();
            } else {
              connectTopRight();
            }
          }
        }
      }
    };

    const strokeSegments = (segs: Segment[], strokeStyle: string, lineWidth: number) => {
      contourCtx.save();
      contourCtx.strokeStyle = strokeStyle;
      contourCtx.lineWidth = lineWidth;
      contourCtx.beginPath();
      for (const s of segs) {
        contourCtx.moveTo(s.ax, s.ay);
        contourCtx.lineTo(s.bx, s.by);
      }
      contourCtx.stroke();
      contourCtx.restore();
    };

    // Minor contours first.
    for (const level of CONTOUR_LEVELS) {
      const major = Math.round(level * 100) % 10 === 0;
      if (major) continue;
      contourCtx.save();
      contourCtx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
      contourCtx.lineWidth = 0.6;
      contourCtx.beginPath();
      traceLevel(level, (ax, ay, bx, by) => {
        contourCtx.moveTo(ax, ay);
        contourCtx.lineTo(bx, by);
      });
      contourCtx.stroke();
      contourCtx.restore();
    }

    // Major contours (collected for labels).
    const majorSegmentsByLevel = new Map<number, Segment[]>();
    for (const level of CONTOUR_LEVELS) {
      const major = Math.round(level * 100) % 10 === 0;
      if (!major) continue;

      const segs: Segment[] = [];
      traceLevel(level, (ax, ay, bx, by) => segs.push({ ax, ay, bx, by }));
      majorSegmentsByLevel.set(level, segs);

      // Very subtle white shadow behind major lines.
      strokeSegments(segs, 'rgba(255, 255, 255, 0.08)', 2.5);
      // Dark major line on top.
      strokeSegments(segs, 'rgba(0, 0, 0, 0.55)', 1.2);
    }

    // --- Major contour value labels ---
    type Label = {
      level: number;
      x: number;
      y: number;
      angle: number;
      text: string;
      w: number;
      h: number;
      aabb: { x0: number; y0: number; x1: number; y1: number };
    };

    const endpointKey = (x: number, y: number) => `${Math.round(x * 2)}/${Math.round(y * 2)}`;
    const buildPolylines = (segs: Segment[]): Array<Array<{ x: number; y: number }>> => {
      const adj = new Map<string, Array<{ idx: number; end: 'a' | 'b' }>>();
      for (let i = 0; i < segs.length; i += 1) {
        const s = segs[i]!;
        const ka = endpointKey(s.ax, s.ay);
        const kb = endpointKey(s.bx, s.by);
        if (!adj.has(ka)) adj.set(ka, []);
        if (!adj.has(kb)) adj.set(kb, []);
        adj.get(ka)!.push({ idx: i, end: 'a' });
        adj.get(kb)!.push({ idx: i, end: 'b' });
      }

      const used = new Array<boolean>(segs.length).fill(false);
      const lines: Array<Array<{ x: number; y: number }>> = [];

      const nextUnusedAt = (k: string): { idx: number; end: 'a' | 'b' } | null => {
        const opts = adj.get(k);
        if (!opts) return null;
        for (const o of opts) {
          if (!used[o.idx]) return o;
        }
        return null;
      };

      for (let i = 0; i < segs.length; i += 1) {
        if (used[i]) continue;
        used[i] = true;
        const s = segs[i]!;
        const line: Array<{ x: number; y: number }> = [
          { x: s.ax, y: s.ay },
          { x: s.bx, y: s.by }
        ];

        // Extend forward.
        for (let keepGoing = true; keepGoing; ) {
          const end = line[line.length - 1]!;
          const k = endpointKey(end.x, end.y);
          const next = nextUnusedAt(k);
          if (!next) {
            keepGoing = false;
            continue;
          }
          used[next.idx] = true;
          const ns = segs[next.idx]!;
          if (next.end === 'a') line.push({ x: ns.bx, y: ns.by });
          else line.push({ x: ns.ax, y: ns.ay });
        }

        // Extend backward.
        const backward: Array<{ x: number; y: number }> = [];
        for (let keepGoing = true; keepGoing; ) {
          const start = line[0]!;
          const k = endpointKey(start.x, start.y);
          const next = nextUnusedAt(k);
          if (!next) {
            keepGoing = false;
            continue;
          }
          used[next.idx] = true;
          const ns = segs[next.idx]!;
          if (next.end === 'a') backward.push({ x: ns.bx, y: ns.by });
          else backward.push({ x: ns.ax, y: ns.ay });
        }
        if (backward.length > 0) {
          backward.reverse();
          line.splice(0, 0, ...backward);
        }

        lines.push(line);
      }

      return lines;
    };

    const labels: Label[] = [];
    contourCtx.save();
    contourCtx.font = '600 9px system-ui, sans-serif';
    contourCtx.textAlign = 'center';
    contourCtx.textBaseline = 'middle';

    const padX = 2;
    const padY = 1;
    const fontH = 9;
    const rectH = fontH + padY * 2;

    for (const [level, segs] of majorSegmentsByLevel) {
      if (segs.length === 0) continue;
      const polylines = buildPolylines(segs);
      if (polylines.length === 0) continue;

      // Longest polyline by arc length.
      let best: Array<{ x: number; y: number }> | null = null;
      let bestLen = 0;
      for (const line of polylines) {
        let len = 0;
        for (let i = 0; i < line.length - 1; i += 1) {
          const a = line[i]!;
          const b = line[i + 1]!;
          len += Math.hypot(b.x - a.x, b.y - a.y);
        }
        if (len > bestLen) {
          bestLen = len;
          best = line;
        }
      }

      if (!best || bestLen < 80) continue;

      const target = bestLen * 0.35;
      let acc = 0;
      let lx = best[0]!.x;
      let ly = best[0]!.y;
      let angle = 0;

      for (let i = 0; i < best.length - 1; i += 1) {
        const a = best[i]!;
        const b = best[i + 1]!;
        const d = Math.hypot(b.x - a.x, b.y - a.y);
        if (acc + d >= target) {
          const u = d > 1e-9 ? (target - acc) / d : 0.5;
          lx = a.x + (b.x - a.x) * u;
          ly = a.y + (b.y - a.y) * u;
          angle = Math.atan2(b.y - a.y, b.x - a.x);
          break;
        }
        acc += d;
      }

      if (angle > Math.PI * 0.5 || angle < -Math.PI * 0.5) angle += Math.PI;

      const text = level.toFixed(1);
      const tw = contourCtx.measureText(text).width;
      const w = tw + padX * 2;
      const h = rectH;

      const hw = w * 0.5;
      const hh = h * 0.5;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      const ex = Math.abs(hw * c) + Math.abs(hh * s);
      const ey = Math.abs(hw * s) + Math.abs(hh * c);

      labels.push({
        level,
        x: lx,
        y: ly,
        angle,
        text,
        w,
        h,
        aabb: { x0: lx - ex, y0: ly - ey, x1: lx + ex, y1: ly + ey }
      });
    }

    // Filter overlapping labels, keeping higher levels (higher tension) first.
    labels.sort((a, b) => b.level - a.level);
    const placed: Label[] = [];
    const overlapsAabb = (a: Label, b: Label) =>
      a.aabb.x0 < b.aabb.x1 && a.aabb.x1 > b.aabb.x0 && a.aabb.y0 < b.aabb.y1 && a.aabb.y1 > b.aabb.y0;

    for (const l of labels) {
      // Skip labels that would be clipped hard by edges.
      if (l.aabb.x0 < 4 || l.aabb.y0 < 4 || l.aabb.x1 > width - 4 || l.aabb.y1 > height - 4) continue;

      let ok = true;
      for (const p0 of placed) {
        if (overlapsAabb(l, p0)) {
          ok = false;
          break;
        }
      }
      if (ok) placed.push(l);
    }

    // Render: cut a small gap, then background + text.
    for (const l of placed) {
      contourCtx.save();
      contourCtx.translate(l.x, l.y);
      contourCtx.rotate(l.angle);

      const rx = -l.w * 0.5;
      const ry = -l.h * 0.5;

      // Clear a gap in the contour stroke under the label.
      contourCtx.globalCompositeOperation = 'destination-out';
      contourCtx.fillStyle = 'rgba(0,0,0,1)';
      roundedRectPath(contourCtx, rx - 1, ry - 1, l.w + 2, l.h + 2, 2);
      contourCtx.fill();

      // Background + text.
      contourCtx.globalCompositeOperation = 'source-over';
      contourCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      roundedRectPath(contourCtx, rx, ry, l.w, l.h, 2);
      contourCtx.fill();

      contourCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      contourCtx.fillText(l.text, 0, 0);
      contourCtx.restore();
    }
    contourCtx.restore();

    // --- Summit markers ---
    type Peak = { x: number; y: number; t: number };
    const peaks: Peak[] = [];
    let global: Peak | null = null;
    for (const e of events) {
      const t = e.glowIntensity;
      if (!Number.isFinite(t)) continue;
      const ex = ((e.simTime - domainMinT) / domainSpan) * width;
      const ey = e.y;
      if (!Number.isFinite(ex) || !Number.isFinite(ey)) continue;
      if (!global || t > global.t) global = { x: ex, y: ey, t };
    }
    if (global) peaks.push(global);

    const localCandidates: Peak[] = [];
    for (const e of events) {
      const t = e.glowIntensity;
      if (!Number.isFinite(t) || t < 0.5) continue;
      const ex = ((e.simTime - domainMinT) / domainSpan) * width;
      const ey = e.y;
      if (!Number.isFinite(ex) || !Number.isFinite(ey)) continue;
      localCandidates.push({ x: ex, y: ey, t });
    }
    localCandidates.sort((a, b) => b.t - a.t);

    const minDist = 40;
    for (const c of localCandidates) {
      if (peaks.length >= 18) break;
      let ok = true;
      for (const p0 of peaks) {
        if (Math.hypot(c.x - p0.x, c.y - p0.y) < minDist) {
          ok = false;
          break;
        }
      }
      if (ok) peaks.push(c);
    }

    if (peaks.length > 0) {
      contourCtx.save();
      contourCtx.font = 'bold 10px system-ui, sans-serif';
      contourCtx.textAlign = 'left';
      contourCtx.textBaseline = 'middle';
      contourCtx.lineJoin = 'round';
      contourCtx.strokeStyle = 'rgba(0,0,0,0.85)';
      contourCtx.lineWidth = 3;
      contourCtx.fillStyle = 'rgba(255,255,255,0.92)';

      for (const p0 of peaks) {
        const tx = Math.max(4, Math.min(width - 40, p0.x + 6));
        const ty = Math.max(10, Math.min(height - 10, p0.y - 8));
        const label = `▲ ${p0.t.toFixed(2)}`;
        contourCtx.strokeText(label, tx, ty);
        contourCtx.fillText(label, tx, ty);
      }

      contourCtx.restore();
    }

    contourCtx.restore();

    // Composite contours/labels/peaks over the banded terrain.
    offCtx.save();
    offCtx.drawImage(asCanvasImageSource(contour), 0, 0);
    offCtx.restore();

    // Mask final output by the band alpha so contour lines also fade out into the ocean.
    offCtx.save();
    offCtx.globalCompositeOperation = 'destination-in';
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = 'high';
    offCtx.drawImage(asCanvasImageSource(band), 0, 0, cellsW, cellsH, 0, 0, width, height);
    offCtx.restore();

    // Coastline glow: a blurred alpha ring outside the island, tinted like shallow water.
    const glow = createOffscreenCanvas(width, height);
    const glowCtx = get2d(glow);
    glowCtx.save();
    glowCtx.setTransform(1, 0, 0, 1, 0, 0);
    glowCtx.clearRect(0, 0, width, height);
    // Blur the island mask outward.
    if ('filter' in glowCtx) {
      (glowCtx as Any2dContext & FilterableContext2D).filter = 'blur(12px)';
    }
    glowCtx.drawImage(asCanvasImageSource(off), 0, 0);
    if ('filter' in glowCtx) {
      (glowCtx as Any2dContext & FilterableContext2D).filter = 'none';
    }
    // Tint the blurred mask.
    glowCtx.globalCompositeOperation = 'source-in';
    glowCtx.fillStyle = 'rgba(120,190,255,0.26)';
    glowCtx.fillRect(0, 0, width, height);
    // Remove the interior so only the outside glow remains.
    glowCtx.globalCompositeOperation = 'destination-out';
    glowCtx.drawImage(asCanvasImageSource(off), 0, 0);
    glowCtx.restore();

    // Cache as ImageBitmap when possible (fast blit), otherwise fall back to a canvas.
    let bitmap: ImageBitmap | null = null;
    let canvas: HTMLCanvasElement | null = null;
    if (typeof createImageBitmap === 'function') {
      try {
        bitmap = await createImageBitmap(off);
      } catch {
        bitmap = null;
      }
    }
    if (!bitmap) {
      // Best-effort fallback for environments without (working) createImageBitmap.
      const c = document.createElement('canvas');
      c.width = width;
      c.height = height;
      const cctx = c.getContext('2d');
      if (cctx) cctx.drawImage(asCanvasImageSource(off), 0, 0);
      canvas = c;
    }

    let coastGlowBitmap: ImageBitmap | null = null;
    let coastGlowCanvas: HTMLCanvasElement | null = null;
    if (typeof createImageBitmap === 'function') {
      try {
        coastGlowBitmap = await createImageBitmap(glow);
      } catch {
        coastGlowBitmap = null;
      }
    }
    if (!coastGlowBitmap) {
      const c = document.createElement('canvas');
      c.width = width;
      c.height = height;
      const cctx = c.getContext('2d');
      if (cctx) cctx.drawImage(asCanvasImageSource(glow), 0, 0);
      coastGlowCanvas = c;
    }

    return { key, width, height, domainMinT, domainMaxT, bitmap, canvas, coastGlowBitmap, coastGlowCanvas };
  }

  draw(
    ctx: CanvasRenderingContext2D,
    args: {
      width: number;
      height: number;
      scenes: Scene[];
      sceneAvgTension?: number[];
      renderedEvents: RenderedEvent[];
      fieldEvents?: RenderedEvent[];
      timeToX: (t: number) => number;
      viewportY?: number;
      viewportHeight?: number;
      selectedArcAgentId: string | null;
      selectedArcPathPoints?: Array<[number, number]> | null;
      viewMode: ViewMode;
      zoomLevel: ZoomLevel;
      viewportScale: number;
    }
  ) {
    const {
      width,
      height,
      scenes,
      renderedEvents,
      fieldEvents,
      timeToX,
      viewportScale,
      selectedArcPathPoints,
      viewportY,
      viewportHeight
    } = args;

    const eventsForField = fieldEvents ?? renderedEvents;
    const { minT: domainMinT, maxT: domainMaxT } = computeDomainBounds(scenes, eventsForField);
    const sigmaPx = Math.max(16, Math.min(width, height) * DEFAULT_SIGMA_FRACTION);

    const desiredKey = computeTerrainCacheKey({
      width,
      height,
      cellPx: DEFAULT_CELL_PX,
      sigmaPx,
      domainMinT,
      domainMaxT,
      events: eventsForField
    });

    if (this.cache?.key !== desiredKey && this.inFlightKey !== desiredKey) {
      this.requestRecompute({
        key: desiredKey,
        width,
        height,
        domainMinT,
        domainMaxT,
        events: eventsForField
      });
    }

    // Base background: the narrative ocean.
    ctx.save();
    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, width, height);

    if (!this.oceanTile) {
      const tile = document.createElement('canvas');
      tile.width = 128;
      tile.height = 128;
      const tctx = tile.getContext('2d');
      if (tctx) {
        tctx.clearRect(0, 0, tile.width, tile.height);
        // Deterministic LCG noise so the ocean texture doesn't shimmer.
        let seed = 1337;
        const rnd = () => {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          return seed / 4294967296;
        };
        for (let i = 0; i < 1400; i += 1) {
          const x = Math.floor(rnd() * tile.width);
          const y = Math.floor(rnd() * tile.height);
          const a = 0.02 + rnd() * 0.06;
          const c = 180 + Math.floor(rnd() * 50);
          tctx.fillStyle = `rgba(${c},${c},${c},${a.toFixed(4)})`;
          tctx.fillRect(x, y, 1, 1);
        }
      }
      this.oceanTile = tile;
    }

    const pat = ctx.createPattern(this.oceanTile, 'repeat');
    if (pat) {
      ctx.save();
      ctx.globalAlpha = 0.10;
      ctx.fillStyle = pat;
      ctx.fillRect(0, 0, width, height);
      ctx.restore();
    }

    // Terrain bitmap blit (cropped/scaled to the current viewport).
    const cache = this.cache;
    if (cache && (cache.bitmap || cache.canvas)) {
      const sameDomain =
        Math.abs(cache.domainMinT - domainMinT) < 1e-6 && Math.abs(cache.domainMaxT - domainMaxT) < 1e-6;
      if (sameDomain) {
        // While a new cache is computing (e.g. during slider drags), keep showing the last bitmap.
        const viewport =
          inferViewportFromTimeToX({ timeToX, widthPx: width, t1: cache.domainMinT, t2: cache.domainMaxT }) ?? null;
        const cacheSpan = Math.max(1e-6, cache.domainMaxT - cache.domainMinT);
        const vpX = viewport?.x ?? cache.domainMinT;
        const vpW = viewport?.width ?? cacheSpan;

        const srcW0 = (vpW / cacheSpan) * cache.width;
        const srcX0 = ((vpX - cache.domainMinT) / cacheSpan) * cache.width;

        const img = (cache.bitmap ?? cache.canvas)!;
        const imgW = cache.width;
        const imgH = cache.height;

        let sx = srcX0;
        let sw = srcW0;
        let dx = 0;
        let dw = width;

        const vpY0 = Number.isFinite(viewportY ?? NaN) ? (viewportY as number) : 0;
        const vpH0 = Number.isFinite(viewportHeight ?? NaN) ? (viewportHeight as number) : height;
        const srcH0 = (vpH0 / Math.max(1e-6, height)) * imgH;
        const srcY0 = (vpY0 / Math.max(1e-6, height)) * imgH;

        let sy = srcY0;
        let sh = srcH0;
        let dy = 0;
        let dh = height;

        // Clamp if viewport extends beyond the cached domain image.
        if (sw > 0 && sx < 0) {
          const frac = Math.min(1, (-sx) / sw);
          dx += frac * width;
          dw -= frac * width;
          sw += sx;
          sx = 0;
        }
        if (sw > 0 && sx + sw > imgW) {
          const overflow = sx + sw - imgW;
          const frac = Math.min(1, overflow / sw);
          dw -= frac * dw;
          sw -= overflow;
        }

        if (sh > 0 && sy < 0) {
          const frac = Math.min(1, (-sy) / sh);
          dy += frac * height;
          dh -= frac * height;
          sh += sy;
          sy = 0;
        }
        if (sh > 0 && sy + sh > imgH) {
          const overflow = sy + sh - imgH;
          const frac = Math.min(1, overflow / sh);
          dh -= frac * dh;
          sh -= overflow;
        }

        if (sw > 1 && dw > 1 && sh > 1 && dh > 1) {
          ctx.save();
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(asCanvasImageSource(img), sx, sy, sw, sh, dx, dy, dw, dh);

          // Shallow-water boundary glow.
          const glowImg = cache.coastGlowBitmap ?? cache.coastGlowCanvas;
          if (glowImg) {
            ctx.globalCompositeOperation = 'screen';
            ctx.globalAlpha = 0.9;
            ctx.drawImage(asCanvasImageSource(glowImg), sx, sy, sw, sh, dx, dy, dw, dh);
          }

          ctx.restore();
        }
      }
    }

    // Terrain focus dim: keep a bright corridor around the selected arc, dim everything else.
    if (selectedArcPathPoints && selectedArcPathPoints.length >= 2) {
      if (!this.focusOverlay || this.focusOverlayW !== width || this.focusOverlayH !== height) {
        this.focusOverlay = createOffscreenCanvas(width, height);
        this.focusOverlayCtx = get2d(this.focusOverlay);
        this.focusOverlayW = width;
        this.focusOverlayH = height;
      }

      const octx = this.focusOverlayCtx!;
      octx.save();
      octx.setTransform(1, 0, 0, 1, 0, 0);
      octx.clearRect(0, 0, width, height);
      octx.fillStyle = 'rgba(0,0,0,0.30)';
      octx.fillRect(0, 0, width, height);

      // Cut out the corridor along the selected arc.
      octx.globalCompositeOperation = 'destination-out';
      octx.strokeStyle = 'rgba(0,0,0,1)';
      octx.lineWidth = 120;
      octx.lineCap = 'round';
      octx.lineJoin = 'round';
      octx.beginPath();
      for (let i = 0; i < selectedArcPathPoints.length; i += 1) {
        const p0 = selectedArcPathPoints[i]!;
        const x = timeToX(p0[0]);
        const y = p0[1];
        if (i === 0) octx.moveTo(x, y);
        else octx.lineTo(x, y);
      }
      octx.stroke();
      octx.restore();

      ctx.save();
      ctx.drawImage(asCanvasImageSource(this.focusOverlay), 0, 0, width, height);
      ctx.restore();
    }

    // Subtle time gridlines for temporal reference.
    const viewport =
      inferViewportFromTimeToX({ timeToX, widthPx: width, t1: domainMinT, t2: domainMaxT }) ?? null;
    if (viewport) {
      const pxPerMin = width / Math.max(1e-6, viewport.width);
      let step = 5;
      if (step * pxPerMin < 55) step = 10;
      if (step * pxPerMin < 55) step = 15;

      const t0 = Math.floor(viewport.x / step) * step;
      const t1 = viewport.x + viewport.width;

      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.font = '8px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.30)';
      ctx.textBaseline = 'alphabetic';
      ctx.textAlign = 'center';

      for (let t = t0; t <= t1 + 1e-6; t += step) {
        const x = timeToX(t);
        if (x < -40 || x > width + 40) continue;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        const label = `${Math.round(t - t0)}m`;
        ctx.fillText(label, x, height - 6);
      }
      ctx.restore();
    }

    // Scene boundaries (vertical dashed dividers).
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 7]);
    for (let i = 0; i < scenes.length; i += 1) {
      const s = scenes[i]!;
      const x0 = timeToX(s.time_start);
      if (i > 0) {
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, height);
        ctx.stroke();
      }
    }
    ctx.restore();

    // Scene label strip (inside the canvas, not above).
    if (viewportScale >= 0.45) {
      const stripH = 18;
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.fillRect(0, 0, width, stripH);
      ctx.restore();

      ctx.save();
      ctx.font = '9px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.60)';
      ctx.textBaseline = 'middle';
      ctx.shadowColor = 'rgba(0,0,0,0.75)';
      ctx.shadowBlur = 6;

      type LabelCandidate = {
        x: number;
        w: number;
        text: string;
        priority: number;
      };

      const candidates: LabelCandidate[] = [];
      for (const s of scenes) {
        const sceneType = String(s.scene_type ?? 'scene').toLowerCase();
        const highDrama =
          sceneType === 'catastrophe' || sceneType === 'confrontation' || sceneType === 'revelation';

        const x0 = timeToX(s.time_start);
        const x1 = timeToX(s.time_end);
        const bandW = Math.max(0, x1 - x0);

        if (!highDrama && bandW < 80) continue;

        const label = truncateLabel(sceneType.toUpperCase(), 18);
        const w = measureTextWithTracking(ctx, label, 1.0);
        if (!highDrama && bandW < w + 14) continue;

        const tx = Math.max(6, Math.min(width - 6 - w, x0 + 8));
        candidates.push({ x: tx, w, text: label, priority: highDrama ? 2 : 1 });
      }

      // Drop overlapping labels (keep higher priority).
      candidates.sort((a, b) => a.x - b.x);
      const placed: LabelCandidate[] = [];
      for (const c of candidates) {
        const prev = placed[placed.length - 1];
        if (!prev) {
          placed.push(c);
          continue;
        }
        if (c.x < prev.x + prev.w + 10) {
          if (c.priority > prev.priority) placed[placed.length - 1] = c;
          continue;
        }
        placed.push(c);
      }

      for (const c of placed) drawTextWithTracking(ctx, c.text, c.x, stripH / 2, 1.0);
      ctx.restore();
    }

    // Tension legend (bottom-right).
    {
      const legendW = 170;
      const legendH = 44;
      const x0 = width - 12 - legendW;
      const y0 = height - 12 - legendH;

      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      roundedRectPath(ctx, x0, y0, legendW, legendH, 6);
      ctx.fill();

      ctx.font = 'bold 10px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('Tension', x0 + legendW / 2, y0 + 14);

      const barW = 150;
      const barH = 12;
      const barX = x0 + (legendW - barW) / 2;
      const barY = y0 + 18;
      const grd = ctx.createLinearGradient(barX, 0, barX + barW, 0);
      for (const [t, c] of TOPO_RAMP) grd.addColorStop(t, c);
      ctx.fillStyle = grd;
      roundedRectPath(ctx, barX, barY, barW, barH, 3);
      ctx.fill();

      ctx.font = '9px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('Low', barX, y0 + 42);
      ctx.textAlign = 'right';
      ctx.fillText('High', barX + barW, y0 + 42);

      ctx.restore();
    }

    ctx.restore();
  }
}
