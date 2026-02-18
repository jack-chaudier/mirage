import { ZoomLevel } from '../../types';

export type TopologyChevronKind = 'converge' | 'diverge';

export interface TopologyPeakLabel {
  x: number;
  y: number;
  text: string;
  tension: number;
}

export interface TopologyEventLabel {
  eventId: string;
  ax: number; // anchor x (event marker)
  ay: number; // anchor y (event marker)
  x: number; // label rect top-left
  y: number; // label rect top-left
  w: number;
  h: number;
  text: string;
  tier: 1 | 2 | 3;
  tension: number;
}

export interface TopologyChevron {
  t: number;
  y: number;
  kind: TopologyChevronKind;
}

export interface TopologyAnnotations {
  peaks: TopologyPeakLabel[];
  chevrons: TopologyChevron[];
  eventLabels: TopologyEventLabel[];
}

function zoomRank(z: ZoomLevel): number {
  switch (z) {
    case ZoomLevel.CLOUD:
      return 0;
    case ZoomLevel.THREADS:
      return 1;
    case ZoomLevel.DETAIL:
      return 2;
  }
}

function isVisibleAtZoom(minZoom: ZoomLevel, current: ZoomLevel): boolean {
  return zoomRank(current) >= zoomRank(minZoom);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function roundedRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.max(0, Math.min(r, w * 0.5, h * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function drawChevron(ctx: CanvasRenderingContext2D, args: { x: number; y: number; kind: TopologyChevronKind }) {
  const { x, y, kind } = args;
  const w = 10;
  const h = 6;
  ctx.beginPath();
  if (kind === 'converge') {
    // Downward V
    ctx.moveTo(x - w * 0.5, y - h * 0.5);
    ctx.lineTo(x, y + h * 0.5);
    ctx.lineTo(x + w * 0.5, y - h * 0.5);
  } else {
    // Upward ^
    ctx.moveTo(x - w * 0.5, y + h * 0.5);
    ctx.lineTo(x, y - h * 0.5);
    ctx.lineTo(x + w * 0.5, y + h * 0.5);
  }
  ctx.stroke();
}

export class AnnotationLayer {
  draw(
    ctx: CanvasRenderingContext2D,
    args: {
      width: number;
      height: number;
      timeToX: (t: number) => number;
      zoomLevel: ZoomLevel;
      viewportScale: number;
      annotations: TopologyAnnotations | null;
    }
  ) {
    const { width, height, timeToX, zoomLevel, viewportScale, annotations } = args;
    if (!annotations) return;

    // Mid-to-high zoom only.
    if (!isVisibleAtZoom(ZoomLevel.THREADS, zoomLevel) || viewportScale < 0.65) return;

    // Convergence/divergence markers
    if (annotations.chevrons.length > 0) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      for (const m of annotations.chevrons) {
        const x = timeToX(m.t);
        const y = clamp(m.y, 6, height - 6);
        if (x < -20 || x > width + 20) continue;
        drawChevron(ctx, { x, y, kind: m.kind });
      }
      ctx.restore();
    }

    // Curated event labels (precomputed placement; no per-frame collision work).
    if (annotations.eventLabels.length > 0 && viewportScale >= 0.7) {
      ctx.save();
      ctx.font = '10px system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';

      for (const l of annotations.eventLabels) {
        const x0 = l.x;
        const y0 = l.y;
        const w = l.w;
        const h = l.h;

        // Skip far-offscreen labels.
        if (x0 + w < -40 || x0 > width + 40) continue;
        if (y0 + h < -40 || y0 > height + 40) continue;

        const ax = l.ax;
        const ay = clamp(l.ay, 0, height);

        // Leader line if the label box is offset away from the anchor.
        const inside = ax >= x0 && ax <= x0 + w && ay >= y0 && ay <= y0 + h;
        if (!inside) {
          const nx = clamp(ax, x0, x0 + w);
          const ny = clamp(ay, y0, y0 + h);
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.30)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(nx, ny);
          ctx.stroke();
          ctx.restore();
        }

        // Background.
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        roundedRectPath(ctx, x0, y0, w, h, 3);
        ctx.fill();
        ctx.restore();

        // Text.
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.90)';
        const padH = 4;
        ctx.fillText(l.text, x0 + padH, y0 + h * 0.5);
        ctx.restore();
      }
      ctx.restore();
    }
  }
}
