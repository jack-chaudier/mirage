import type { RenderedEvent, Scene, ViewMode, ZoomLevel } from '../../types';

type Rgba = { r: number; g: number; b: number; a: number };

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function mixRgba(a: Rgba, b: Rgba, t0: number): Rgba {
  const t = clamp01(t0);
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
    a: a.a + (b.a - a.a) * t
  };
}

function rgbaToCss(c: Rgba): string {
  return `rgba(${Math.round(c.r)},${Math.round(c.g)},${Math.round(c.b)},${c.a.toFixed(4)})`;
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

export class BackgroundLayer {
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
      selectedArcAgentId?: string | null;
      selectedArcPathPoints?: Array<[number, number]> | null;
      viewMode?: ViewMode;
      zoomLevel?: ZoomLevel;
      viewportScale?: number;
    }
  ) {
    const { width, height, scenes, timeToX, sceneAvgTension } = args;

    // Threads view: tension-responsive scene washes + subtle depth gradient.
    const cool: Rgba = { r: 59, g: 130, b: 246, a: 0.04 };
    const warm: Rgba = { r: 239, g: 68, b: 68, a: 0.06 };

    for (let i = 0; i < scenes.length; i += 1) {
      const s = scenes[i]!;
      const x0 = timeToX(s.time_start);
      const x1 = timeToX(s.time_end);
      const w = Math.max(0, x1 - x0);
      const t = clamp01(sceneAvgTension?.[i] ?? 0);
      ctx.fillStyle = rgbaToCss(mixRgba(cool, warm, t));
      ctx.fillRect(x0, 0, w, height);

      // Scene boundary line
      if (i > 0) {
        ctx.save();
        ctx.strokeStyle = 'rgba(0,0,0,0.10)';
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, height);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Top-to-bottom depth gradient (subtle).
    {
      const v = ctx.createLinearGradient(0, 0, 0, height);
      v.addColorStop(0, 'rgba(0,0,0,0.03)');
      v.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = v;
      ctx.fillRect(0, 0, width, height);
    }

    // Edge fade to reduce hard clipping (threads background reads like a panel).
    const grd = ctx.createLinearGradient(0, 0, width, 0);
    grd.addColorStop(0, 'rgba(255,255,255,0.08)');
    grd.addColorStop(0.03, 'rgba(255,255,255,0.0)');
    grd.addColorStop(0.97, 'rgba(255,255,255,0.0)');
    grd.addColorStop(1, 'rgba(255,255,255,0.08)');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, width, height);

    // Threads legend: tension encoded as thread thickness.
    {
      const legendW = 190;
      const legendH = 44;
      const x0 = width - 12 - legendW;
      const y0 = height - 12 - legendH;

      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.78)';
      ctx.strokeStyle = 'rgba(0,0,0,0.10)';
      ctx.lineWidth = 1;
      roundedRectPath(ctx, x0, y0, legendW, legendH, 10);
      ctx.fill();
      ctx.stroke();

      ctx.font = 'bold 10px system-ui, sans-serif';
      ctx.fillStyle = 'rgba(0,0,0,0.70)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('Tension → Thread Thickness', x0 + legendW / 2, y0 + 14);

      const barX = x0 + 18;
      const barY = y0 + 26;
      const barW = legendW - 36;

      // Thin-to-thick sample with a faint shadow to match thread styling.
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.strokeStyle = 'rgba(0,0,0,0.12)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(barX, barY);
      ctx.lineTo(barX + barW, barY);
      ctx.stroke();

      const thinW = 1.5;
      const thickW = 4.5;
      // Draw 2 segments to communicate the mapping, not a full chart.
      ctx.strokeStyle = 'rgba(20,20,20,0.70)';
      ctx.lineWidth = thinW;
      ctx.beginPath();
      ctx.moveTo(barX, barY);
      ctx.lineTo(barX + barW * 0.45, barY);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(20,20,20,0.90)';
      ctx.lineWidth = thickW;
      ctx.beginPath();
      ctx.moveTo(barX + barW * 0.55, barY);
      ctx.lineTo(barX + barW, barY);
      ctx.stroke();

      ctx.restore();
    }
  }
}
