import type { RenderedEvent } from '../../types';

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function hexToRgb(hex: string): [number, number, number] | null {
  const m = hex.trim().match(/^#([0-9a-fA-F]{6})$/);
  if (!m) return null;
  const n = Number.parseInt(m[1]!, 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

function saturateRgb(rgb: [number, number, number], boost0: number): [number, number, number] {
  // Saturation boost by pushing away from grayscale (cheap + fast).
  const boost = Math.max(1, boost0);
  const [r0, g0, b0] = rgb;
  const gray = 0.3 * r0 + 0.59 * g0 + 0.11 * b0;
  const r = Math.max(0, Math.min(255, gray + (r0 - gray) * boost));
  const g = Math.max(0, Math.min(255, gray + (g0 - gray) * boost));
  const b = Math.max(0, Math.min(255, gray + (b0 - gray) * boost));
  return [Math.round(r), Math.round(g), Math.round(b)];
}

function drawDiamond(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
}

function drawStar(ctx: CanvasRenderingContext2D, x: number, y: number, rOuter: number, rInner: number) {
  const spikes = 5;
  let rot = (Math.PI / 2) * 3;
  const step = Math.PI / spikes;
  ctx.beginPath();
  ctx.moveTo(x, y - rOuter);
  for (let i = 0; i < spikes; i += 1) {
    ctx.lineTo(x + Math.cos(rot) * rOuter, y + Math.sin(rot) * rOuter);
    rot += step;
    ctx.lineTo(x + Math.cos(rot) * rInner, y + Math.sin(rot) * rInner);
    rot += step;
  }
  ctx.closePath();
}

export class EventNodeLayer {
  draw(
    ctx: CanvasRenderingContext2D,
    // Keep arg shape compatible with the topology event layer signature (LayerManager calls through a union type).
    args: { renderedEvents: RenderedEvent[]; now?: number; selectedArcAgentId?: string | null }
  ) {
    const { renderedEvents } = args;
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';

    for (const e of renderedEvents) {
      const t = clamp01(e.glowIntensity);
      const satBoost = 1.0 + t * 0.55;
      const rgb = hexToRgb(e.color);
      const boosted = rgb ? saturateRgb(rgb, satBoost) : null;
      const fillColor = boosted ? `rgb(${boosted[0]},${boosted[1]},${boosted[2]})` : e.color;

      const isCatastrophe = e.eventType === 'catastrophe';
      const isConflict = e.eventType === 'conflict';
      const isReveal = e.eventType === 'reveal';

      ctx.save();
      ctx.globalAlpha = e.opacity;
      ctx.fillStyle = fillColor;
      ctx.shadowColor = e.glowColor;
      ctx.shadowBlur = Math.max(0, e.glowIntensity * 12);

      const r = Math.max(2, e.radius);

      // Shape by event type (threads view).
      if (isReveal) {
        // Hollow ring.
        ctx.shadowBlur = 0;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // White outer ring for contrast.
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.arc(e.x, e.y, r, 0, Math.PI * 2);
        ctx.stroke();

        // Inner colored ring.
        ctx.strokeStyle = fillColor;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(e.x, e.y, r, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        // Filled marker.
        if (isCatastrophe) drawStar(ctx, e.x, e.y, r, Math.max(2, r * 0.5));
        else if (isConflict) drawDiamond(ctx, e.x, e.y, r);
        else {
          ctx.beginPath();
          ctx.arc(e.x, e.y, r, 0, Math.PI * 2);
        }

        ctx.fill();

        // White border for definition.
        ctx.shadowBlur = 0;
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.restore();
  }
}
