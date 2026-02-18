import { CHARACTER_COLORS } from '../../constants/colors';
import { BeatType, EventType, type RenderedEvent } from '../../types';

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function hash01(s: string): number {
  // Fast, stable-ish hash for desynchronizing pulses.
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // 0..1
  return ((h >>> 0) % 10000) / 10000;
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

export class TopologyEventLayer {
  draw(
    ctx: CanvasRenderingContext2D,
    args: { renderedEvents: RenderedEvent[]; now: number; selectedArcAgentId?: string | null }
  ) {
    const { renderedEvents, now, selectedArcAgentId } = args;
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';

    for (const e of renderedEvents) {
      const isCatastrophe = e.eventType === EventType.CATASTROPHE;
      const isTurningPoint = e.beatType === BeatType.TURNING_POINT;
      const isSelectedArc = !!selectedArcAgentId && e.threadId === selectedArcAgentId;

      let r = 5.0;
      if (isTurningPoint) r = 7.0;
      if (isCatastrophe) r = 8.0;
      if (isSelectedArc) r *= 1.3;

      const alpha = clamp01(e.opacity);
      if (alpha <= 0.01) continue;

      const color = CHARACTER_COLORS[e.threadId] ?? e.color;
      const phase = hash01(e.eventId) * Math.PI * 2;
      const outlinePulse = isTurningPoint ? 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(now / 320 + phase)) : 0.85;

      // Dark contrast disc (keeps markers readable/clickable over terrain).
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = 'rgba(0,0,0,0.4)';
      ctx.beginPath();
      ctx.arc(e.x, e.y, r + 2.0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Main marker: saturated fill + white outline.
      ctx.save();
      ctx.shadowBlur = 0;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.0;
      ctx.strokeStyle = 'rgba(255,255,255,1)';

      // Fill
      ctx.globalAlpha = alpha;
      if (isCatastrophe) drawStar(ctx, e.x, e.y, r, Math.max(2, r * 0.5));
      else drawDiamond(ctx, e.x, e.y, r);
      ctx.fill();

      // Outline (turning-point pulses the outline opacity, not the fill).
      ctx.globalAlpha = alpha * outlinePulse;
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  }
}
