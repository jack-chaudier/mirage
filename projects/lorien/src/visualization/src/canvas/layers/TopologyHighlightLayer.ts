import { HIGHLIGHT_BACKWARD, HIGHLIGHT_FORWARD } from '../../constants/colors';
import { ZoomLevel, type RenderedEvent } from '../../types';
import { AnnotationLayer, type TopologyAnnotations } from './AnnotationLayer';

function rgbaWithAlpha(rgba: string, alpha: number): string {
  // Input strings are already rgba(...,a). For simplicity, just override if it's in rgba form.
  const m = rgba.match(/^rgba\(([^,]+),([^,]+),([^,]+),([^)]+)\)$/);
  if (!m) return rgba;
  const r = m[1]!.trim();
  const g = m[2]!.trim();
  const b = m[3]!.trim();
  return `rgba(${r},${g},${b},${alpha})`;
}

function drawSoftGlow(
  ctx: CanvasRenderingContext2D,
  args: { x: number; y: number; radius: number; color: string; alpha: number }
) {
  const { x, y, radius, color, alpha } = args;
  const grd = ctx.createRadialGradient(x, y, 0, x, y, radius);
  grd.addColorStop(0, rgbaWithAlpha(color, alpha));
  grd.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grd;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawDiamondRing(
  ctx: CanvasRenderingContext2D,
  e: RenderedEvent,
  strokeStyle: string,
  width = 2
) {
  const r = Math.max(6, e.radius + 4);
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(e.x, e.y - r);
  ctx.lineTo(e.x + r, e.y);
  ctx.lineTo(e.x, e.y + r);
  ctx.lineTo(e.x - r, e.y);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

export class TopologyHighlightLayer {
  private annotationLayer = new AnnotationLayer();

  draw(
    ctx: CanvasRenderingContext2D,
    args: {
      width: number;
      height: number;
      renderedEvents: RenderedEvent[];
      hoveredEventId: string | null;
      highlighted: { backward: Set<string>; forward: Set<string> } | null;
      selectedEventId: string | null;
      selectedArcEventIds: Set<string> | null;
      timeToX: (t: number) => number;
      zoomLevel: ZoomLevel;
      viewportScale: number;
      topologyAnnotations?: TopologyAnnotations | null;
    }
  ) {
    const {
      width,
      height,
      renderedEvents,
      hoveredEventId,
      highlighted,
      selectedEventId,
      selectedArcEventIds,
      timeToX,
      zoomLevel,
      viewportScale,
      topologyAnnotations
    } = args;

    const byId = new Map(renderedEvents.map((e) => [e.eventId, e] as const));

    // Causal flashlight glow.
    if (highlighted) {
      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      for (const id of highlighted.backward) {
        const e = byId.get(id);
        if (!e) continue;
        const r = 70 + e.glowIntensity * 60;
        drawSoftGlow(ctx, { x: e.x, y: e.y, radius: r, color: HIGHLIGHT_BACKWARD, alpha: 0.22 });
      }
      for (const id of highlighted.forward) {
        const e = byId.get(id);
        if (!e) continue;
        const r = 70 + e.glowIntensity * 60;
        drawSoftGlow(ctx, { x: e.x, y: e.y, radius: r, color: HIGHLIGHT_FORWARD, alpha: 0.22 });
      }
      ctx.restore();
    }

    // Selected arc ring hints
    if (selectedArcEventIds && selectedArcEventIds.size > 0) {
      ctx.save();
      ctx.globalAlpha = 0.55;
      for (const id of selectedArcEventIds) {
        const e = byId.get(id);
        if (!e) continue;
        // Keep very subtle at low zoom.
        if (zoomLevel === ZoomLevel.THREADS && viewportScale < 1.0) continue;
        drawDiamondRing(ctx, e, 'rgba(255,255,255,0.20)', 1.25);
      }
      ctx.restore();
    }

    // Selection/hover rings
    if (selectedEventId) {
      const e = byId.get(selectedEventId);
      if (e) drawDiamondRing(ctx, e, 'rgba(255,255,255,0.85)', 2.25);
    }
    if (hoveredEventId) {
      const e = byId.get(hoveredEventId);
      if (e) drawDiamondRing(ctx, e, 'rgba(255,255,255,0.95)', 2.25);
    }

    // Annotations: peaks + convergence/divergence chevrons
    this.annotationLayer.draw(ctx, {
      width,
      height,
      timeToX,
      zoomLevel,
      viewportScale,
      annotations: topologyAnnotations ?? null
    });

    // Subtle edge fade to reduce hard clipping on glows.
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    const grd = ctx.createLinearGradient(0, 0, width, 0);
    grd.addColorStop(0, 'rgba(0,0,0,0.55)');
    grd.addColorStop(0.04, 'rgba(0,0,0,0.0)');
    grd.addColorStop(0.96, 'rgba(0,0,0,0.0)');
    grd.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, width, height);
    ctx.restore();

    // Top/bottom vignette
    ctx.save();
    const v = ctx.createLinearGradient(0, 0, 0, height);
    v.addColorStop(0, 'rgba(0,0,0,0.35)');
    v.addColorStop(0.08, 'rgba(0,0,0,0)');
    v.addColorStop(0.92, 'rgba(0,0,0,0)');
    v.addColorStop(1, 'rgba(0,0,0,0.35)');
    ctx.fillStyle = v;
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  }
}
