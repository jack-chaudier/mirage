import { HIGHLIGHT_BACKWARD, HIGHLIGHT_FORWARD } from '../../constants/colors';
import type { RenderedEvent, ZoomLevel } from '../../types';
import { AnnotationLayer, type TopologyAnnotations } from './AnnotationLayer';

export class HighlightLayer {
  private annotationLayer = new AnnotationLayer();

  draw(
    ctx: CanvasRenderingContext2D,
    args: {
      width?: number;
      height?: number;
      renderedEvents: RenderedEvent[];
      hoveredEventId: string | null;
      highlighted: { backward: Set<string>; forward: Set<string> } | null;
      selectedEventId: string | null;
      selectedArcEventIds: Set<string> | null;
      topologyAnnotations?: TopologyAnnotations | null;
      timeToX?: (t: number) => number;
      zoomLevel?: ZoomLevel;
      viewportScale?: number;
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
      topologyAnnotations,
      timeToX,
      zoomLevel,
      viewportScale
    } = args;
    const byId = new Map(renderedEvents.map((e) => [e.eventId, e] as const));

    const drawRing = (e: RenderedEvent, strokeStyle: string, width = 2) => {
      ctx.save();
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.radius + 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };

    if (highlighted) {
      for (const id of highlighted.backward) {
        const e = byId.get(id);
        if (!e) continue;
        drawRing(e, HIGHLIGHT_BACKWARD, 2);
      }
      for (const id of highlighted.forward) {
        const e = byId.get(id);
        if (!e) continue;
        drawRing(e, HIGHLIGHT_FORWARD, 2);
      }
    }

    if (selectedArcEventIds) {
      for (const id of selectedArcEventIds) {
        const e = byId.get(id);
        if (!e) continue;
        drawRing(e, 'rgba(255,255,255,0.25)', 1.5);
      }
    }

    if (selectedEventId) {
      const e = byId.get(selectedEventId);
      if (e) drawRing(e, 'rgba(255,255,255,0.85)', 2.5);
    }

    if (hoveredEventId) {
      const e = byId.get(hoveredEventId);
      if (e) drawRing(e, 'rgba(255,255,255,0.95)', 2.5);
    }

    // Threads annotations: reuse the topology label system for curated event callouts.
    if (width != null && height != null && timeToX && zoomLevel && viewportScale != null) {
      this.annotationLayer.draw(ctx, {
        width,
        height,
        timeToX,
        zoomLevel,
        viewportScale,
        annotations: topologyAnnotations ?? null
      });
    }
  }
}
