import type { ThreadPath, ViewMode } from '../../types';
import type { InteractionLink } from '../renderModel';

export type ThreadEndpointsByAgent = Map<
  string,
  {
    start: { x: number; y: number };
    end: { x: number; y: number };
  }
>;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function interpolateControlY(points: [number, number][], t: number): number {
  if (points.length === 0) return 0;
  if (t <= points[0]![0]) return points[0]![1];
  if (t >= points[points.length - 1]![0]) return points[points.length - 1]![1];

  let lo = 0;
  let hi = points.length - 2;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t0 = points[mid]![0];
    const t1 = points[mid + 1]![0];
    if (t0 <= t && t < t1) {
      lo = mid;
      break;
    }
    if (t0 < t) lo = mid + 1;
    else hi = mid - 1;
  }

  const i = Math.max(0, Math.min(points.length - 2, lo));
  const p0 = points[i]!;
  const p1 = points[i + 1]!;
  const dt = Math.max(1e-6, p1[0] - p0[0]);
  const a = (t - p0[0]) / dt;
  return p0[1] + (p1[1] - p0[1]) * a;
}

export class ThreadLayer {
  draw(
    ctx: CanvasRenderingContext2D,
    args: {
      width: number;
      height: number;
      threadPaths: ThreadPath[];
      threadEndpoints?: ThreadEndpointsByAgent;
      timeToX: (t: number) => number;
      visibleAgents: Set<string>;
      selectedArcAgentId: string | null;
      viewMode: ViewMode;
      viewportTimeStart?: number;
      viewportTimeEnd?: number;
      viewportScale?: number;
      interactionLinks?: InteractionLink[];
    }
  ) {
    const {
      width,
      height,
      threadPaths,
      threadEndpoints,
      timeToX,
      visibleAgents,
      selectedArcAgentId,
      viewMode,
      viewportTimeStart,
      viewportTimeEnd,
      interactionLinks
    } = args;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (interactionLinks && interactionLinks.length > 0) {
      ctx.save();
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle =
        viewMode === 'topology' ? 'rgba(255,255,255,0.15)' : 'rgba(150,150,150,0.12)';
      ctx.lineWidth = 0.5;
      for (const l of interactionLinks) {
        const x = timeToX(l.t);
        if (x < -20 || x > width + 20) continue;
        const yA = clamp(l.yA, 0, height);
        const yB = clamp(l.yB, 0, height);
        ctx.beginPath();
        ctx.moveTo(x, yA);
        ctx.lineTo(x, yB);
        ctx.stroke();

        if (viewMode === 'topology') {
          const my = (yA + yB) * 0.5;
          ctx.fillStyle = 'rgba(255,255,255,0.12)';
          ctx.beginPath();
          ctx.arc(x, my, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    const labelCandidates: Array<{ y: number; name: string; color: string; alpha: number }> = [];
    const endpointMarkers: Array<{ x: number; y: number; color: string; alpha: number }> = [];

    for (const path of threadPaths) {
      const isVisible = visibleAgents.has(path.agentId);
      const hasSelection = !!selectedArcAgentId;
      const isSelected = hasSelection && path.agentId === selectedArcAgentId;

      const dimmed = hasSelection ? !isSelected : !isVisible;
      const alpha =
        viewMode === 'topology'
          ? hasSelection
            ? isSelected
              ? 0.9
              : isVisible
                ? 0.2
                : 0.12
            : dimmed
              ? 0.12
              : 0.9
          : dimmed
            ? 0.15
            : 0.85;

      if (alpha <= 0.01) continue;

      const color = path.color ?? '#999';
      const points = path.controlPoints;
      if (points.length < 2) continue;

      if (viewMode === 'topology') {
        const strokeSpline = (strokeStyle: string, lineWidth: number, a: number, opts?: { shadowColor?: string; shadowBlur?: number; composite?: GlobalCompositeOperation }) => {
          ctx.save();
          ctx.globalAlpha = a;
          ctx.globalCompositeOperation = opts?.composite ?? 'source-over';
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = lineWidth;
          ctx.shadowColor = opts?.shadowColor ?? 'rgba(0,0,0,0)';
          ctx.shadowBlur = opts?.shadowBlur ?? 0;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;

          ctx.beginPath();
          for (let i = 0; i < points.length - 1; i += 1) {
            const p0 = points[i - 1] ?? points[i]!;
            const p1 = points[i]!;
            const p2 = points[i + 1]!;
            const p3 = points[i + 2] ?? points[i + 1]!;

            const x0 = timeToX(p0[0]);
            const y0 = p0[1];
            const x1 = timeToX(p1[0]);
            const y1 = p1[1];
            const x2 = timeToX(p2[0]);
            const y2 = p2[1];
            const x3 = timeToX(p3[0]);
            const y3 = p3[1];

            const cp1x = x1 + (x2 - x0) / 6;
            const cp1y = y1 + (y2 - y0) / 6;
            const cp2x = x2 - (x3 - x1) / 6;
            const cp2y = y2 - (y3 - y1) / 6;

            if (i === 0) ctx.moveTo(x1, y1);
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
          }
          ctx.stroke();
          ctx.restore();
        };

        if (hasSelection && isSelected) {
          // Selected arc: strong crystallized glow stack.
          const coreWidth = 3.0;

          // Outer glow (screen).
          strokeSpline(color, coreWidth + 6, 0.30, { composite: 'screen', shadowColor: color, shadowBlur: 8 });
          // Dark halo.
          strokeSpline('rgba(0,0,0,0.5)', coreWidth + 2, 0.95);
          // White edge highlight.
          strokeSpline('rgba(255,255,255,0.90)', coreWidth + 1.0, 0.9);
          // Core stroke.
          strokeSpline(color, coreWidth, 1.0);
        } else {
          // Non-selected arcs: per-segment thickness/alpha encodes personal tension.
          for (let i = 0; i < points.length - 1; i += 1) {
            const p0 = points[i - 1] ?? points[i]!;
            const p1 = points[i]!;
            const p2 = points[i + 1]!;
            const p3 = points[i + 2] ?? points[i + 1]!;

            const x0 = timeToX(p0[0]);
            const y0 = p0[1];
            const x1 = timeToX(p1[0]);
            const y1 = p1[1];
            const x2 = timeToX(p2[0]);
            const y2 = p2[1];
            const x3 = timeToX(p3[0]);
            const y3 = p3[1];

            const cp1x = x1 + (x2 - x0) / 6;
            const cp1y = y1 + (y2 - y0) / 6;
            const cp2x = x2 - (x3 - x1) / 6;
            const cp2y = y2 - (y3 - y1) / 6;

            const thick = path.thickness[i] ?? 2;
            const avgT = clamp01((thick - 2) / 4);
            const coreW = hasSelection ? 1.0 : 1.5 + avgT * 1.5;
            const segAlpha = hasSelection ? alpha : alpha * (0.45 + 0.55 * avgT);
            if (segAlpha <= 0.01) continue;

            // Dark halo.
            ctx.save();
            ctx.globalAlpha = segAlpha;
            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth = coreW + 2.0;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
            ctx.stroke();
            ctx.restore();

            // Core.
            ctx.save();
            ctx.globalAlpha = segAlpha;
            ctx.strokeStyle = color;
            ctx.lineWidth = coreW;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
            ctx.stroke();
            ctx.restore();
          }
        }

        // Arc endpoints + name labels (viewport-aware).
        if (viewportTimeStart != null && viewportTimeEnd != null) {
          const yLeft = clamp(interpolateControlY(points, viewportTimeStart), 0, height);
          const yRight = clamp(interpolateControlY(points, viewportTimeEnd), 0, height);

          const startX = 4;
          const endX = width - 4;

          ctx.save();
          ctx.globalAlpha = hasSelection && !isSelected ? alpha : 1.0;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(startX, yLeft, 4, 0, Math.PI * 2);
          ctx.arc(endX, yRight, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();

          // Name label on the left edge.
          ctx.save();
          const labelOffset = yLeft < height * 0.5 ? -10 : 10;
          const ly = clamp(yLeft + labelOffset, 12, height - 12);
          ctx.font = 'bold 11px system-ui, sans-serif';
          ctx.textBaseline = 'middle';
          ctx.textAlign = 'left';
          ctx.lineJoin = 'round';
          ctx.strokeStyle = 'rgba(0,0,0,0.85)';
          ctx.lineWidth = 3;
          ctx.globalAlpha = hasSelection && !isSelected ? alpha : 1.0;
          ctx.strokeText(path.agentName, 6, ly);
          ctx.fillStyle = color;
          ctx.fillText(path.agentName, 6, ly);
          ctx.restore();
        }
      } else {
        // Threads view: per-segment thickness/opacity encodes tension + subtle shadow.
        for (let i = 0; i < points.length - 1; i += 1) {
          const p0 = points[i - 1] ?? points[i]!;
          const p1 = points[i]!;
          const p2 = points[i + 1]!;
          const p3 = points[i + 2] ?? points[i + 1]!;

          const x0 = timeToX(p0[0]);
          const y0 = p0[1];
          const x1 = timeToX(p1[0]);
          const y1 = p1[1];
          const x2 = timeToX(p2[0]);
          const y2 = p2[1];
          const x3 = timeToX(p3[0]);
          const y3 = p3[1];

          const cp1x = x1 + (x2 - x0) / 6;
          const cp1y = y1 + (y2 - y0) / 6;
          const cp2x = x2 - (x3 - x1) / 6;
          const cp2y = y2 - (y3 - y1) / 6;

          const thick = path.thickness[i] ?? 2;
          const avgT = clamp01((thick - 2) / 4);
          const coreW = 1.5 + avgT * 3.0;
          const segAlpha = alpha * (0.5 + 0.4 * avgT);
          if (segAlpha <= 0.01) continue;

          // Shadow pass.
          ctx.globalAlpha = segAlpha;
          ctx.strokeStyle = 'rgba(0,0,0,0.12)';
          ctx.lineWidth = coreW + 2.0;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
          ctx.stroke();

          // Core pass.
          ctx.globalAlpha = segAlpha;
          ctx.strokeStyle = color;
          ctx.lineWidth = coreW;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
          ctx.stroke();
        }

        // Endpoint markers (first/last visible events) + left-edge name labels.
        if (viewportTimeStart != null && (isVisible || isSelected)) {
          const yLeft = clamp(interpolateControlY(points, viewportTimeStart), 0, height);
          labelCandidates.push({ y: yLeft, name: path.agentName, color, alpha });
        }

        const endpoints = threadEndpoints?.get(path.agentId);
        if (endpoints && (isVisible || isSelected)) {
          endpointMarkers.push({ x: endpoints.start.x, y: endpoints.start.y, color, alpha });
          endpointMarkers.push({ x: endpoints.end.x, y: endpoints.end.y, color, alpha });
        }
      }
    }

    // Threads mode: draw endpoint markers and labels in a dedicated pass for cleaner overlap handling.
    if (viewMode === 'threads') {
      if (endpointMarkers.length > 0) {
        ctx.save();
        for (const m of endpointMarkers) {
          if (m.x < -20 || m.x > width + 20) continue;
          if (m.y < -20 || m.y > height + 20) continue;
          ctx.globalAlpha = m.alpha;
          ctx.fillStyle = m.color;
          ctx.beginPath();
          ctx.arc(m.x, m.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      if (labelCandidates.length > 0) {
        // Simple vertical collision avoidance: keep 16px spacing by pushing lower labels down.
        const minSpacing = 16;
        const sorted = [...labelCandidates].sort((a, b) => a.y - b.y);

        const placed: Array<{ y: number; name: string; color: string; alpha: number }> = [];
        for (const c of sorted) {
          const prev = placed[placed.length - 1];
          const y0 = c.y;
          const y = prev && y0 < prev.y + minSpacing ? prev.y + minSpacing : y0;
          placed.push({ ...c, y });
        }

        // Shift back into the visible frame if we overflow.
        const pad = 12;
        const last = placed[placed.length - 1]!;
        const bottomOverflow = last.y - (height - pad);
        if (bottomOverflow > 0) {
          for (const p0 of placed) p0.y -= bottomOverflow;
        }
        const first = placed[0]!;
        const topOverflow = pad - first.y;
        if (topOverflow > 0) {
          for (const p0 of placed) p0.y += topOverflow;
        }

        ctx.save();
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'rgba(0,0,0,0.80)';
        ctx.lineWidth = 3;

        for (const p0 of placed) {
          const y = clamp(p0.y, pad, height - pad);
          ctx.globalAlpha = p0.alpha;
          ctx.strokeText(p0.name, 8, y);
          ctx.fillStyle = p0.color;
          ctx.fillText(p0.name, 8, y);
        }
        ctx.restore();
      }
    }

    ctx.restore();
  }
}
