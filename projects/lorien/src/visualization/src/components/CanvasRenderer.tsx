import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from 'react';

import { LayerManager } from '../canvas/LayerManager';
import {
  buildRenderedEvents,
  buildThreadPathsWithMetadata,
  computeInteractionLinks,
  createTimeToX
} from '../canvas/renderModel';
import { buildTopologyEventLabels } from '../canvas/topologyEventLabels';
import { computeThreadLayout } from '../layout/threadLayout';
import { useNarrativeFieldStore, useVisibleEvents } from '../store/narrativeFieldStore';
import { ZoomLevel } from '../types';
import type { TopologyAnnotations } from '../canvas/layers/AnnotationLayer';

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

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

type DragMode = 'pan' | 'select';
type DragSession = {
  mode: DragMode;
  pointerId: number;
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  startViewportX: number;
  startViewportY: number;
  startViewportWidth: number;
  startViewportHeight: number;
  didDrag: boolean;
};

export function CanvasRenderer() {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasHostRef = useRef<HTMLDivElement | null>(null);
  const layerManagerRef = useRef<LayerManager | null>(null);
  const dragRef = useRef<DragSession | null>(null);
  const zoomAnimRef = useRef<{
    rafId: number | null;
    startTime: number;
    startScale: number;
    targetScale: number;
    anchorTime: number;
    anchorY: number;
  } | null>(null);

  const events = useVisibleEvents();
  const allEvents = useNarrativeFieldStore((s) => s.events);
  const scenes = useNarrativeFieldStore((s) => s.scenes);
  const agents = useNarrativeFieldStore((s) => s.agents);
  const computedTension = useNarrativeFieldStore((s) => s.computedTension);
  const viewport = useNarrativeFieldStore((s) => s.viewport);
  const zoomLevel = useNarrativeFieldStore((s) => s.zoomLevel);
  const viewMode = useNarrativeFieldStore((s) => s.viewMode);
  const visibleAgents = useNarrativeFieldStore((s) => s.activeFilters.visibleAgents);
  const visibleEventTypes = useNarrativeFieldStore((s) => s.activeFilters.eventTypeFilter);
  const minTension = useNarrativeFieldStore((s) => s.activeFilters.minTension);
  const hoveredEventId = useNarrativeFieldStore((s) => s.hoveredEventId);
  const selectedEventId = useNarrativeFieldStore((s) => s.selectedEventId);
  const selectedArcAgentId = useNarrativeFieldStore((s) => s.selectedArcAgentId);
  const causalIndex = useNarrativeFieldStore((s) => s.causalIndex);
  const threadIndex = useNarrativeFieldStore((s) => s.threadIndex);
  const loadErrors = useNarrativeFieldStore((s) => s.loadErrors);
  const clearLoadErrors = useNarrativeFieldStore((s) => s.clearLoadErrors);
  const setHoveredEvent = useNarrativeFieldStore((s) => s.setHoveredEvent);
  const selectEvent = useNarrativeFieldStore((s) => s.selectEvent);
  const selectArc = useNarrativeFieldStore((s) => s.selectArc);
  const setViewport = useNarrativeFieldStore((s) => s.setViewport);
  const setViewportScale = useNarrativeFieldStore((s) => s.setViewportScale);
  const setRegionSelection = useNarrativeFieldStore((s) => s.setRegionSelection);

  const [size, setSize] = useState<{ width: number; height: number }>({ width: 1, height: 1 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [selectionBox, setSelectionBox] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);

  useEffect(() => {
    const host = canvasHostRef.current;
    const wrapper = wrapperRef.current;
    if (!host || !wrapper) return;

    const lm = new LayerManager(host);
    lm.setViewMode(useNarrativeFieldStore.getState().viewMode);
    layerManagerRef.current = lm;

    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      const next = { width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
      setSize(next);
      lm.setSize(next.width, next.height);
      setViewport({ pixelWidth: next.width, pixelHeight: next.height });
    });
    ro.observe(wrapper);

    return () => {
      ro.disconnect();
      lm.dispose();
      layerManagerRef.current = null;
      // Canvases are owned by the container; clear children on unmount.
      host.innerHTML = '';
    };
  }, [setViewport]);

  useEffect(() => {
    const lm = layerManagerRef.current;
    if (!lm) return;
    lm.setViewMode(viewMode);
  }, [viewMode]);

  const layout = useMemo(() => {
    if (agents.length === 0) return null;
    return computeThreadLayout({
      events: allEvents,
      agents,
      scenes,
      canvasHeight: size.height
    });
  }, [agents, allEvents, scenes, size.height]);

  const baseModel = useMemo(() => {
    if (events.length === 0) return null;
    if (agents.length === 0) return null;
    if (!layout) return null;

    const baseThreadPaths = buildThreadPathsWithMetadata({ basePaths: layout.threadPaths, agents });

    const timeToX = createTimeToX({ viewport, width: size.width });

    const viewportRendered = buildRenderedEvents({
      events,
      computedTension,
      timeSamples: layout.timeSamples,
      positions: layout.positions,
      timeToX,
      zoomLevel
    });

    const yScale = viewMode === 'topology' ? size.height / Math.max(1e-6, viewport.height) : 1;
    const yToPx = (y0: number) => (viewMode === 'topology' ? (y0 - viewport.y) * yScale : y0);

    const visibleRendered = viewportRendered
      .filter((e) => isVisibleAtZoom(e.zoomVisibility, zoomLevel))
      .map((e) => (viewMode === 'topology' ? { ...e, y: yToPx(e.y) } : e));

    const fieldEventsData =
      viewMode === 'topology'
        ? allEvents.filter((e) => {
            if (!visibleAgents.has(e.source_agent)) return false;
            if (!visibleEventTypes.has(e.type)) return false;
            const t = computedTension.get(e.id) ?? e.metrics.tension ?? 0;
            if (t < minTension) return false;
            return true;
          })
        : [];

    const fieldRendered =
      viewMode === 'topology'
        ? buildRenderedEvents({
            events: fieldEventsData,
            computedTension,
            timeSamples: layout.timeSamples,
            positions: layout.positions,
            timeToX,
            zoomLevel
          })
        : undefined;

    const agentNameById = new Map(agents.map((a) => [a.id, a.name] as const));

    const interactionLinks = computeInteractionLinks({
      events,
      timeSamples: layout.timeSamples,
      positions: layout.positions,
      tMin: viewport.x,
      tMax: viewport.x + viewport.width,
      maxDyPx: viewMode === 'threads' ? 20 : 15,
      tensionById: computedTension
    })
      .filter((l) => visibleAgents.has(l.a) && visibleAgents.has(l.b))
      .map((l) => (viewMode === 'topology' ? { ...l, yA: yToPx(l.yA), yB: yToPx(l.yB) } : l));

    const measureTextWidth = (() => {
      const c = document.createElement('canvas');
      const cctx = c.getContext('2d');
      if (!cctx) return (s: string) => Array.from(s).length * 7;
      cctx.font = '10px system-ui, sans-serif';
      return (s: string) => cctx.measureText(s).width;
    })();

    const topologyAnnotations: TopologyAnnotations | null = (() => {
      // Threads mode: only curated event labels (no chevrons).
      const chevrons: TopologyAnnotations['chevrons'] = [];

      if (viewMode === 'topology') {
        // --- Convergence/divergence chevrons (lane distance crossings) ---
        const agentIds2 = agents.map((a) => a.id);
        const threshold = 26;
        const hysteresis = 3;
        const tMin = viewport.x;
        const tMax = viewport.x + viewport.width;

        for (let ai = 0; ai < agentIds2.length; ai += 1) {
          for (let bi = ai + 1; bi < agentIds2.length; bi += 1) {
            const a = agentIds2[ai]!;
            const b = agentIds2[bi]!;
            const ysA = layout.positions.get(a);
            const ysB = layout.positions.get(b);
            if (!ysA || !ysB) continue;

            let wasClose = false;
            let lastMarkT = Number.NEGATIVE_INFINITY;
            for (let k = 0; k < layout.timeSamples.length; k += 1) {
              const t = layout.timeSamples[k]!;
              if (t < tMin || t > tMax) continue;
              const ya = ysA[k]!;
              const yb = ysB[k]!;
              const d = Math.abs(ya - yb);
              const close = d < threshold;

              if (!wasClose && close && d < threshold - hysteresis && t - lastMarkT > 1.0) {
                chevrons.push({ t, y: yToPx((ya + yb) / 2), kind: 'converge' });
                lastMarkT = t;
              } else if (wasClose && !close && d > threshold + hysteresis && t - lastMarkT > 1.0) {
                chevrons.push({ t, y: yToPx((ya + yb) / 2), kind: 'diverge' });
                lastMarkT = t;
              }

              wasClose = close;
            }
          }
        }
      }

      const renderedById = new Map(visibleRendered.map((re) => [re.eventId, re] as const));

      const eventLabels = buildTopologyEventLabels({
        width: size.width,
        height: size.height,
        events,
        renderedById,
        computedTension,
        agentNameById,
        zoomLevel,
        viewportScale: viewport.scale,
        measureTextWidth
      });

      return { peaks: [], chevrons, eventLabels };
    })();

    const eventById = new Map(allEvents.map((e) => [e.id, e] as const));
    const sceneAvgTension = scenes.map((s) => {
      let sum = 0;
      let count = 0;
      for (const id of s.event_ids) {
        const t = computedTension.get(id) ?? eventById.get(id)?.metrics.tension ?? 0;
        if (!Number.isFinite(t)) continue;
        sum += t;
        count += 1;
      }
      return count > 0 ? clamp01(sum / count) : 0;
    });

    const threadPaths =
      viewMode === 'topology'
        ? baseThreadPaths.map((p) => ({
            ...p,
            controlPoints: p.controlPoints.map((cp) => [cp[0], yToPx(cp[1])] as [number, number])
          }))
        : baseThreadPaths;

    return {
      threadPaths,
      sceneAvgTension,
      visibleRendered,
      fieldEvents: viewMode === 'topology' ? fieldRendered : undefined,
      timeToX,
      topologyAnnotations,
      viewportTimeStart: viewport.x,
      viewportTimeEnd: viewport.x + viewport.width,
      interactionLinks
    };
  }, [
    allEvents,
    agents,
    computedTension,
    events,
    layout,
    scenes,
    size.height,
    size.width,
    minTension,
    visibleAgents,
    visibleEventTypes,
    viewMode,
    viewport,
    zoomLevel
  ]);

  const drawModel = useMemo(() => {
    if (!baseModel) return null;

    const selectedArcEventIds =
      selectedArcAgentId && threadIndex.get(selectedArcAgentId)
        ? new Set(threadIndex.get(selectedArcAgentId)!)
        : null;

    const baseRadius = zoomLevel === ZoomLevel.CLOUD ? 4 : zoomLevel === ZoomLevel.THREADS ? 8 : 12;

    const renderedEvents = baseModel.visibleRendered.map((re) => {
      let radius = re.radius;
      let opacity = re.opacity;

      // Threads view: tension-driven marker sizing.
      if (viewMode === 'threads') {
        const t = clamp01(re.glowIntensity);
        radius = baseRadius + t * 4;
      }

      if (selectedArcEventIds) {
        if (selectedArcEventIds.has(re.eventId)) radius *= 1.2;
        else opacity = Math.min(opacity, 0.15);
      }

      return { ...re, radius, opacity };
    });

    const threadEndpoints = (() => {
      const acc = new Map<
        string,
        {
          startT: number;
          endT: number;
          start: { x: number; y: number };
          end: { x: number; y: number };
        }
      >();
      for (const re of renderedEvents) {
        const id = re.threadId;
        const cur = acc.get(id);
        if (!cur) {
          acc.set(id, {
            startT: re.simTime,
            endT: re.simTime,
            start: { x: re.x, y: re.y },
            end: { x: re.x, y: re.y }
          });
          continue;
        }
        if (re.simTime < cur.startT) {
          cur.startT = re.simTime;
          cur.start = { x: re.x, y: re.y };
        }
        if (re.simTime > cur.endT) {
          cur.endT = re.simTime;
          cur.end = { x: re.x, y: re.y };
        }
      }

      const out = new Map<string, { start: { x: number; y: number }; end: { x: number; y: number } }>();
      for (const [k, v] of acc) out.set(k, { start: v.start, end: v.end });
      return out;
    })();

    return {
      threadPaths: baseModel.threadPaths,
      threadEndpoints,
      sceneAvgTension: baseModel.sceneAvgTension,
      renderedEvents,
      fieldEvents: baseModel.fieldEvents,
      timeToX: baseModel.timeToX,
      selectedArcEventIds,
      topologyAnnotations: baseModel.topologyAnnotations,
      viewportTimeStart: baseModel.viewportTimeStart,
      viewportTimeEnd: baseModel.viewportTimeEnd,
      interactionLinks: baseModel.interactionLinks
    };
  }, [baseModel, selectedArcAgentId, threadIndex, viewMode, zoomLevel]);

  useEffect(() => {
    const lm = layerManagerRef.current;
    if (!lm) return;
    if (!drawModel) return;

    const highlighted = hoveredEventId ? causalIndex.get(hoveredEventId) : null;

    lm.draw({
      scenes,
      sceneAvgTension: drawModel.sceneAvgTension,
      renderedEvents: drawModel.renderedEvents,
      fieldEvents: drawModel.fieldEvents,
      threadPaths: drawModel.threadPaths,
      threadEndpoints: drawModel.threadEndpoints,
      visibleAgents,
      selectedArcAgentId,
      viewportTimeStart: drawModel.viewportTimeStart,
      viewportTimeEnd: drawModel.viewportTimeEnd,
      viewportY: viewport.y,
      viewportHeight: viewport.height,
      interactionLinks: drawModel.interactionLinks,
      hoveredEventId,
      highlighted,
      selectedEventId,
      selectedArcEventIds: drawModel.selectedArcEventIds,
      topologyAnnotations: drawModel.topologyAnnotations,
      timeToX: drawModel.timeToX,
      zoomLevel,
      viewportScale: viewport.scale,
      viewMode
    });
  }, [
    causalIndex,
    drawModel,
    hoveredEventId,
    scenes,
    selectedArcAgentId,
    selectedEventId,
    visibleAgents,
    viewMode,
    viewport.height,
    viewport.y,
    viewport.scale,
    zoomLevel
  ]);

  const localXY = (evt: { clientX: number; clientY: number }): { x: number; y: number } => {
    const el = wrapperRef.current;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
  };

  const getEventIdAt = (x: number, y: number): string | null => {
    const lm = layerManagerRef.current;
    if (!lm) return null;
    return lm.hitCanvas.pickEventIdAt(x, y);
  };

  const xToTime = (x: number): number => {
    const px = Math.max(1, size.width);
    const frac = Math.max(0, Math.min(1, x / px));
    return viewport.x + frac * viewport.width;
  };

  const commitClickSelection = (id: string | null) => {
    selectEvent(id);
    if (!id) {
      selectArc(null);
      return;
    }
    const ev = useNarrativeFieldStore.getState().events.find((e) => e.id === id);
    if (ev) selectArc(ev.source_agent);
  };

  const handlePointerDown = (evt: ReactPointerEvent) => {
    if (evt.button !== 0) return;
    const { x, y } = localXY(evt);
    setCursor({ x, y });

    evt.currentTarget.setPointerCapture(evt.pointerId);

    if (evt.shiftKey) {
      setHoveredEvent(null);
      setSelectionBox({ x0: x, y0: y, x1: x, y1: y });
      dragRef.current = {
        mode: 'select',
        pointerId: evt.pointerId,
        startX: x,
        startY: y,
        lastX: x,
        lastY: y,
        startViewportX: viewport.x,
        startViewportY: viewport.y,
        startViewportWidth: viewport.width,
        startViewportHeight: viewport.height,
        didDrag: false
      };
      return;
    }

    dragRef.current = {
      mode: 'pan',
      pointerId: evt.pointerId,
      startX: x,
      startY: y,
      lastX: x,
      lastY: y,
      startViewportX: viewport.x,
      startViewportY: viewport.y,
      startViewportWidth: viewport.width,
      startViewportHeight: viewport.height,
      didDrag: false
    };
  };

  const handlePointerMove = (evt: ReactPointerEvent) => {
    const { x, y } = localXY(evt);
    setCursor({ x, y });

    const drag = dragRef.current;
    if (drag) {
      drag.lastX = x;
      drag.lastY = y;
      const dist = Math.hypot(x - drag.startX, y - drag.startY);
      if (dist > 3) drag.didDrag = true;

      if (drag.mode === 'pan') {
        const dx = x - drag.startX;
        const dy = y - drag.startY;
        const deltaTime = -(dx / Math.max(1, size.width)) * drag.startViewportWidth;
        if (viewMode === 'topology') {
          const deltaY = -(dy / Math.max(1, size.height)) * drag.startViewportHeight;
          setViewport({ x: drag.startViewportX + deltaTime, y: drag.startViewportY + deltaY });
        } else {
          setViewport({ x: drag.startViewportX + deltaTime });
        }
        return;
      }

      if (drag.mode === 'select') {
        setSelectionBox((prev) =>
          prev ? { ...prev, x1: x, y1: y } : { x0: x, y0: y, x1: x, y1: y }
        );
        return;
      }
    }

    const id = getEventIdAt(x, y);
    const current = useNarrativeFieldStore.getState().hoveredEventId;
    if (id !== current) setHoveredEvent(id);
  };

  const handlePointerUp = (evt: ReactPointerEvent) => {
    try {
      evt.currentTarget.releasePointerCapture(evt.pointerId);
    } catch {
      // ignore
    }

    const { x, y } = localXY(evt);
    setCursor({ x, y });

    const drag = dragRef.current;
    dragRef.current = null;

    if (!drag) {
      const id = getEventIdAt(x, y);
      commitClickSelection(id);
      return;
    }

    if (drag.mode === 'select') {
      const t0 = xToTime(drag.startX);
      const t1 = xToTime(drag.lastX);
      const lo = Math.min(t0, t1);
      const hi = Math.max(t0, t1);

      const agentSet = new Set<string>();
      for (const e of allEvents) {
        if (e.sim_time < lo || e.sim_time > hi) continue;
        agentSet.add(e.source_agent);
      }

      setRegionSelection({ timeStart: lo, timeEnd: hi, agentIds: Array.from(agentSet).sort() });
      setSelectionBox(null);
      return;
    }

    if (!drag.didDrag) {
      const id = getEventIdAt(x, y);
      commitClickSelection(id);
    }
  };

  const handlePointerLeave = () => {
    setCursor(null);
    if (!dragRef.current) setHoveredEvent(null);
  };

  const handleWheel = (evt: ReactWheelEvent) => {
    const { x, y } = localXY(evt);
    const anchorTime = xToTime(x);
    const factor = Math.exp(-evt.deltaY * 0.0015);
    const targetScale = viewport.scale * factor;

    if (viewMode !== 'topology') {
      setViewportScale(targetScale, anchorTime);
      return;
    }

    // Topology mode: cursor-anchored proportional zoom (X+Y) with a short easing animation.
    const anchorY = viewport.y + (y / Math.max(1, size.height)) * viewport.height;
    const now = performance.now();
    const currentScale = useNarrativeFieldStore.getState().viewport.scale;

    const existing = zoomAnimRef.current;
    if (existing && existing.rafId != null) {
      zoomAnimRef.current = {
        ...existing,
        startTime: now,
        startScale: currentScale,
        targetScale,
        anchorTime,
        anchorY
      };
      return;
    }

    const anim = {
      rafId: null as number | null,
      startTime: now,
      startScale: currentScale,
      targetScale,
      anchorTime,
      anchorY
    };
    zoomAnimRef.current = anim;

    const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
    const durationMs = 100;

    const step = (tNow: number) => {
      const st = zoomAnimRef.current;
      if (!st) return;
      const u = Math.max(0, Math.min(1, (tNow - st.startTime) / durationMs));
      const k = easeOutCubic(u);
      const s = st.startScale + (st.targetScale - st.startScale) * k;
      setViewportScale(s, st.anchorTime, st.anchorY);

      if (u < 1) {
        st.rafId = window.requestAnimationFrame(step);
      } else {
        zoomAnimRef.current = null;
      }
    };

    anim.rafId = window.requestAnimationFrame(step);
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toUpperCase() ?? '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      const st = useNarrativeFieldStore.getState();

      if (e.key === 'Escape') {
        st.setHoveredEvent(null);
        st.selectEvent(null);
        st.selectArc(null);
        st.setRegionSelection(null);
        setSelectionBox(null);
        dragRef.current = null;
        return;
      }

      if (e.key === '1') st.setViewportScale(0.3);
      if (e.key === '2') st.setViewportScale(1.0);
      if (e.key === '3') st.setViewportScale(2.0);

      if (e.key === 'f' || e.key === 'F') {
        if (st.viewMode === 'topology') st.fitWorld({ animate: true });
        else st.fitAll();
      }

      if (e.key === 'Tab') {
        e.preventDefault();
        const ids = st.agents.map((a) => a.id);
        if (ids.length === 0) return;
        const current = st.selectedArcAgentId;
        const idx = current ? ids.indexOf(current) : -1;
        const next = ids[(idx + 1 + ids.length) % ids.length]!;
        st.selectArc(next);
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const hoveredEvent = useMemo(
    () => (hoveredEventId ? allEvents.find((e) => e.id === hoveredEventId) ?? null : null),
    [allEvents, hoveredEventId]
  );

  return (
    <div
      ref={wrapperRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        background: viewMode === 'topology' ? '#080c18' : '#fafafa',
        touchAction: 'none'
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onWheel={handleWheel}
    >
      <div
        ref={canvasHostRef}
        style={{ width: '100%', height: '100%' }}
      />

      {loadErrors ? (
        <div
          role="alert"
          aria-live="polite"
          onPointerDown={(e) => e.stopPropagation()}
          onWheel={(e) => e.stopPropagation()}
          style={{
            position: 'absolute',
            left: 12,
            right: 12,
            top: 12,
            zIndex: 50,
            background: 'rgba(255, 247, 237, 0.98)',
            border: '1px solid rgba(234, 88, 12, 0.35)',
            borderRadius: 12,
            boxShadow: '0 12px 30px rgba(0,0,0,0.16)',
            padding: 12
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
            <strong style={{ fontSize: 13, color: '#9a3412' }}>Invalid .nf-viz payload</strong>
            <button onClick={clearLoadErrors} type="button">
              Dismiss
            </button>
          </div>
          <div style={{ marginTop: 8, maxHeight: 220, overflow: 'auto', fontSize: 12, color: '#7c2d12' }}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {loadErrors.map((err, idx) => (
                <li key={idx} style={{ marginBottom: 4 }}>
                  {err}
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      {selectionBox ? (
        <div
          style={{
            position: 'absolute',
            left: Math.min(selectionBox.x0, selectionBox.x1),
            top: Math.min(selectionBox.y0, selectionBox.y1),
            width: Math.abs(selectionBox.x1 - selectionBox.x0),
            height: Math.abs(selectionBox.y1 - selectionBox.y0),
            border: '1px solid rgba(245,158,11,0.95)',
            background: 'rgba(245,158,11,0.18)',
            borderRadius: 6,
            pointerEvents: 'none'
          }}
        />
      ) : null}

      {cursor && hoveredEvent ? (
        <div
          style={{
            position: 'absolute',
            left: Math.min(size.width - 260, Math.max(8, cursor.x + 14)),
            top: Math.min(size.height - 120, Math.max(8, cursor.y + 14)),
            width: 260,
            background: viewMode === 'topology' ? 'rgba(255,255,255,0.97)' : 'rgba(255,255,255,0.96)',
            border: viewMode === 'topology' ? '1px solid rgba(255,255,255,0.25)' : '1px solid rgba(0,0,0,0.12)',
            borderRadius: 10,
            boxShadow: viewMode === 'topology' ? '0 10px 30px rgba(0,0,0,0.45)' : '0 10px 24px rgba(0,0,0,0.14)',
            padding: 10,
            pointerEvents: 'none'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
            <strong style={{ fontSize: 12 }}>{hoveredEvent.id}</strong>
            <span style={{ fontSize: 12, color: '#666' }}>{hoveredEvent.sim_time.toFixed(1)}m</span>
          </div>
          <div style={{ marginTop: 6, fontSize: 12, color: '#222' }}>{hoveredEvent.description}</div>
        </div>
      ) : null}
    </div>
  );
}
