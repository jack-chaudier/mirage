import { BackgroundLayer } from './layers/BackgroundLayer';
import { EventNodeLayer } from './layers/EventNodeLayer';
import { HighlightLayer } from './layers/HighlightLayer';
import { TensionTerrainLayer } from './layers/TensionTerrainLayer';
import { TopologyEventLayer } from './layers/TopologyEventLayer';
import { TopologyHighlightLayer } from './layers/TopologyHighlightLayer';
import { ThreadLayer, type ThreadEndpointsByAgent } from './layers/ThreadLayer';
import { HitCanvas } from './HitCanvas';
import type { InteractionLink } from './renderModel';

import type { RenderedEvent, Scene, ThreadPath, ViewMode, ZoomLevel } from '../types';
import type { TopologyAnnotations } from './layers/AnnotationLayer';

export type LayerName = 'background' | 'threads' | 'events' | 'highlight';

export interface LayerManagerDrawArgs {
  scenes: Scene[];
  sceneAvgTension?: number[];
  renderedEvents: RenderedEvent[];
  fieldEvents?: RenderedEvent[];
  threadPaths: ThreadPath[];
  threadEndpoints?: ThreadEndpointsByAgent;
  visibleAgents: Set<string>;
  selectedArcAgentId: string | null;
  viewportTimeStart?: number;
  viewportTimeEnd?: number;
  viewportY?: number;
  viewportHeight?: number;
  interactionLinks?: InteractionLink[];
  hoveredEventId: string | null;
  highlighted?: { backward: Set<string>; forward: Set<string> } | null;
  selectedEventId: string | null;
  selectedArcEventIds?: Set<string> | null;
  topologyAnnotations?: TopologyAnnotations | null;
  timeToX: (t: number) => number;
  zoomLevel: ZoomLevel;
  viewportScale: number;
  viewMode: ViewMode;
}

export class LayerManager {
  private container: HTMLElement;
  private canvases: Record<LayerName, HTMLCanvasElement>;
  private ctx: Record<LayerName, CanvasRenderingContext2D>;
  private dpr = 1;
  private width = 1;
  private height = 1;

  private threadLayer = new ThreadLayer();

  private layersThreads = {
    background: new BackgroundLayer(),
    events: new EventNodeLayer(),
    highlight: new HighlightLayer()
  };

  private layersTopology = {
    background: new TensionTerrainLayer(),
    events: new TopologyEventLayer(),
    highlight: new TopologyHighlightLayer()
  };

  private viewMode: ViewMode = 'threads';
  private backgroundLayer: BackgroundLayer | TensionTerrainLayer = this.layersThreads.background;
  private eventLayer: EventNodeLayer | TopologyEventLayer = this.layersThreads.events;
  private highlightLayer: HighlightLayer | TopologyHighlightLayer = this.layersThreads.highlight;
  readonly hitCanvas = new HitCanvas();

  private dirty: Record<LayerName, boolean> = {
    background: true,
    threads: true,
    events: true,
    highlight: true
  };
  private rafId: number | null = null;
  private lastArgs: LayerManagerDrawArgs | null = null;
  private lastHitEventsRef: RenderedEvent[] | null = null;

  private continuousLayers: Set<LayerName> = new Set();

  constructor(container: HTMLElement) {
    this.container = container;
    this.container.style.position = 'relative';
    this.container.style.overflow = 'hidden';

    this.canvases = {
      background: document.createElement('canvas'),
      threads: document.createElement('canvas'),
      events: document.createElement('canvas'),
      highlight: document.createElement('canvas')
    };

    this.ctx = {
      background: this.mustGet2d(this.canvases.background),
      threads: this.mustGet2d(this.canvases.threads),
      events: this.mustGet2d(this.canvases.events),
      highlight: this.mustGet2d(this.canvases.highlight)
    };

    for (const layer of Object.keys(this.canvases) as LayerName[]) {
      const c = this.canvases[layer];
      c.style.position = 'absolute';
      c.style.left = '0';
      c.style.top = '0';
      c.style.width = '100%';
      c.style.height = '100%';
      c.style.pointerEvents = layer === 'highlight' ? 'auto' : 'none';
      this.container.appendChild(c);
    }

    // Allow the topology terrain layer to request a redraw when its async cache recompute finishes.
    this.layersTopology.background.setInvalidateCallback(() => this.invalidate(['background']));
  }

  private mustGet2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context');
    return ctx;
  }

  setSize(width: number, height: number, dpr = window.devicePixelRatio || 1) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.dpr = dpr;

    for (const layer of Object.keys(this.canvases) as LayerName[]) {
      const c = this.canvases[layer];
      c.width = Math.floor(this.width * dpr);
      c.height = Math.floor(this.height * dpr);
      const ctx = this.ctx[layer];
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.scale(dpr, dpr);
    }

    this.hitCanvas.setSize(this.width, this.height, dpr);
    this.invalidate(['background', 'threads', 'events', 'highlight']);
  }

  invalidate(layers: LayerName[]) {
    for (const l of layers) this.dirty[l] = true;
    this.scheduleDraw();
  }

  setViewMode(mode: ViewMode) {
    if (this.viewMode === mode) return;
    this.viewMode = mode;

    if (mode === 'topology') {
      this.backgroundLayer = this.layersTopology.background;
      this.eventLayer = this.layersTopology.events;
      this.highlightLayer = this.layersTopology.highlight;
      // Animate the topology events layer (turning-point pulses).
      this.continuousLayers = new Set<LayerName>(['events']);
    } else {
      this.backgroundLayer = this.layersThreads.background;
      this.eventLayer = this.layersThreads.events;
      this.highlightLayer = this.layersThreads.highlight;
      this.continuousLayers = new Set();
    }

    this.invalidate(['background', 'threads', 'events', 'highlight']);
  }

  dispose() {
    // Best-effort cleanup for async timers/bitmaps.
    this.layersTopology.background.dispose();
  }

  draw(args: LayerManagerDrawArgs) {
    const prev = this.lastArgs;
    this.lastArgs = args;

    if (!prev) {
      this.invalidate(['background', 'threads', 'events', 'highlight']);
      return;
    }

    // Layer-specific invalidation to keep hover/drag responsive (especially in topology mode).
    if (args.viewMode !== prev.viewMode) {
      this.invalidate(['background', 'threads', 'events', 'highlight']);
      return;
    }

    const nextField = args.fieldEvents ?? args.renderedEvents;
    const prevField = prev.fieldEvents ?? prev.renderedEvents;

    if (
      args.scenes !== prev.scenes ||
      args.sceneAvgTension !== prev.sceneAvgTension ||
      nextField !== prevField ||
      args.timeToX !== prev.timeToX ||
      args.zoomLevel !== prev.zoomLevel ||
      args.viewportScale !== prev.viewportScale ||
      args.viewportY !== prev.viewportY ||
      args.viewportHeight !== prev.viewportHeight ||
      (args.viewMode === 'topology' && args.selectedArcAgentId !== prev.selectedArcAgentId)
    ) {
      this.dirty.background = true;
    }

    if (
      args.threadPaths !== prev.threadPaths ||
      args.threadEndpoints !== prev.threadEndpoints ||
      args.timeToX !== prev.timeToX ||
      args.visibleAgents !== prev.visibleAgents ||
      args.selectedArcAgentId !== prev.selectedArcAgentId ||
      args.interactionLinks !== prev.interactionLinks ||
      args.viewportTimeStart !== prev.viewportTimeStart ||
      args.viewportTimeEnd !== prev.viewportTimeEnd
    ) {
      this.dirty.threads = true;
    }

    if (args.renderedEvents !== prev.renderedEvents) {
      this.dirty.events = true;
    }

    if (
      args.renderedEvents !== prev.renderedEvents ||
      args.hoveredEventId !== prev.hoveredEventId ||
      args.highlighted !== prev.highlighted ||
      args.selectedEventId !== prev.selectedEventId ||
      args.selectedArcEventIds !== prev.selectedArcEventIds ||
      args.topologyAnnotations !== prev.topologyAnnotations ||
      args.zoomLevel !== prev.zoomLevel ||
      args.viewportScale !== prev.viewportScale ||
      args.timeToX !== prev.timeToX
    ) {
      this.dirty.highlight = true;
    }

    this.scheduleDraw();
  }

  private scheduleDraw() {
    if (this.rafId != null) return;
    this.rafId = window.requestAnimationFrame(() => {
      this.rafId = null;
      if (!this.lastArgs) return;
      this.drawDirty(this.lastArgs);
      // Optional animation loop for specific layers (topology pulsing markers, etc).
      if (this.continuousLayers.size > 0) {
        for (const l of this.continuousLayers) this.dirty[l] = true;
        this.scheduleDraw();
      }
    });
  }

  private clear(layer: LayerName) {
    const ctx = this.ctx[layer];
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvases[layer].width, this.canvases[layer].height);
    ctx.restore();
    // Re-apply DPR scaling
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(this.dpr, this.dpr);
  }

  private drawDirty(args: LayerManagerDrawArgs) {
    const now = performance.now();

    if (this.dirty.background) {
      this.clear('background');
      const selectedArcPathPoints =
        args.viewMode === 'topology' && args.selectedArcAgentId
          ? args.threadPaths.find((p) => p.agentId === args.selectedArcAgentId)?.controlPoints ?? null
          : null;
      this.backgroundLayer.draw(this.ctx.background, {
        width: this.width,
        height: this.height,
        scenes: args.scenes,
        sceneAvgTension: args.sceneAvgTension,
        renderedEvents: args.renderedEvents,
        fieldEvents: args.fieldEvents,
        timeToX: args.timeToX,
        viewportY: args.viewportY,
        viewportHeight: args.viewportHeight,
        selectedArcAgentId: args.selectedArcAgentId,
        selectedArcPathPoints,
        viewMode: args.viewMode,
        zoomLevel: args.zoomLevel,
        viewportScale: args.viewportScale
      });
      this.dirty.background = false;
    }

    if (this.dirty.threads) {
      this.clear('threads');
      this.threadLayer.draw(this.ctx.threads, {
        width: this.width,
        height: this.height,
        threadPaths: args.threadPaths,
        threadEndpoints: args.threadEndpoints,
        timeToX: args.timeToX,
        visibleAgents: args.visibleAgents,
        selectedArcAgentId: args.selectedArcAgentId,
        viewMode: args.viewMode,
        viewportTimeStart: args.viewportTimeStart,
        viewportTimeEnd: args.viewportTimeEnd,
        viewportScale: args.viewportScale,
        interactionLinks: args.interactionLinks
      });
      this.dirty.threads = false;
    }

    if (this.dirty.events) {
      this.clear('events');
      this.eventLayer.draw(this.ctx.events, { renderedEvents: args.renderedEvents, now, selectedArcAgentId: args.selectedArcAgentId });
      // HitCanvas is aligned with the event layer; avoid re-rasterizing on purely visual animations.
      if (this.lastHitEventsRef !== args.renderedEvents) {
        this.hitCanvas.draw(args.renderedEvents);
        this.lastHitEventsRef = args.renderedEvents;
      }
      this.dirty.events = false;
    }

    if (this.dirty.highlight) {
      this.clear('highlight');
      this.highlightLayer.draw(this.ctx.highlight, {
        width: this.width,
        height: this.height,
        renderedEvents: args.renderedEvents,
        hoveredEventId: args.hoveredEventId,
        highlighted: args.highlighted ?? null,
        selectedEventId: args.selectedEventId,
        selectedArcEventIds: args.selectedArcEventIds ?? null,
        topologyAnnotations: args.topologyAnnotations ?? null,
        timeToX: args.timeToX,
        zoomLevel: args.zoomLevel,
        viewportScale: args.viewportScale
      });
      this.dirty.highlight = false;
    }
  }
}
