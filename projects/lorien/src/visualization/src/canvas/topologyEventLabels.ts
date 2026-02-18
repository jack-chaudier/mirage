import { EventType, ZoomLevel, type Event, type RenderedEvent } from '../types';
import type { TopologyEventLabel } from './layers/AnnotationLayer';

type Tier = 1 | 2 | 3;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function cleanText(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

function descriptionPrefixKey(desc: string): string {
  const clean = cleanText(desc);
  return clean.slice(0, 25);
}

function byTierThenTension(a: { tier: Tier; tension: number }, b: { tier: Tier; tension: number }): number {
  if (a.tier !== b.tier) return a.tier - b.tier;
  return b.tension - a.tension;
}

function truncateToWidth(args: {
  text: string;
  maxWidthPx: number;
  measureTextWidth: (text: string) => number;
}): string {
  const { text, maxWidthPx, measureTextWidth } = args;
  if (measureTextWidth(text) <= maxWidthPx) return text;

  const ellipsis = '...';
  const chars = Array.from(text);
  while (chars.length > 0 && measureTextWidth(chars.join('') + ellipsis) > maxWidthPx) chars.pop();
  const base = chars.join('').trimEnd();
  return base.length > 0 ? base + ellipsis : ellipsis;
}

function overlaps(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

export function buildTopologyEventLabels(args: {
  width: number;
  height: number;
  events: Event[];
  renderedById: Map<string, RenderedEvent>;
  computedTension: Map<string, number>;
  agentNameById: Map<string, string>;
  zoomLevel: ZoomLevel;
  viewportScale: number;
  measureTextWidth: (text: string) => number;
}): TopologyEventLabel[] {
  const {
    width,
    height,
    events,
    renderedById,
    computedTension,
    agentNameById,
    zoomLevel,
    viewportScale,
    measureTextWidth
  } = args;

  type Candidate = {
    eventId: string;
    ax: number;
    ay: number;
    tension: number;
    tier: Tier;
    text: string;
    dedupKey: string;
  };

  const never = new Set<EventType>([
    EventType.CHAT,
    EventType.OBSERVE,
    EventType.PHYSICAL,
    EventType.INTERNAL,
    EventType.SOCIAL_MOVE
  ]);

  const candidates: Candidate[] = [];

  for (const e of events) {
    if (never.has(e.type)) continue;

    const tension = computedTension.get(e.id) ?? e.metrics.tension ?? 0;

    let tier: Tier | null = null;
    if (e.type === EventType.CATASTROPHE) tier = 1;
    else if ((e.type === EventType.CONFLICT || e.type === EventType.REVEAL) && tension > 0.4) tier = 2;
    else if (
      (e.type === EventType.LIE || e.type === EventType.CONFIDE) &&
      tension > 0.3 &&
      zoomLevel === ZoomLevel.DETAIL &&
      viewportScale >= 1.0
    )
      tier = 3;

    if (!tier) continue;

    const re = renderedById.get(e.id);
    if (!re) continue;

    const ax = re.x;
    const ay = re.y;
    if (!Number.isFinite(ax) || !Number.isFinite(ay)) continue;

    const sourceName = agentNameById.get(e.source_agent) ?? e.source_agent;
    const targetId = e.target_agents[0] ?? null;
    const targetName = targetId ? agentNameById.get(targetId) ?? targetId : '???';

    const desc = cleanText(e.description);
    const dedupKey = descriptionPrefixKey(desc);
    let text = desc;
    if (e.type === EventType.CATASTROPHE) text = `⚡ ${sourceName}: ${desc}`;
    else if (e.type === EventType.CONFLICT) text = `⚔ ${sourceName} vs ${targetName}`;
    else if (e.type === EventType.REVEAL) text = `👁 ${desc}`;
    else if (e.type === EventType.LIE) text = `🎭 ${sourceName} lies to ${targetName}`;
    else if (e.type === EventType.CONFIDE) text = `🤝 ${sourceName} confides in ${targetName}`;

    candidates.push({ eventId: e.id, ax, ay, tension, tier, text, dedupKey });
  }

  // Deduplicate by description prefix (first 25 chars) and keep best (tier, tension).
  const bestByKey = new Map<string, Candidate>();
  for (const c of candidates) {
    const prev = bestByKey.get(c.dedupKey);
    if (!prev || byTierThenTension(c, prev) < 0) bestByKey.set(c.dedupKey, c);
  }
  const deduped = Array.from(bestByKey.values());

  // Placement + collision avoidance.
  const padH = 4;
  const padV = 2;
  const boxH = 10 + padV * 2; // 14px total height
  const maxBoxW = 180;
  const maxTextW = Math.max(10, maxBoxW - padH * 2);

  const ordered = [...deduped].sort((a, b) => {
    if (a.tier !== b.tier) return a.tier - b.tier;
    return b.tension - a.tension;
  });

  const placed: TopologyEventLabel[] = [];
  const placedRects: Array<{ x: number; y: number; w: number; h: number; cx: number; cy: number }> = [];

  const margin = 6;

  for (const c of ordered) {
    // Truncate to max width.
    const text = truncateToWidth({ text: c.text, maxWidthPx: maxTextW, measureTextWidth });
    const textW = measureTextWidth(text);
    const w = Math.min(maxBoxW, Math.max(16, textW + padH * 2));
    const h = boxH;

    const ax = c.ax;
    const ay = c.ay;
    if (ax < -40 || ax > width + 40 || ay < -40 || ay > height + 40) continue;

    const tries: Array<{ x: number; y: number }> = [
      { x: ax - w * 0.5, y: ay - 10 - h }, // above
      { x: ax - w * 0.5, y: ay + 10 }, // below
      { x: ax + 12, y: ay - h * 0.5 }, // right
      { x: ax - 12 - w, y: ay - h * 0.5 } // left
    ];

    let placedRect: { x: number; y: number; w: number; h: number; cx: number; cy: number } | null = null;
    for (const t of tries) {
      const x = clamp(t.x, margin, Math.max(margin, width - margin - w));
      const y = clamp(t.y, margin, Math.max(margin, height - margin - h));
      const rect = { x, y, w, h };
      const cx = x + w * 0.5;
      const cy = y + h * 0.5;

      let collide = false;
      for (const p of placedRects) {
        if (overlaps(rect, p)) {
          collide = true;
          break;
        }
        if (Math.abs(cx - p.cx) < 60 && Math.abs(cy - p.cy) < 20) {
          collide = true;
          break;
        }
      }

      if (!collide) {
        placedRect = { ...rect, cx, cy };
        break;
      }
    }

    if (!placedRect) continue;

    placedRects.push(placedRect);
    placed.push({
      eventId: c.eventId,
      ax,
      ay,
      x: placedRect.x,
      y: placedRect.y,
      w: placedRect.w,
      h: placedRect.h,
      text,
      tier: c.tier,
      tension: c.tension
    });
  }

  return placed;
}
