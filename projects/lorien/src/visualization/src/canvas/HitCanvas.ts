import type { RenderedEvent } from '../types';

function encodeIndexToRGB(index: number): [number, number, number] {
  // Use 1-based indices so 0,0,0 means "no hit".
  const id = index + 1;
  const r = (id >> 16) & 0xff;
  const g = (id >> 8) & 0xff;
  const b = id & 0xff;
  return [r, g, b];
}

function decodeRGBToIndex(r: number, g: number, b: number): number | null {
  const id = (r << 16) | (g << 8) | b;
  if (id === 0) return null;
  return id - 1;
}

export class HitCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private eventIds: string[] = [];

  constructor() {
    this.canvas = document.createElement('canvas');
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Failed to create 2D context for HitCanvas');
    this.ctx = ctx;
  }

  setSize(width: number, height: number, dpr: number) {
    this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.floor(width * dpr));
    this.canvas.height = Math.max(1, Math.floor(height * dpr));
  }

  draw(events: RenderedEvent[]) {
    this.eventIds = events.map((e) => e.eventId);
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.scale(this.dpr, this.dpr);

    for (let i = 0; i < events.length; i += 1) {
      const e = events[i]!;
      const [r, g, b] = encodeIndexToRGB(i);
      this.ctx.fillStyle = `rgb(${r},${g},${b})`;
      this.ctx.beginPath();
      this.ctx.arc(e.x, e.y, Math.max(3, e.radius + 2), 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  pickEventIdAt(clientX: number, clientY: number): string | null {
    const x = Math.floor(clientX * this.dpr);
    const y = Math.floor(clientY * this.dpr);
    if (x < 0 || y < 0 || x >= this.canvas.width || y >= this.canvas.height) return null;
    const data = this.ctx.getImageData(x, y, 1, 1).data;
    const idx = decodeRGBToIndex(data[0]!, data[1]!, data[2]!);
    if (idx === null) return null;
    return this.eventIds[idx] ?? null;
  }
}

