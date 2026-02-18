import { useMemo, useState } from 'react';

import { EVENT_TYPE_COLORS } from '../constants/colors';
import { useNarrativeFieldStore } from '../store/narrativeFieldStore';
import { BeatType, type Event, type StoryExtractionRequest, type StoryExtractionResponse } from '../types';

function assertNever(x: never): never {
  throw new Error(`Unhandled BeatType: ${String(x)}`);
}

function beatColor(beat: BeatType): string {
  switch (beat) {
    case BeatType.SETUP:
      return '#6c757d';
    case BeatType.COMPLICATION:
      return '#0d6efd';
    case BeatType.ESCALATION:
      return '#fd7e14';
    case BeatType.TURNING_POINT:
      return '#dc3545';
    case BeatType.CONSEQUENCE:
      return '#198754';
    default: {
      return assertNever(beat);
    }
  }
}

function sparklinePath(values: number[], width: number, height: number): string {
  if (values.length === 0) return '';
  const max = Math.max(...values, 1e-6);
  const min = Math.min(...values, 0);
  const span = Math.max(1e-6, max - min);
  const step = values.length === 1 ? 0 : width / (values.length - 1);
  return values
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / span) * height;
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

function causalDensity(events: Event[]): number {
  if (events.length < 2) return 0;
  const ids = new Set(events.map((e) => e.id));
  let links = 0;
  for (const e of events) {
    links += e.causal_links.filter((id) => ids.has(id)).length;
  }
  const possible = events.length * (events.length - 1);
  return possible > 0 ? links / possible : 0;
}

export function BeatSheetPanel() {
  const region = useNarrativeFieldStore((s) => s.regionSelection);
  const events = useNarrativeFieldStore((s) => s.events);
  const computedTension = useNarrativeFieldStore((s) => s.computedTension);
  const tensionWeights = useNarrativeFieldStore((s) => s.tensionWeights);
  const metadata = useNarrativeFieldStore((s) => s.metadata);
  const agents = useNarrativeFieldStore((s) => s.agents);
  const scenes = useNarrativeFieldStore((s) => s.scenes);
  const secrets = useNarrativeFieldStore((s) => s.secrets);
  const setRegionSelection = useNarrativeFieldStore((s) => s.setRegionSelection);

  const [extracting, setExtracting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [extractResponse, setExtractResponse] = useState<StoryExtractionResponse | null>(null);

  const selection = useMemo(() => {
    if (!region) return [];
    const lo = Math.min(region.timeStart, region.timeEnd);
    const hi = Math.max(region.timeStart, region.timeEnd);
    return events
      .filter((e) => e.sim_time >= lo && e.sim_time <= hi)
      .slice()
      .sort((a, b) => a.sim_time - b.sim_time);
  }, [events, region]);

  const tensions = useMemo(
    () =>
      selection.map((e) => {
        const t = computedTension.get(e.id) ?? e.metrics.tension ?? 0;
        return Math.max(0, Math.min(1, t));
      }),
    [computedTension, selection]
  );

  const density = useMemo(() => causalDensity(selection), [selection]);

  async function onExtract() {
    setExtractError(null);
    setExtractResponse(null);
    if (!metadata) {
      setExtractError('Missing metadata (payload not loaded yet).');
      return;
    }
    setExtracting(true);
    try {
      const endpoint =
        (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_EXTRACTION_URL ??
        'http://localhost:8000/extract';

      const req: StoryExtractionRequest = {
        selection_type: 'region',
        event_ids: selection.map((e) => e.id),
        tension_weights: tensionWeights,
        genre_preset: 'custom',
        selected_events: selection,
        context: {
          metadata,
          agents,
          scenes,
          secrets
        }
      };

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(req)
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }
      const json = (await res.json()) as StoryExtractionResponse;
      setExtractResponse(json);
    } catch (e) {
      setExtractError(e instanceof Error ? e.message : String(e));
    } finally {
      setExtracting(false);
    }
  }

  if (!region) return null;

  return (
    <div
      style={{
        position: 'absolute',
        right: 16,
        bottom: 16,
        width: 420,
        maxHeight: '70vh',
        background: 'rgba(255,255,255,0.96)',
        border: '1px solid rgba(0,0,0,0.12)',
        borderRadius: 12,
        boxShadow: '0 12px 30px rgba(0,0,0,0.16)',
        overflow: 'hidden',
        display: 'grid',
        gridTemplateRows: 'auto auto 1fr auto'
      }}
    >
      <div style={{ padding: 12, display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <strong style={{ fontSize: 13 }}>Beat Sheet Preview</strong>
          <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>
            t={Math.min(region.timeStart, region.timeEnd).toFixed(1)}m to{' '}
            {Math.max(region.timeStart, region.timeEnd).toFixed(1)}m ({selection.length} events)
          </div>
        </div>
        <button onClick={() => setRegionSelection(null)} aria-label="Close beat sheet">
          Close
        </button>
      </div>

      <div style={{ padding: '0 12px 10px 12px', display: 'grid', gap: 8 }}>
        <svg width="100%" height={36} viewBox="0 0 200 36" preserveAspectRatio="none">
          <path
            d={sparklinePath(tensions, 200, 36)}
            fill="none"
            stroke="rgba(220,53,69,0.9)"
            strokeWidth={2}
          />
        </svg>
        <div style={{ fontSize: 12, color: '#444', display: 'flex', gap: 12 }}>
          <span>peak={Math.max(0, ...tensions).toFixed(2)}</span>
          <span>mean={(tensions.reduce((a, b) => a + b, 0) / Math.max(1, tensions.length)).toFixed(2)}</span>
          <span>causal_density={density.toFixed(3)}</span>
        </div>
      </div>

      <div style={{ padding: 12, overflow: 'auto', borderTop: '1px solid rgba(0,0,0,0.08)' }}>
        <div style={{ display: 'grid', gap: 8 }}>
          {selection.map((e) => {
            const beat = e.beat_type ?? null;
            return (
              <div
                key={e.id}
                style={{
                  border: '1px solid rgba(0,0,0,0.08)',
                  borderRadius: 10,
                  padding: 10
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: '#666' }}>{e.sim_time.toFixed(1)}m</span>
                    <span
                      style={{
                        fontSize: 11,
                        padding: '2px 6px',
                        borderRadius: 999,
                        background: EVENT_TYPE_COLORS[e.type],
                        color: '#fff'
                      }}
                    >
                      {e.type}
                    </span>
                    {beat ? (
                      <span
                        style={{
                          fontSize: 11,
                          padding: '2px 6px',
                          borderRadius: 999,
                          background: beatColor(beat),
                          color: '#fff'
                        }}
                      >
                        {beat}
                      </span>
                    ) : null}
                  </div>
                  <span style={{ fontSize: 12, color: '#444' }}>
                    tension={(computedTension.get(e.id) ?? e.metrics.tension ?? 0).toFixed(2)}
                  </span>
                </div>
                <div style={{ marginTop: 6, fontSize: 12, color: '#222' }}>{e.description}</div>
              </div>
            );
          })}
        </div>
      </div>

      <div
        style={{
          padding: 12,
          borderTop: '1px solid rgba(0,0,0,0.08)',
          display: 'flex',
          justifyContent: 'space-between',
          gap: 12
        }}
      >
        <button onClick={onExtract} disabled={extracting || selection.length === 0}>
          {extracting ? 'Extracting...' : 'Extract Story'}
        </button>
        <span style={{ fontSize: 12, color: '#666' }}>
          {extractResponse?.validation?.valid
            ? `score=${(extractResponse.score?.composite ?? 0).toFixed(2)}`
            : extractResponse
              ? 'invalid arc'
              : 'Phase 4'}
        </span>
      </div>

      {extractError ? (
        <div style={{ padding: 12, borderTop: '1px solid rgba(0,0,0,0.08)', color: '#b02a37' }}>
          <strong style={{ fontSize: 12 }}>Extraction Error</strong>
          <div style={{ fontSize: 12, marginTop: 6, whiteSpace: 'pre-wrap' }}>{extractError}</div>
        </div>
      ) : null}

      {extractResponse?.prose ? (
        <div style={{ padding: 12, borderTop: '1px solid rgba(0,0,0,0.08)' }}>
          <strong style={{ fontSize: 12 }}>Prose</strong>
          <div style={{ marginTop: 8, fontSize: 12, color: '#222', whiteSpace: 'pre-wrap' }}>
            {extractResponse.prose}
          </div>
        </div>
      ) : extractResponse?.beat_sheet ? (
        <div style={{ padding: 12, borderTop: '1px solid rgba(0,0,0,0.08)' }}>
          <strong style={{ fontSize: 12 }}>Beat Sheet (Fallback)</strong>
          <pre
            style={{
              marginTop: 8,
              fontSize: 11,
              color: '#222',
              background: 'rgba(0,0,0,0.04)',
              padding: 10,
              borderRadius: 10,
              overflow: 'auto',
              maxHeight: 240
            }}
          >
            {JSON.stringify(extractResponse.beat_sheet, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  );
}
