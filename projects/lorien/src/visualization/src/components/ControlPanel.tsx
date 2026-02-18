import { useMemo } from 'react';

import { CHARACTER_COLORS } from '../constants/colors';
import { useNarrativeFieldStore } from '../store/narrativeFieldStore';
import { EventType, type TensionWeights, ZoomLevel } from '../types';

const WEIGHT_KEYS: Array<keyof TensionWeights> = [
  'danger',
  'time_pressure',
  'goal_frustration',
  'relationship_volatility',
  'information_gap',
  'resource_scarcity',
  'moral_cost',
  'irony_density'
];

export function ControlPanel() {
  const metadata = useNarrativeFieldStore((s) => s.metadata);
  const loadEventLog = useNarrativeFieldStore((s) => s.loadEventLog);
  const viewMode = useNarrativeFieldStore((s) => s.viewMode);
  const setViewMode = useNarrativeFieldStore((s) => s.setViewMode);
  const agents = useNarrativeFieldStore((s) => s.agents);
  const filters = useNarrativeFieldStore((s) => s.activeFilters);
  const scale = useNarrativeFieldStore((s) => s.viewport.scale);
  const tensionWeights = useNarrativeFieldStore((s) => s.tensionWeights);
  const setTensionWeights = useNarrativeFieldStore((s) => s.setTensionWeights);
  const applyGenrePreset = useNarrativeFieldStore((s) => s.applyGenrePreset);
  const resetTensionWeights = useNarrativeFieldStore((s) => s.resetTensionWeights);
  const toggleCharacterFilter = useNarrativeFieldStore((s) => s.toggleCharacterFilter);
  const toggleEventTypeFilter = useNarrativeFieldStore((s) => s.toggleEventTypeFilter);
  const zoomLevel = useNarrativeFieldStore((s) => s.zoomLevel);
  const setViewportScale = useNarrativeFieldStore((s) => s.setViewportScale);
  const fitAll = useNarrativeFieldStore((s) => s.fitAll);
  const fitWorld = useNarrativeFieldStore((s) => s.fitWorld);

  const visible = filters.visibleAgents;
  const visibleEventTypes = filters.eventTypeFilter;

  const sliderRows = useMemo(
    () =>
      WEIGHT_KEYS.map((key) => (
        <label key={key} style={{ display: 'grid', gridTemplateColumns: '1fr 64px', gap: 8 }}>
          <span style={{ fontSize: 12, color: '#333' }}>{key}</span>
          <input
            type="range"
            min={0}
            max={3}
            step={0.05}
            value={tensionWeights[key]}
            onChange={(e) => setTensionWeights({ [key]: Number(e.target.value) })}
          />
        </label>
      )),
    [setTensionWeights, tensionWeights]
  );

  return (
    <div style={{ padding: 12, height: '100%', overflow: 'auto' }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, color: '#444' }}>
          View Mode
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            marginTop: 8,
            border: '1px solid rgba(0,0,0,0.14)',
            borderRadius: 12,
            overflow: 'hidden'
          }}
        >
          <button
            type="button"
            onClick={() => setViewMode('threads')}
            aria-pressed={viewMode === 'threads'}
            style={{
              padding: '10px 10px',
              border: 'none',
              background: viewMode === 'threads' ? 'rgba(0,0,0,0.08)' : 'transparent',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Threads View
          </button>
          <button
            type="button"
            onClick={() => setViewMode('topology')}
            aria-pressed={viewMode === 'topology'}
            style={{
              padding: '10px 10px',
              border: 'none',
              background: viewMode === 'topology' ? 'rgba(0,0,0,0.08)' : 'transparent',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Topology View
          </button>
        </div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <strong style={{ fontSize: 13 }}>Dataset</strong>
        <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
          {metadata ? (
            <>
              {metadata.scenario} ({metadata.event_count} events)
            </>
          ) : (
            <>loading...</>
          )}
        </div>
        <label style={{ display: 'block', marginTop: 8 }}>
          <span style={{ fontSize: 12, color: '#333' }}>Load .nf-viz.json</span>
          <input
            type="file"
            accept=".json,.nf-viz.json"
            style={{ display: 'block', marginTop: 6 }}
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const text = await file.text();
              loadEventLog(text);
              // Reset input so selecting the same file again re-triggers onChange.
              e.target.value = '';
            }}
          />
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
        <strong style={{ fontSize: 13 }}>Tension Presets</strong>
        <button onClick={() => applyGenrePreset('thriller')}>Thriller</button>
        <button onClick={() => applyGenrePreset('relationship_drama')}>Drama</button>
        <button onClick={() => applyGenrePreset('mystery')}>Mystery</button>
        <button onClick={() => resetTensionWeights()}>Reset</button>
      </div>

      <div style={{ display: 'grid', gap: 10, marginBottom: 16 }}>{sliderRows}</div>

      <div style={{ marginBottom: 16 }}>
        <strong style={{ fontSize: 13 }}>Characters</strong>
        <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
          {agents.map((a) => (
            <label key={a.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={visible.has(a.id)}
                onChange={() => toggleCharacterFilter(a.id)}
              />
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: CHARACTER_COLORS[a.id] ?? '#999'
                }}
              />
              <span style={{ fontSize: 12 }}>{a.name}</span>
            </label>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <strong style={{ fontSize: 13 }}>Event Types</strong>
        <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
          {(Object.values(EventType) as EventType[]).map((t) => (
            <label key={t} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={visibleEventTypes.has(t)}
                onChange={() => toggleEventTypeFilter(t)}
              />
              <span style={{ fontSize: 12 }}>{t}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <strong style={{ fontSize: 13 }}>Zoom</strong>
        <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            onClick={() => setViewportScale(0.3)}
            aria-pressed={zoomLevel === ZoomLevel.CLOUD}
            style={{
              fontWeight: zoomLevel === ZoomLevel.CLOUD ? 600 : 400,
              background: zoomLevel === ZoomLevel.CLOUD ? 'rgba(0,0,0,0.08)' : 'transparent'
            }}
          >
            Cloud
          </button>
          <button
            onClick={() => setViewportScale(1.0)}
            aria-pressed={zoomLevel === ZoomLevel.THREADS}
            style={{
              fontWeight: zoomLevel === ZoomLevel.THREADS ? 600 : 400,
              background: zoomLevel === ZoomLevel.THREADS ? 'rgba(0,0,0,0.08)' : 'transparent'
            }}
          >
            Threads
          </button>
          <button
            onClick={() => setViewportScale(2.0)}
            aria-pressed={zoomLevel === ZoomLevel.DETAIL}
            style={{
              fontWeight: zoomLevel === ZoomLevel.DETAIL ? 600 : 400,
              background: zoomLevel === ZoomLevel.DETAIL ? 'rgba(0,0,0,0.08)' : 'transparent'
            }}
          >
            Detail
          </button>
          <button
            type="button"
            onClick={() => (viewMode === 'topology' ? fitWorld({ animate: true }) : fitAll())}
            style={{ marginLeft: 4 }}
          >
            Fit
          </button>
        </div>

        <label style={{ display: 'grid', gridTemplateColumns: '1fr 64px', gap: 8, marginTop: 10 }}>
          <span style={{ fontSize: 12, color: '#333' }}>scale</span>
          <input
            type="range"
            min={viewMode === 'topology' ? 0.5 : 0.1}
            max={viewMode === 'topology' ? 4 : 5}
            step={0.05}
            value={scale}
            onChange={(e) => setViewportScale(Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
