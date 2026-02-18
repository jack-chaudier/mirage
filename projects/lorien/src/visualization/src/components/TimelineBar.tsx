import { useMemo } from 'react';

import { useNarrativeFieldStore } from '../store/narrativeFieldStore';
import { computeTimelineDomain } from './timelineDomain';

export function TimelineBar() {
  const viewport = useNarrativeFieldStore((s) => s.viewport);
  const scenes = useNarrativeFieldStore((s) => s.scenes);
  const events = useNarrativeFieldStore((s) => s.events);

  const domain = useMemo(() => {
    return computeTimelineDomain(events);
  }, [events]);

  return (
    <div
      style={{
        height: 48,
        borderTop: '1px solid rgba(0,0,0,0.08)',
        background: '#fff',
        padding: '8px 12px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#444' }}>
        <span>
          time: {domain.min.toFixed(0)}m - {domain.max.toFixed(0)}m
        </span>
        <span>
          viewport: {viewport.x.toFixed(1)}m - {(viewport.x + viewport.width).toFixed(1)}m
        </span>
      </div>

      <div style={{ position: 'relative', height: 10, marginTop: 8, borderRadius: 999, background: '#f1f1f1' }}>
        {scenes.map((s, idx) => {
          const left = ((s.time_start - domain.min) / (domain.max - domain.min)) * 100;
          const right = ((s.time_end - domain.min) / (domain.max - domain.min)) * 100;
          const width = Math.max(0, right - left);
          return (
            <div
              key={s.id}
              title={s.summary}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${width}%`,
                top: 0,
                bottom: 0,
                borderRadius: 999,
                background: idx % 2 === 0 ? 'rgba(0,0,0,0.12)' : 'rgba(0,0,0,0.18)'
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
