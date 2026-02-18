import { useEffect } from 'react';

import fakePayload from '../../../data/fake-dinner-party.nf-viz.json';
import { BeatSheetPanel } from './components/BeatSheetPanel';
import { CanvasRenderer } from './components/CanvasRenderer';
import { ControlPanel } from './components/ControlPanel';
import { InfoPanel } from './components/InfoPanel';
import { TimelineBar } from './components/TimelineBar';
import { useNarrativeFieldStore } from './store/narrativeFieldStore';

export function App() {
  const loadEventLog = useNarrativeFieldStore((s) => s.loadEventLog);

  useEffect(() => {
    loadEventLog(JSON.stringify(fakePayload));
  }, [loadEventLog]);

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '320px 1fr 360px',
        gridTemplateRows: '1fr 48px',
        height: '100vh',
        fontFamily: 'system-ui, sans-serif'
      }}
    >
      <div style={{ borderRight: '1px solid rgba(0,0,0,0.08)', background: '#fff' }}>
        <ControlPanel />
      </div>

      <div style={{ position: 'relative', background: '#fafafa' }}>
        <CanvasRenderer />
        <BeatSheetPanel />
      </div>

      <div style={{ borderLeft: '1px solid rgba(0,0,0,0.08)', background: '#fff' }}>
        <InfoPanel />
      </div>

      <div style={{ gridColumn: '1 / -1' }}>
        <TimelineBar />
      </div>
    </div>
  );
}
