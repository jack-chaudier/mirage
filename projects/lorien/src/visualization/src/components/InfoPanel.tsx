import { useMemo } from 'react';

import { useNarrativeFieldStore } from '../store/narrativeFieldStore';

export function InfoPanel() {
  const events = useNarrativeFieldStore((s) => s.events);
  const agents = useNarrativeFieldStore((s) => s.agents);
  const locations = useNarrativeFieldStore((s) => s.locations);
  const selectedEventId = useNarrativeFieldStore((s) => s.selectedEventId);
  const selectedArcAgentId = useNarrativeFieldStore((s) => s.selectedArcAgentId);
  const computedTension = useNarrativeFieldStore((s) => s.computedTension);

  const event = useMemo(
    () => (selectedEventId ? events.find((e) => e.id === selectedEventId) ?? null : null),
    [events, selectedEventId]
  );

  const agent = useMemo(
    () => (selectedArcAgentId ? agents.find((a) => a.id === selectedArcAgentId) ?? null : null),
    [agents, selectedArcAgentId]
  );

  const location = useMemo(() => {
    if (!event) return null;
    return locations.find((l) => l.id === event.location_id) ?? null;
  }, [event, locations]);

  return (
    <div style={{ padding: 12, height: '100%', overflow: 'auto' }}>
      {event ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <strong style={{ fontSize: 13 }}>{event.id}</strong>
            <span style={{ fontSize: 12, color: '#555' }}>
              t={event.sim_time.toFixed(1)}m
            </span>
          </div>
          <div style={{ fontSize: 12, color: '#333', marginTop: 8 }}>{event.description}</div>

          <div style={{ marginTop: 12, fontSize: 12 }}>
            <div>
              <strong>Type:</strong> {event.type}
            </div>
            <div>
              <strong>Location:</strong> {location?.name ?? event.location_id}
            </div>
            <div>
              <strong>Source:</strong> {event.source_agent}
            </div>
            <div>
              <strong>Targets:</strong> {event.target_agents.join(', ') || '(none)'}
            </div>
          </div>

          <div style={{ marginTop: 12, fontSize: 12 }}>
            <strong>Metrics</strong>
            <div style={{ marginTop: 6, color: '#333' }}>
              tension: {computedTension.get(event.id)?.toFixed(3) ?? event.metrics.tension.toFixed(3)}
            </div>
            <div style={{ color: '#333' }}>irony: {event.metrics.irony.toFixed(3)}</div>
            <div style={{ color: '#333' }}>
              significance: {event.metrics.significance.toFixed(3)}
            </div>
          </div>

          <div style={{ marginTop: 12, fontSize: 12 }}>
            <strong>Deltas</strong>
            <div style={{ marginTop: 6, display: 'grid', gap: 6 }}>
              {event.deltas.length === 0 ? (
                <div style={{ color: '#666' }}>(none)</div>
              ) : (
                event.deltas.map((d, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: 8,
                      border: '1px solid rgba(0,0,0,0.08)',
                      borderRadius: 6
                    }}
                  >
                    <div style={{ color: '#222' }}>
                      {d.kind} {d.op} {d.attribute} = {String(d.value)}
                    </div>
                    <div style={{ color: '#666', marginTop: 2 }}>{d.reason_display}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      ) : agent ? (
        <>
          <strong style={{ fontSize: 13 }}>{agent.name}</strong>
          <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>
            <div>
              <strong>id:</strong> {agent.id}
            </div>
            <div>
              <strong>initial_location:</strong> {agent.initial_location}
            </div>
            <div>
              <strong>goal_summary:</strong> {agent.goal_summary}
            </div>
            <div>
              <strong>primary_flaw:</strong> {agent.primary_flaw}
            </div>
          </div>
        </>
      ) : (
        <div style={{ fontSize: 12, color: '#666' }}>Click an event to see details.</div>
      )}
    </div>
  );
}

