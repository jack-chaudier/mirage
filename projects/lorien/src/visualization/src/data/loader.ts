import {
  BeatType,
  BeliefState,
  DeltaKind,
  DeltaOp,
  EventType,
  SceneType,
  THEMATIC_AXES,
  type NarrativeFieldPayload
} from '../types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value) && value >= 0;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function isStringArray(value: unknown, opts?: { nonEmptyItems?: boolean }): value is string[] {
  if (!Array.isArray(value)) return false;
  return value.every((v) => (opts?.nonEmptyItems ? isNonEmptyString(v) : typeof v === 'string'));
}

const EVENT_TYPE_VALUES = new Set<string>(Object.values(EventType));
const SCENE_TYPE_VALUES = new Set<string>(Object.values(SceneType));
const BEAT_TYPE_VALUES = new Set<string>(Object.values(BeatType));
const DELTA_KIND_VALUES = new Set<string>(Object.values(DeltaKind));
const DELTA_OP_VALUES = new Set<string>(Object.values(DeltaOp));
const BELIEF_STATE_VALUES = new Set<string>(Object.values(BeliefState));
const THEMATIC_AXIS_VALUES = new Set<string>(THEMATIC_AXES);

const TENSION_COMPONENT_KEYS = [
  'danger',
  'time_pressure',
  'goal_frustration',
  'relationship_volatility',
  'information_gap',
  'resource_scarcity',
  'moral_cost',
  'irony_density'
] as const;

function isEventType(value: unknown): value is EventType {
  return typeof value === 'string' && EVENT_TYPE_VALUES.has(value);
}

function isBeatType(value: unknown): value is BeatType {
  return typeof value === 'string' && BEAT_TYPE_VALUES.has(value);
}

function isDeltaKind(value: unknown): value is DeltaKind {
  return typeof value === 'string' && DELTA_KIND_VALUES.has(value);
}

function isDeltaOp(value: unknown): value is DeltaOp {
  return typeof value === 'string' && DELTA_OP_VALUES.has(value);
}

function isBeliefState(value: unknown): value is BeliefState {
  return typeof value === 'string' && BELIEF_STATE_VALUES.has(value);
}

function expectString(
  obj: Record<string, unknown>,
  key: string,
  errors: string[],
  ctx: string,
  opts?: { nonEmpty?: boolean }
): string {
  const value = obj[key];
  if (typeof value !== 'string') {
    errors.push(`${ctx}.${key} must be a string`);
    return '';
  }
  if (opts?.nonEmpty && value.trim().length === 0) {
    errors.push(`${ctx}.${key} must be a non-empty string`);
  }
  return value;
}

function expectNumber(
  obj: Record<string, unknown>,
  key: string,
  errors: string[],
  ctx: string,
  opts?: { min?: number; max?: number; integer?: boolean }
): number {
  const value = obj[key];
  if (!isFiniteNumber(value)) {
    errors.push(`${ctx}.${key} must be a finite number`);
    return 0;
  }
  if (opts?.integer && !Number.isInteger(value)) {
    errors.push(`${ctx}.${key} must be an integer`);
  }
  if (opts?.min !== undefined && value < opts.min) {
    errors.push(`${ctx}.${key} must be >= ${opts.min}`);
  }
  if (opts?.max !== undefined && value > opts.max) {
    errors.push(`${ctx}.${key} must be <= ${opts.max}`);
  }
  return value;
}

function readOptionalString(obj: Record<string, unknown>, key: string, ctx: string): string | undefined {
  const value = obj[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') return value;
  console.warn(`[payload] ${ctx}.${key} should be a string; ignoring non-string value.`);
  return undefined;
}

function readOptionalNumber(obj: Record<string, unknown>, key: string, ctx: string): number | undefined {
  const value = obj[key];
  if (value === undefined || value === null) return undefined;
  if (isFiniteNumber(value)) return value;
  console.warn(`[payload] ${ctx}.${key} should be a finite number; ignoring invalid value.`);
  return undefined;
}

function readOptionalBoolean(obj: Record<string, unknown>, key: string, ctx: string): boolean | undefined {
  const value = obj[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'boolean') return value;
  console.warn(`[payload] ${ctx}.${key} should be a boolean; ignoring invalid value.`);
  return undefined;
}

function validateAgentManifestItem(raw: unknown, idx: number, errors: string[]) {
  if (!isRecord(raw)) {
    errors.push(`agents[${idx}] must be an object`);
    return;
  }
  const ctx = `agents[${idx}]`;
  expectString(raw, 'id', errors, ctx, { nonEmpty: true });
  expectString(raw, 'name', errors, ctx, { nonEmpty: true });
  expectString(raw, 'initial_location', errors, ctx, { nonEmpty: true });
  expectString(raw, 'goal_summary', errors, ctx);
  expectString(raw, 'primary_flaw', errors, ctx);
}

function validateSceneItem(raw: unknown, idx: number, errors: string[]) {
  if (!isRecord(raw)) {
    errors.push(`scenes[${idx}] must be an object`);
    return;
  }

  const id = typeof raw.id === 'string' ? raw.id : '';
  const ctx = id.trim().length > 0 ? `scenes[${idx}] (${id})` : `scenes[${idx}]`;

  expectString(raw, 'id', errors, ctx, { nonEmpty: true });

  if (typeof raw.scene_type !== 'string') {
    errors.push(`${ctx}.scene_type must be a string`);
  } else if (!SCENE_TYPE_VALUES.has(raw.scene_type)) {
    // Forward-compatible warning: tolerate unknown scene types from newer engines.
    console.warn(`[payload] Unknown scene_type "${raw.scene_type}" at ${ctx}`);
  }

  if (!isStringArray(raw.event_ids, { nonEmptyItems: true }) || raw.event_ids.length === 0) {
    errors.push(`${ctx}.event_ids must be a non-empty array of strings`);
  }

  const time_start = raw.time_start;
  const time_end = raw.time_end;
  if (!isFiniteNumber(time_start)) errors.push(`${ctx}.time_start must be a finite number`);
  if (!isFiniteNumber(time_end)) errors.push(`${ctx}.time_end must be a finite number`);
  if (isFiniteNumber(time_start) && isFiniteNumber(time_end) && time_start > time_end) {
    errors.push(`${ctx}.time_start must be <= time_end`);
  }
}

function validateEventItem(raw: unknown, idx: number, errors: string[]) {
  if (!isRecord(raw)) {
    errors.push(`events[${idx}] must be an object`);
    return;
  }

  const id = typeof raw.id === 'string' ? raw.id : '';
  const ctx = id.trim().length > 0 ? `events[${idx}] (${id})` : `events[${idx}]`;

  expectString(raw, 'id', errors, ctx, { nonEmpty: true });

  if (!isEventType(raw.type)) {
    errors.push(
      `${ctx}.type must be a valid EventType (${Array.from(EVENT_TYPE_VALUES).join(', ')})`
    );
  }

  const sim_time = raw.sim_time;
  if (!isFiniteNumber(sim_time) || sim_time < 0) errors.push(`${ctx}.sim_time must be a finite number >= 0`);

  if (!isNonNegativeInteger(raw.tick_id)) errors.push(`${ctx}.tick_id must be a non-negative integer`);
  if (!isNonNegativeInteger(raw.order_in_tick))
    errors.push(`${ctx}.order_in_tick must be a non-negative integer`);

  expectString(raw, 'source_agent', errors, ctx, { nonEmpty: true });

  if (!isStringArray(raw.target_agents, { nonEmptyItems: true })) {
    errors.push(`${ctx}.target_agents must be an array of strings`);
  }

  expectString(raw, 'location_id', errors, ctx, { nonEmpty: true });

  if (!isStringArray(raw.causal_links, { nonEmptyItems: true })) {
    errors.push(`${ctx}.causal_links must be an array of strings`);
  }

  if (typeof raw.description !== 'string') {
    errors.push(`${ctx}.description must be a string`);
  }

  // beat_type is optional; allow null/undefined but validate strings.
  if ('beat_type' in raw && raw.beat_type !== null && raw.beat_type !== undefined && !isBeatType(raw.beat_type)) {
    errors.push(
      `${ctx}.beat_type must be a valid BeatType (${Array.from(BEAT_TYPE_VALUES).join(', ')})`
    );
  }

  const metrics = raw.metrics;
  if (!isRecord(metrics)) {
    errors.push(`${ctx}.metrics must be an object`);
  } else {
    const tension = metrics.tension;
    if (!isFiniteNumber(tension) || tension < 0 || tension > 1) {
      errors.push(`${ctx}.metrics.tension must be a number in [0,1]`);
    }
    const irony = metrics.irony;
    if (!isFiniteNumber(irony) || irony < 0) {
      errors.push(`${ctx}.metrics.irony must be a number >= 0`);
    }
    const significance = metrics.significance;
    if (!isFiniteNumber(significance) || significance < 0) {
      errors.push(`${ctx}.metrics.significance must be a number >= 0`);
    }
    const ironyCollapse = metrics.irony_collapse;
    if (ironyCollapse !== null && ironyCollapse !== undefined) {
      if (!isRecord(ironyCollapse)) {
        errors.push(`${ctx}.metrics.irony_collapse must be an object or null`);
      } else {
        if (typeof ironyCollapse.detected !== 'boolean') {
          errors.push(`${ctx}.metrics.irony_collapse.detected must be a boolean`);
        }
        if (!isFiniteNumber(ironyCollapse.drop)) {
          errors.push(`${ctx}.metrics.irony_collapse.drop must be a finite number`);
        }
      }
    }

    const comps = metrics.tension_components;
    if (!isRecord(comps)) {
      errors.push(`${ctx}.metrics.tension_components must be an object`);
    } else {
      for (const k of TENSION_COMPONENT_KEYS) {
        const value = comps[k];
        if (!isFiniteNumber(value) || value < 0 || value > 1) {
          errors.push(`${ctx}.metrics.tension_components.${k} must be a number in [0,1]`);
        }
      }
    }

    const thematicShift = metrics.thematic_shift;
    if (!isRecord(thematicShift)) {
      errors.push(`${ctx}.metrics.thematic_shift must be an object`);
    } else {
      for (const [axis, delta] of Object.entries(thematicShift)) {
        if (!isFiniteNumber(delta)) {
          errors.push(`${ctx}.metrics.thematic_shift.${axis} must be a finite number`);
          continue;
        }
        if (!THEMATIC_AXIS_VALUES.has(axis)) {
          // Forward-compatible warning for newly introduced axes.
          console.warn(`[payload] Unknown thematic axis "${axis}" at ${ctx}.metrics.thematic_shift`);
        }
      }
    }
  }

  const deltas = raw.deltas;
  if (!Array.isArray(deltas)) {
    errors.push(`${ctx}.deltas must be an array`);
  } else {
    for (let i = 0; i < deltas.length; i += 1) {
      const d = deltas[i] as unknown;
      const dctx = `${ctx}.deltas[${i}]`;
      if (!isRecord(d)) {
        errors.push(`${dctx} must be an object`);
        continue;
      }
      if (!isNonEmptyString(d.kind)) {
        errors.push(`${dctx}.kind must be a non-empty string`);
      } else if (!isDeltaKind(d.kind)) {
        // Forward-compatible warning: preserve unknown delta kinds instead of rejecting payloads.
        console.warn(`[payload] Unknown delta kind "${d.kind}" at ${dctx}.kind`);
      }
      if (!isDeltaOp(d.op)) {
        errors.push(`${dctx}.op must be a valid DeltaOp (${Array.from(DELTA_OP_VALUES).join(', ')})`);
      }
      // Extra guardrails: these fields are required by the UI (InfoPanel/threadLayout).
      expectString(d, 'agent', errors, dctx, { nonEmpty: true });
      if ('agent_b' in d && d.agent_b !== null && d.agent_b !== undefined && typeof d.agent_b !== 'string') {
        errors.push(`${dctx}.agent_b must be a string or null`);
      }
      expectString(d, 'attribute', errors, dctx);

      const value = d.value;
      if (typeof value !== 'number' && typeof value !== 'string' && typeof value !== 'boolean') {
        errors.push(`${dctx}.value must be a number|string|boolean`);
      }

      expectString(d, 'reason_code', errors, dctx);
      expectString(d, 'reason_display', errors, dctx);
    }
  }
}

function isValidAgent(raw: unknown): raw is NarrativeFieldPayload['agents'][number] {
  return (
    isRecord(raw) &&
    isNonEmptyString(raw.id) &&
    isNonEmptyString(raw.name) &&
    isNonEmptyString(raw.initial_location) &&
    typeof raw.goal_summary === 'string' &&
    typeof raw.primary_flaw === 'string'
  );
}

function isValidLocation(raw: unknown): raw is NarrativeFieldPayload['locations'][number] {
  return (
    isRecord(raw) &&
    isNonEmptyString(raw.id) &&
    isNonEmptyString(raw.name) &&
    isFiniteNumber(raw.privacy) &&
    isFiniteNumber(raw.capacity) &&
    isStringArray(raw.adjacent) &&
    isStringArray(raw.overhear_from) &&
    isFiniteNumber(raw.overhear_probability) &&
    typeof raw.description === 'string'
  );
}

function isValidSecret(raw: unknown): raw is NarrativeFieldPayload['secrets'][number] {
  return (
    isRecord(raw) &&
    isNonEmptyString(raw.id) &&
    typeof raw.description === 'string' &&
    typeof raw.truth_value === 'boolean' &&
    isStringArray(raw.holder) &&
    (raw.about === null || typeof raw.about === 'string') &&
    typeof raw.content_type === 'string' &&
    isStringArray(raw.initial_knowers) &&
    isStringArray(raw.initial_suspecters) &&
    isFiniteNumber(raw.dramatic_weight) &&
    typeof raw.reveal_consequences === 'string'
  );
}

function isValidScene(raw: unknown): raw is NarrativeFieldPayload['scenes'][number] {
  return (
    isRecord(raw) &&
    isNonEmptyString(raw.id) &&
    isStringArray(raw.event_ids, { nonEmptyItems: true }) &&
    isNonEmptyString(raw.location) &&
    isStringArray(raw.participants, { nonEmptyItems: true }) &&
    isFiniteNumber(raw.time_start) &&
    isFiniteNumber(raw.time_end) &&
    isFiniteNumber(raw.tick_start) &&
    isFiniteNumber(raw.tick_end) &&
    Array.isArray(raw.tension_arc) &&
    raw.tension_arc.every((v) => isFiniteNumber(v)) &&
    isFiniteNumber(raw.tension_peak) &&
    isFiniteNumber(raw.tension_mean) &&
    typeof raw.dominant_theme === 'string' &&
    typeof raw.scene_type === 'string' &&
    typeof raw.summary === 'string'
  );
}

function isValidEvent(raw: unknown): raw is NarrativeFieldPayload['events'][number] {
  if (!isRecord(raw)) return false;
  if (!isNonEmptyString(raw.id)) return false;
  if (!isEventType(raw.type)) return false;
  if (!isFiniteNumber(raw.sim_time) || raw.sim_time < 0) return false;
  if (!isNonNegativeInteger(raw.tick_id)) return false;
  if (!isNonNegativeInteger(raw.order_in_tick)) return false;
  if (!isNonEmptyString(raw.source_agent)) return false;
  if (!isStringArray(raw.target_agents, { nonEmptyItems: true })) return false;
  if (!isNonEmptyString(raw.location_id)) return false;
  if (!isStringArray(raw.causal_links, { nonEmptyItems: true })) return false;
  if (typeof raw.description !== 'string') return false;
  if (raw.beat_type !== undefined && raw.beat_type !== null && !isBeatType(raw.beat_type)) return false;
  if (!isRecord(raw.metrics)) return false;
  if (!Array.isArray(raw.deltas)) return false;
  return true;
}

function isValidBeliefSnapshot(raw: unknown): raw is NarrativeFieldPayload['belief_snapshots'][number] {
  if (!isRecord(raw)) return false;
  if (!isNonNegativeInteger(raw.tick_id)) return false;
  if (!isFiniteNumber(raw.sim_time)) return false;
  if (!isRecord(raw.beliefs)) return false;
  if (!isRecord(raw.agent_irony)) return false;
  if (!isFiniteNumber(raw.scene_irony)) return false;
  return true;
}

export type NarrativeFieldPayloadParseResult =
  | { success: true; payload: NarrativeFieldPayload }
  | { success: false; errors: string[] };

export function parseNarrativeFieldPayload(json: string): NarrativeFieldPayloadParseResult {
  const errors: string[] = [];
  let raw: unknown;

  try {
    raw = JSON.parse(json) as unknown;
  } catch (e) {
    return {
      success: false,
      errors: [`Invalid JSON: ${e instanceof Error ? e.message : String(e)}`]
    };
  }

  if (!isRecord(raw)) {
    return { success: false, errors: ['NarrativeFieldPayload must be a JSON object'] };
  }

  const format_version =
    typeof raw.format_version === 'string'
      ? raw.format_version
      : raw.format_version === undefined
        ? '1.0.0'
        : (() => {
            errors.push('NarrativeFieldPayload.format_version must be a string');
            return '1.0.0';
          })();

  const metadata = raw.metadata;
  const agents = raw.agents;
  const locations = raw.locations;
  const secrets = raw.secrets;
  const events = raw.events;
  const scenes = raw.scenes;
  const belief_snapshots = raw.belief_snapshots;

  const metadataRecord = isRecord(metadata) ? metadata : null;
  if (!metadataRecord) errors.push('NarrativeFieldPayload.metadata must be an object');

  if (!Array.isArray(agents)) errors.push('NarrativeFieldPayload.agents must be an array');
  if (!Array.isArray(locations)) errors.push('NarrativeFieldPayload.locations must be an array');
  if (!Array.isArray(secrets)) errors.push('NarrativeFieldPayload.secrets must be an array');
  if (!Array.isArray(events)) errors.push('NarrativeFieldPayload.events must be an array');
  if (!Array.isArray(scenes)) errors.push('NarrativeFieldPayload.scenes must be an array');
  if (!Array.isArray(belief_snapshots))
    errors.push('NarrativeFieldPayload.belief_snapshots must be an array');

  const parsedMetadata: NarrativeFieldPayload['metadata'] = metadataRecord
    ? {
        simulation_id:
          readOptionalString(metadataRecord, 'simulation_id', 'metadata')?.trim() ||
          readOptionalString(metadataRecord, 'deterministic_id', 'metadata')?.trim() ||
          'sim_unknown',
        deterministic_id: readOptionalString(metadataRecord, 'deterministic_id', 'metadata'),
        scenario: expectString(metadataRecord, 'scenario', errors, 'metadata', { nonEmpty: true }),
        total_ticks: expectNumber(metadataRecord, 'total_ticks', errors, 'metadata', {
          min: 0,
          integer: true
        }),
        total_sim_time: expectNumber(metadataRecord, 'total_sim_time', errors, 'metadata', { min: 0 }),
        agent_count: expectNumber(metadataRecord, 'agent_count', errors, 'metadata', { min: 0, integer: true }),
        event_count: expectNumber(metadataRecord, 'event_count', errors, 'metadata', { min: 0, integer: true }),
        snapshot_interval: expectNumber(metadataRecord, 'snapshot_interval', errors, 'metadata', { min: 0 }),
        timestamp: readOptionalString(metadataRecord, 'timestamp', 'metadata')?.trim() || 'unknown',
        raw_event_count:
          metadataRecord.raw_event_count === undefined
            ? undefined
            : isFiniteNumber(metadataRecord.raw_event_count)
              ? metadataRecord.raw_event_count
              : (() => {
                  errors.push('metadata.raw_event_count must be a finite number');
                  return undefined;
                })(),
        seed: readOptionalNumber(metadataRecord, 'seed', 'metadata'),
        time_scale: readOptionalNumber(metadataRecord, 'time_scale', 'metadata'),
        truncated: readOptionalBoolean(metadataRecord, 'truncated', 'metadata'),
        config_hash: readOptionalString(metadataRecord, 'config_hash', 'metadata'),
        python_version: readOptionalString(metadataRecord, 'python_version', 'metadata'),
        git_commit:
          metadataRecord.git_commit === null
            ? null
            : readOptionalString(metadataRecord, 'git_commit', 'metadata')
      }
    : {
        simulation_id: 'sim_unknown',
        scenario: '',
        total_ticks: 0,
        total_sim_time: 0,
        agent_count: 0,
        event_count: 0,
        snapshot_interval: 0,
        timestamp: 'unknown'
      };

  const agentArray = Array.isArray(agents) ? agents : [];
  const sceneArray = Array.isArray(scenes) ? scenes : [];
  const eventArray = Array.isArray(events) ? events : [];

  for (let i = 0; i < agentArray.length; i += 1) validateAgentManifestItem(agentArray[i], i, errors);
  for (let i = 0; i < sceneArray.length; i += 1) validateSceneItem(sceneArray[i], i, errors);
  for (let i = 0; i < eventArray.length; i += 1) validateEventItem(eventArray[i], i, errors);

  const locationArray = Array.isArray(locations) ? locations : [];
  const secretArray = Array.isArray(secrets) ? secrets : [];
  const beliefSnapshotArray = Array.isArray(belief_snapshots) ? belief_snapshots : [];

  for (let i = 0; i < locationArray.length; i += 1) {
    if (!isValidLocation(locationArray[i])) {
      errors.push(`locations[${i}] is invalid`);
    }
  }
  for (let i = 0; i < secretArray.length; i += 1) {
    if (!isValidSecret(secretArray[i])) {
      errors.push(`secrets[${i}] is invalid`);
    }
  }
  for (let i = 0; i < beliefSnapshotArray.length; i += 1) {
    if (!isValidBeliefSnapshot(beliefSnapshotArray[i])) {
      errors.push(`belief_snapshots[${i}] is invalid`);
    }
  }

  const parsedAgents = agentArray.filter(isValidAgent);
  const parsedLocations = locationArray.filter(isValidLocation);
  const parsedSecrets = secretArray.filter(isValidSecret);
  const parsedEvents = eventArray.filter(isValidEvent);
  const parsedScenes = sceneArray.filter(isValidScene);
  const parsedBeliefSnapshots = beliefSnapshotArray.filter(isValidBeliefSnapshot);

  if (parsedAgents.length !== agentArray.length) errors.push('agents contains invalid entries');
  if (parsedLocations.length !== locationArray.length) errors.push('locations contains invalid entries');
  if (parsedSecrets.length !== secretArray.length) errors.push('secrets contains invalid entries');
  if (parsedEvents.length !== eventArray.length) errors.push('events contains invalid entries');
  if (parsedScenes.length !== sceneArray.length) errors.push('scenes contains invalid entries');
  if (parsedBeliefSnapshots.length !== beliefSnapshotArray.length)
    errors.push('belief_snapshots contains invalid entries');

  if (errors.length > 0) return { success: false, errors };

  const payload: NarrativeFieldPayload = {
    // Best-effort forward compatibility: default if missing.
    format_version,
    metadata: parsedMetadata,
    agents: parsedAgents,
    locations: parsedLocations,
    secrets: parsedSecrets,
    events: parsedEvents,
    scenes: parsedScenes,
    belief_snapshots: parsedBeliefSnapshots
  };

  return { success: true, payload };
}

export function validateNarrativeFieldPayloadCli(payload: NarrativeFieldPayload): string[] {
  // Additional invariants/cross-references used by the CLI validator (not required for UI load).
  const errors: string[] = [];

  if (payload.format_version !== '1.0.0') {
    errors.push(`format_version must be "1.0.0" (got ${JSON.stringify(payload.format_version)})`);
  }

  if (payload.metadata.event_count !== payload.events.length) {
    errors.push(
      `metadata.event_count (${payload.metadata.event_count}) != events.length (${payload.events.length})`
    );
  }
  if (payload.metadata.agent_count !== payload.agents.length) {
    errors.push(
      `metadata.agent_count (${payload.metadata.agent_count}) != agents.length (${payload.agents.length})`
    );
  }

  const agentIds = new Set(payload.agents.map((a) => a.id));
  const eventIds = new Set(payload.events.map((e) => e.id));

  // --- secrets ---
  for (const s of payload.secrets) {
    if (!Array.isArray(s.holder) || !s.holder.every((h) => typeof h === 'string' && h.length > 0)) {
      errors.push(`secret ${s.id}: holder must be string[]`);
    }
  }

  // --- events ---
  for (const e of payload.events) {
    if (!e.id) errors.push(`event has empty id`);
    if (!agentIds.has(e.source_agent))
      errors.push(`event ${e.id}: unknown source_agent ${e.source_agent}`);
    for (const t of e.target_agents) {
      if (!agentIds.has(t)) errors.push(`event ${e.id}: unknown target_agent ${t}`);
    }

    // causal links must reference valid IDs
    for (const link of e.causal_links) {
      if (!eventIds.has(link)) {
        errors.push(`event ${e.id}: causal_links references missing event id ${link}`);
      }
      if (link === e.id) {
        errors.push(`event ${e.id}: causal_links must not reference itself`);
      }
    }
  }

  // --- scenes cover all events without overlaps/gaps ---
  const seen = new Map<string, string>(); // eventId -> sceneId
  for (const sc of payload.scenes) {
    for (const id of sc.event_ids) {
      const prev = seen.get(id);
      if (prev) errors.push(`event ${id} appears in multiple scenes (${prev}, ${sc.id})`);
      else seen.set(id, sc.id);
    }
  }

  for (const id of eventIds) {
    if (!seen.has(id)) errors.push(`event ${id} is not included in any scene.event_ids`);
  }

  // --- belief snapshots ---
  for (const [idx, snap] of payload.belief_snapshots.entries()) {
    if (!isFiniteNumber(snap.sim_time)) errors.push(`belief_snapshot[${idx}]: sim_time must be number`);
    if (typeof snap.tick_id !== 'number') errors.push(`belief_snapshot[${idx}]: tick_id must be number`);

    for (const agentId of Object.keys(snap.beliefs)) {
      if (!agentIds.has(agentId)) errors.push(`belief_snapshot[${idx}]: unknown agent key ${agentId}`);
      const row = snap.beliefs[agentId]!;
      for (const secretId of Object.keys(row)) {
        if (secretId.trim().length === 0) {
          errors.push(`belief_snapshot[${idx}]: empty belief key for agent ${agentId}`);
          continue;
        }
        const belief = row[secretId];
        if (!isBeliefState(belief))
          errors.push(
            `belief_snapshot[${idx}]: invalid BeliefState ${String(belief)} for ${agentId}/${secretId}`
          );
      }
    }
  }

  return errors;
}
