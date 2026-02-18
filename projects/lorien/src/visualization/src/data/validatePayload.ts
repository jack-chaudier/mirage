import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseNarrativeFieldPayload, validateNarrativeFieldPayloadCli } from './loader';
import type { NarrativeFieldPayload } from '../types';

function fail(errors: string[]): never {
  // eslint-disable-next-line no-console
  console.error(`Payload validation failed (${errors.length} errors):`);
  for (const e of errors) {
    // eslint-disable-next-line no-console
    console.error(`- ${e}`);
  }
  process.exit(1);
}

async function main() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const argPath = process.argv[2];
  const payloadPath = argPath
    ? path.resolve(process.cwd(), argPath)
    : path.resolve(here, '../../../../data/fake-dinner-party.nf-viz.json');

  const json = await fs.readFile(payloadPath, 'utf-8');
  const parsed = parseNarrativeFieldPayload(json);
  if (!parsed.success) fail(parsed.errors);
  const payload = parsed.payload;

  const errors = validateNarrativeFieldPayloadCli(payload);
  if (errors.length > 0) fail(errors);

  const summary: Pick<NarrativeFieldPayload, 'format_version'> = { format_version: payload.format_version };

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        ok: true,
        path: payloadPath,
        ...summary,
        agents: payload.agents.length,
        secrets: payload.secrets.length,
        events: payload.events.length,
        scenes: payload.scenes.length,
        belief_snapshots: payload.belief_snapshots.length
      },
      null,
      2
    )
  );
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
