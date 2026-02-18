import type { CausalIndex } from '../types';
import type { Event } from '../types';

type Cone = { backward: Set<string>; forward: Set<string> };

function bfs(start: string, links: Map<string, string[]>, maxDepth: number): Set<string> {
  const visited = new Set<string>([start]);
  const result = new Set<string>();

  let frontier: string[] = [start];

  for (let depth = 0; depth < maxDepth; depth += 1) {
    const next: string[] = [];
    for (const current of frontier) {
      for (const neighbor of links.get(current) ?? []) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        result.add(neighbor);
        next.push(neighbor);
      }
    }
    frontier = next;
    if (frontier.length === 0) break;
  }

  return result;
}

export function buildCausalIndex(events: Event[], depth = 3): CausalIndex {
  const backwardLinks = new Map<string, string[]>();
  const forwardLinks = new Map<string, string[]>();

  for (const e of events) {
    backwardLinks.set(e.id, e.causal_links ?? []);
    for (const causeId of e.causal_links ?? []) {
      const list = forwardLinks.get(causeId);
      if (list) list.push(e.id);
      else forwardLinks.set(causeId, [e.id]);
    }
  }

  const neighborhoods = new Map<string, Cone>();
  for (const e of events) {
    neighborhoods.set(e.id, {
      backward: bfs(e.id, backwardLinks, depth),
      forward: bfs(e.id, forwardLinks, depth)
    });
  }

  // Map<string, Cone> structurally satisfies the CausalIndex interface.
  return neighborhoods as unknown as CausalIndex;
}

