"""Pool construction strategies for candidate extraction sequences."""

from __future__ import annotations

from collections import deque

from rhun.schemas import CausalGraph


def bfs_pool(
    graph: CausalGraph,
    anchor_id: str,
    focal_actor: str,
    max_depth: int = 3,
) -> set[str]:
    del focal_actor  # unused by BFS-only baseline

    event = graph.event_by_id(anchor_id)
    if event is None:
        return set()

    adjacency = graph.build_adjacency()
    reverse = graph.build_reverse_adjacency()

    visited: set[str] = {anchor_id}
    queue: deque[tuple[str, int]] = deque([(anchor_id, 0)])

    while queue:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = adjacency.get(current_id, []) + reverse.get(current_id, [])
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, depth + 1))

    return visited


def _top_actor_events(
    graph: CausalGraph,
    focal_actor: str,
    injection_top_n: int,
) -> list[str]:
    actor_events = sorted(
        graph.events_for_actor(focal_actor),
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    return [event.id for event in actor_events[:injection_top_n]]


def injection_pool(
    graph: CausalGraph,
    anchor_id: str,
    focal_actor: str,
    max_depth: int = 3,
    injection_top_n: int = 40,
) -> set[str]:
    pool = bfs_pool(graph, anchor_id, focal_actor, max_depth=max_depth)
    pool.update(_top_actor_events(graph, focal_actor, injection_top_n))
    return pool


def filtered_injection_pool(
    graph: CausalGraph,
    anchor_id: str,
    focal_actor: str,
    max_depth: int = 3,
    injection_top_n: int = 40,
    min_position: float = 0.0,
) -> set[str]:
    pool = bfs_pool(graph, anchor_id, focal_actor, max_depth=max_depth)

    focal_events = sorted(
        graph.events_for_actor(focal_actor),
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    if not focal_events:
        return pool

    focal_weights = sorted(event.weight for event in focal_events)
    median_weight = focal_weights[len(focal_weights) // 2]

    injected = 0
    for event in focal_events:
        if injected >= injection_top_n:
            break
        if graph.global_position(event) < min_position:
            continue
        if event.weight < median_weight:
            continue
        pool.add(event.id)
        injected += 1

    return pool
