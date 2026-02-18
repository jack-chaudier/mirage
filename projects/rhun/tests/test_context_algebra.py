from __future__ import annotations

from rhun.schemas import Event
from rhun.theory.context_algebra import (
    bridge_budget_lower_bound,
    build_context_state,
    compress_events,
    compose_context_states,
    no_absorption_invariant,
)


def _events_for_assoc() -> tuple[Event, ...]:
    return (
        Event(id="e0", timestamp=0.00, weight=0.10, actors=frozenset({"actor_0"})),
        Event(id="e1", timestamp=0.10, weight=0.20, actors=frozenset({"actor_0"})),
        Event(id="e2", timestamp=0.20, weight=0.95, actors=frozenset({"actor_0"})),
        Event(id="e3", timestamp=0.30, weight=0.30, actors=frozenset({"actor_0"})),
        Event(id="e4", timestamp=0.40, weight=0.40, actors=frozenset({"actor_0"})),
        Event(id="e5", timestamp=0.50, weight=0.50, actors=frozenset({"actor_0"})),
    )


def test_context_composition_matches_ground_truth_qr() -> None:
    events = _events_for_assoc()
    a = events[:2]
    b = events[2:4]
    c = events[4:]

    state_a = build_context_state(a, focal_actor="actor_0", M=10)
    state_b = build_context_state(b, focal_actor="actor_0", M=10)
    state_c = build_context_state(c, focal_actor="actor_0", M=10)

    left = compose_context_states(compose_context_states(state_a, state_b, M=10), state_c, M=10)
    right = compose_context_states(state_a, compose_context_states(state_b, state_c, M=10), M=10)
    truth = build_context_state(events, focal_actor="actor_0", M=10)

    assert left.q == right.q == truth.q
    assert left.r == right.r == truth.r


def test_contract_guarded_compression_preserves_invariant() -> None:
    events = (
        Event(id="e0", timestamp=0.00, weight=0.10, actors=frozenset({"actor_0"})),
        Event(id="e1", timestamp=0.10, weight=0.11, actors=frozenset({"actor_0"})),
        Event(id="e2", timestamp=0.20, weight=0.95, actors=frozenset({"actor_0"})),
        Event(id="e3", timestamp=0.30, weight=0.30, actors=frozenset({"actor_0"})),
        Event(id="e4", timestamp=0.40, weight=0.40, actors=frozenset({"actor_0"})),
    )
    pre = build_context_state(events, focal_actor="actor_0", M=1)
    assert no_absorption_invariant(pre, k=1)

    compressed, diag = compress_events(
        events=events,
        focal_actor="actor_0",
        k=1,
        M=1,
        target_size=4,
        strategy="contract_guarded",
        max_gap=float("inf"),
        min_length=4,
    )
    post = build_context_state(compressed, focal_actor="actor_0", M=1)

    assert no_absorption_invariant(post, k=1)
    assert diag["contract_holds_final"] is True


def test_bridge_budget_lower_bound_detects_oversized_gap() -> None:
    events = (
        Event(id="e0", timestamp=0.00, weight=0.1, actors=frozenset({"actor_0"})),
        Event(id="e1", timestamp=0.40, weight=0.2, actors=frozenset({"actor_0"})),
    )
    # gap=0.40 with max_gap=0.10 requires ceil(4)-1 = 3 bridges.
    assert bridge_budget_lower_bound(events, max_gap=0.10) == 3
