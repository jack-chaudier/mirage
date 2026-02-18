"""Compositional context algebra for endogenous-pivot sequential systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, inf, isinf
from typing import Iterable

from rhun.schemas import CausalGraph, Event


@dataclass(frozen=True)
class PivotCandidate:
    """A pivot candidate with enough metadata for compositional updates."""

    event_id: str
    weight: float
    timestamp: float
    prefix_count: int
    j_dev: int


@dataclass(frozen=True)
class ResourceCounters:
    """Resource counters needed for failure diagnostics and composition."""

    slots_used: int
    dev_count: int
    first_time: float | None
    last_time: float | None
    prefix_count: int
    setup_count: int
    post_count: int


@dataclass(frozen=True)
class ContextState:
    """
    Context state tuple:
        (q, r, Π, t_bounds)
    """

    q: str
    r: ResourceCounters
    pivots: tuple[PivotCandidate, ...]
    t_bounds: tuple[float | None, float | None]
    metadata: dict = field(default_factory=dict)


def development_eligible_count(prefix_count: int) -> int:
    """All-actor development-eligible count before a pivot."""
    n_setup = ceil(0.2 * prefix_count) if prefix_count > 0 else 0
    return max(0, prefix_count - n_setup)


def _sorted_events(events: Iterable[Event]) -> list[Event]:
    return sorted(events, key=lambda event: (float(event.timestamp), event.id))


def _pivot_candidates(events: list[Event], focal_actor: str, M: int) -> tuple[PivotCandidate, ...]:
    if M <= 0:
        return ()

    ranked: list[tuple[Event, int]] = []
    for idx, event in enumerate(events):
        if focal_actor in event.actors:
            ranked.append((event, idx))

    ranked.sort(
        key=lambda row: (float(row[0].weight), -float(row[0].timestamp), row[0].id),
        reverse=True,
    )

    pivots: list[PivotCandidate] = []
    for event, idx in ranked[:M]:
        prefix_count = int(idx)
        pivots.append(
            PivotCandidate(
                event_id=str(event.id),
                weight=float(event.weight),
                timestamp=float(event.timestamp),
                prefix_count=prefix_count,
                j_dev=development_eligible_count(prefix_count),
            )
        )
    return tuple(pivots)


def _phase_state_for_committed_pivot(slots_used: int, prefix_count: int) -> str:
    if slots_used <= 0:
        return "START"
    post_count = max(0, slots_used - prefix_count - 1)
    if post_count == 0:
        return "TURNING_POINT"
    return "RESOLUTION"


def build_context_state(
    events: Iterable[Event],
    focal_actor: str,
    M: int = 10,
) -> ContextState:
    """Build a context state for a concrete event segment."""
    ordered = _sorted_events(events)
    slots_used = len(ordered)
    t_min = float(ordered[0].timestamp) if ordered else None
    t_max = float(ordered[-1].timestamp) if ordered else None

    pivots = _pivot_candidates(ordered, focal_actor=focal_actor, M=M)
    if not pivots:
        q = "START" if slots_used == 0 else "DEVELOPMENT"
        r = ResourceCounters(
            slots_used=int(slots_used),
            dev_count=0,
            first_time=t_min,
            last_time=t_max,
            prefix_count=0,
            setup_count=0,
            post_count=max(0, slots_used),
        )
        return ContextState(
            q=q,
            r=r,
            pivots=pivots,
            t_bounds=(t_min, t_max),
            metadata={"has_focal_pivot": False},
        )

    committed = pivots[0]
    prefix_count = int(committed.prefix_count)
    dev_count = int(committed.j_dev)
    setup_count = max(0, prefix_count - dev_count)
    post_count = max(0, slots_used - prefix_count - 1)
    q = _phase_state_for_committed_pivot(slots_used=slots_used, prefix_count=prefix_count)

    r = ResourceCounters(
        slots_used=int(slots_used),
        dev_count=dev_count,
        first_time=t_min,
        last_time=t_max,
        prefix_count=prefix_count,
        setup_count=int(setup_count),
        post_count=int(post_count),
    )
    return ContextState(
        q=q,
        r=r,
        pivots=pivots,
        t_bounds=(t_min, t_max),
        metadata={"has_focal_pivot": True},
    )


def no_absorption_invariant(state: ContextState, k: int) -> bool:
    if k <= 0:
        return True
    if not state.pivots:
        return True
    return all(int(candidate.j_dev) >= int(k) for candidate in state.pivots)


def is_absorbing(state: ContextState, k: int) -> bool:
    if k <= 0:
        return False
    if not state.pivots:
        return False
    return any(int(candidate.j_dev) < int(k) for candidate in state.pivots)


def compose_context_states(
    left: ContextState,
    right: ContextState,
    M: int = 10,
) -> ContextState:
    """Compose context states: C(prefix) ⊗ C(suffix)."""
    left_slots = int(left.r.slots_used)
    right_slots = int(right.r.slots_used)
    slots_used = left_slots + right_slots

    merged: dict[str, PivotCandidate] = {}
    for candidate in left.pivots:
        prefix_count = int(candidate.prefix_count)
        merged[str(candidate.event_id)] = PivotCandidate(
            event_id=str(candidate.event_id),
            weight=float(candidate.weight),
            timestamp=float(candidate.timestamp),
            prefix_count=prefix_count,
            j_dev=development_eligible_count(prefix_count),
        )
    for candidate in right.pivots:
        prefix_count = left_slots + int(candidate.prefix_count)
        merged[str(candidate.event_id)] = PivotCandidate(
            event_id=str(candidate.event_id),
            weight=float(candidate.weight),
            timestamp=float(candidate.timestamp),
            prefix_count=prefix_count,
            j_dev=development_eligible_count(prefix_count),
        )

    pivots = tuple(
        sorted(
            merged.values(),
            key=lambda candidate: (
                float(candidate.weight),
                -float(candidate.timestamp),
                str(candidate.event_id),
            ),
            reverse=True,
        )[: max(0, int(M))]
    )

    first_candidates = [bound for bound in (left.t_bounds[0], right.t_bounds[0]) if bound is not None]
    last_candidates = [bound for bound in (left.t_bounds[1], right.t_bounds[1]) if bound is not None]
    t_min = min(first_candidates) if first_candidates else None
    t_max = max(last_candidates) if last_candidates else None

    if not pivots:
        q = "START" if slots_used == 0 else "DEVELOPMENT"
        r = ResourceCounters(
            slots_used=int(slots_used),
            dev_count=0,
            first_time=t_min,
            last_time=t_max,
            prefix_count=0,
            setup_count=0,
            post_count=max(0, slots_used),
        )
        return ContextState(
            q=q,
            r=r,
            pivots=(),
            t_bounds=(t_min, t_max),
            metadata={"composed": True, "has_focal_pivot": False},
        )

    committed = pivots[0]
    prefix_count = int(committed.prefix_count)
    dev_count = int(committed.j_dev)
    setup_count = max(0, prefix_count - dev_count)
    post_count = max(0, slots_used - prefix_count - 1)
    q = _phase_state_for_committed_pivot(slots_used=slots_used, prefix_count=prefix_count)

    r = ResourceCounters(
        slots_used=int(slots_used),
        dev_count=dev_count,
        first_time=t_min,
        last_time=t_max,
        prefix_count=prefix_count,
        setup_count=int(setup_count),
        post_count=int(post_count),
    )
    return ContextState(
        q=q,
        r=r,
        pivots=pivots,
        t_bounds=(t_min, t_max),
        metadata={"composed": True, "has_focal_pivot": True},
    )


def context_equivalent(
    left: ContextState,
    right: ContextState,
    compare_pivots: bool = True,
    tol: float = 1e-12,
) -> bool:
    if left.q != right.q:
        return False

    if left.r != right.r:
        return False

    if left.t_bounds[0] is None and right.t_bounds[0] is not None:
        return False
    if left.t_bounds[0] is not None and right.t_bounds[0] is None:
        return False
    if left.t_bounds[1] is None and right.t_bounds[1] is not None:
        return False
    if left.t_bounds[1] is not None and right.t_bounds[1] is None:
        return False
    if left.t_bounds[0] is not None and right.t_bounds[0] is not None:
        if abs(float(left.t_bounds[0]) - float(right.t_bounds[0])) > tol:
            return False
    if left.t_bounds[1] is not None and right.t_bounds[1] is not None:
        if abs(float(left.t_bounds[1]) - float(right.t_bounds[1])) > tol:
            return False

    if not compare_pivots:
        return True

    if len(left.pivots) != len(right.pivots):
        return False

    for a, b in zip(left.pivots, right.pivots, strict=True):
        if str(a.event_id) != str(b.event_id):
            return False
        if abs(float(a.weight) - float(b.weight)) > tol:
            return False
        if abs(float(a.timestamp) - float(b.timestamp)) > tol:
            return False
        if int(a.prefix_count) != int(b.prefix_count):
            return False
        if int(a.j_dev) != int(b.j_dev):
            return False
    return True


def bridge_budget_lower_bound(events: Iterable[Event], max_gap: float) -> int:
    """Lower bound on bridges needed to satisfy max-temporal-gap constraints."""
    if isinf(max_gap):
        return 0
    ordered = _sorted_events(events)
    if len(ordered) < 2:
        return 0

    budget = 0
    for left, right in zip(ordered[:-1], ordered[1:]):
        gap = float(right.timestamp - left.timestamp)
        if gap <= max_gap + 1e-12:
            continue
        budget += max(0, int(ceil(gap / max_gap) - 1))
    return int(budget)


def gap_violation_profile(events: Iterable[Event], max_gap: float) -> dict[str, float | int]:
    """
    Gap-violation profile over adjacent retained events.

    Returns count of oversized gaps, the maximum adjacent gap, and excess above max_gap.
    """
    ordered = _sorted_events(events)
    if len(ordered) < 2 or isinf(max_gap):
        return {
            "oversized_count": 0,
            "max_gap": 0.0,
            "max_oversized_excess": 0.0,
        }

    oversized = 0
    max_gap_seen = 0.0
    max_excess = 0.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        gap = float(right.timestamp - left.timestamp)
        if gap > max_gap + 1e-12:
            oversized += 1
            max_excess = max(max_excess, float(gap - max_gap))
        max_gap_seen = max(max_gap_seen, gap)
    return {
        "oversized_count": int(oversized),
        "max_gap": float(max_gap_seen),
        "max_oversized_excess": float(max_excess),
    }


def compress_events(
    events: Iterable[Event],
    focal_actor: str,
    k: int,
    M: int,
    target_size: int,
    strategy: str,
    max_gap: float = inf,
    min_length: int = 4,
) -> tuple[tuple[Event, ...], dict]:
    """
    Compress an event segment while tracking contract-violation pressure.

    Strategies:
    - "naive"
    - "bridge_preserving"
    - "contract_guarded"
    """
    if strategy not in {"naive", "bridge_preserving", "contract_guarded"}:
        raise ValueError(f"Unknown compression strategy: {strategy}")

    retained = _sorted_events(events)
    target = max(1, int(target_size))
    if len(retained) <= target:
        state = build_context_state(retained, focal_actor=focal_actor, M=M)
        return tuple(retained), {
            "strategy": strategy,
            "initial_size": len(retained),
            "target_size": target,
            "final_size": len(retained),
            "drops_attempted": 0,
            "drops_accepted": 0,
            "contract_violating_drop_attempts": 0,
            "contract_holds_final": no_absorption_invariant(state, k),
        }

    drops_attempted = 0
    drops_accepted = 0
    contract_violating_drop_attempts = 0
    rejected_contract = 0
    rejected_bridge = 0
    rejected_gap_guard = 0
    rejected_last_focal = 0

    while len(retained) > target:
        focal_count = sum(1 for event in retained if focal_actor in event.actors)
        candidates = sorted(
            enumerate(retained),
            key=lambda row: (float(row[1].weight), float(row[1].timestamp), str(row[1].id)),
        )
        dropped = False

        for idx, event in candidates:
            if focal_count <= 1 and focal_actor in event.actors:
                rejected_last_focal += 1
                continue

            trial = retained[:idx] + retained[idx + 1 :]
            drops_attempted += 1

            trial_state = build_context_state(trial, focal_actor=focal_actor, M=M)
            contract_ok = no_absorption_invariant(trial_state, k)
            if not contract_ok:
                contract_violating_drop_attempts += 1

            if strategy == "contract_guarded" and not contract_ok:
                rejected_contract += 1
                continue

            if strategy in {"bridge_preserving", "contract_guarded"} and not isinf(max_gap):
                bridge_lb = bridge_budget_lower_bound(trial, max_gap=max_gap)
                bridge_slack = max(0, len(trial) - int(min_length))
                if bridge_lb > bridge_slack:
                    rejected_bridge += 1
                    continue

            # Calibrated gap guard: refuse drops that yield any hard gap violation.
            if strategy == "contract_guarded" and not isinf(max_gap):
                trial_gap_profile = gap_violation_profile(trial, max_gap=max_gap)
                if int(trial_gap_profile["oversized_count"]) > 0:
                    rejected_gap_guard += 1
                    continue

            retained = trial
            drops_accepted += 1
            dropped = True
            break

        if not dropped:
            break

    final_state = build_context_state(retained, focal_actor=focal_actor, M=M)
    return tuple(retained), {
        "strategy": strategy,
        "initial_size": len(_sorted_events(events)),
        "target_size": target,
        "final_size": len(retained),
        "drops_attempted": int(drops_attempted),
        "drops_accepted": int(drops_accepted),
        "contract_violating_drop_attempts": int(contract_violating_drop_attempts),
        "rejected_contract": int(rejected_contract),
        "rejected_bridge": int(rejected_bridge),
        "rejected_gap_guard": int(rejected_gap_guard),
        "rejected_last_focal": int(rejected_last_focal),
        "gap_profile_final": gap_violation_profile(retained, max_gap=max_gap),
        "contract_holds_final": bool(no_absorption_invariant(final_state, k)),
    }


def induced_subgraph(graph: CausalGraph, selected_event_ids: Iterable[str]) -> CausalGraph:
    """Create a deterministic induced subgraph used by compression experiments."""
    selected = set(str(event_id) for event_id in selected_event_ids)
    selected_events = []
    for event in graph.events:
        if event.id not in selected:
            continue
        kept_parents = tuple(parent_id for parent_id in event.causal_parents if parent_id in selected)
        selected_events.append(
            Event(
                id=event.id,
                timestamp=float(event.timestamp),
                weight=float(event.weight),
                actors=event.actors,
                causal_parents=kept_parents,
                metadata=event.metadata,
            )
        )

    return CausalGraph(
        events=tuple(selected_events),
        actors=graph.actors,
        seed=graph.seed,
        metadata=dict(graph.metadata),
    )


def detect_class_a(state: ContextState, k: int) -> bool:
    if k <= 0 or not state.pivots:
        return False
    return bool(int(state.pivots[0].j_dev) < int(k))


def detect_class_b(state: ContextState, k: int) -> bool:
    if k <= 0:
        return False
    return bool(
        int(state.r.prefix_count) >= int(k)
        and int(state.r.setup_count) > 0
        and int(state.r.dev_count) < int(k)
    )


def detect_class_c(state: ContextState, k: int) -> bool:
    if k <= 0 or not state.pivots:
        return False
    top = state.pivots[0]
    if int(top.j_dev) >= int(k):
        return False
    return any(int(candidate.j_dev) >= int(k) for candidate in state.pivots[1:])


def detect_class_d(
    state: ContextState,
    min_timespan_fraction: float,
    graph_duration: float,
    min_length: int = 4,
) -> bool:
    if graph_duration <= 0.0:
        return False
    if int(state.r.slots_used) < int(min_length):
        return False
    low, high = state.t_bounds
    if low is None or high is None:
        return False
    span_fraction = max(0.0, float(high) - float(low)) / float(graph_duration)
    return bool(span_fraction + 1e-12 < float(min_timespan_fraction))
