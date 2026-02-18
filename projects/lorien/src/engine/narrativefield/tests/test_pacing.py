from __future__ import annotations

from narrativefield.schema.agents import AgentState, GoalVector, PacingState
from narrativefield.schema.events import Event, EventMetrics, EventType
from narrativefield.schema.world import Location
from narrativefield.simulation import pacing


def _agent(agent_id: str = "a") -> AgentState:
    return AgentState(
        id=agent_id,
        name=agent_id,
        location="dining_table",
        goals=GoalVector(),
        flaws=[],
        pacing=PacingState(),
        emotional_state={},
        relationships={},
        beliefs={},
        alcohol_level=0.0,
        commitments=[],
    )


def _loc(privacy: float) -> Location:
    return Location(
        id="dining_table",
        name="Dining Table",
        privacy=privacy,
        capacity=6,
        adjacent=[],
        overhear_from=[],
        overhear_probability=0.0,
        description="",
    )


def test_catastrophe_potential_formula() -> None:
    a = _agent()
    a.pacing.stress = 0.5
    a.pacing.commitment = 0.8
    a.pacing.suppression_count = 4
    pot = pacing.catastrophe_potential(a)
    # base = 0.5 * 0.8^2 = 0.32; bonus = 4 * 0.03 = 0.12
    assert abs(pot - 0.44) < 1e-9


def test_check_catastrophe_respects_recovery_timer() -> None:
    a = _agent()
    a.pacing.stress = 0.9
    a.pacing.commitment = 0.9
    a.pacing.composure = 0.1
    a.pacing.recovery_timer = 2
    assert pacing.check_catastrophe(a) is False


def test_check_catastrophe_triggers_when_conditions_met() -> None:
    a = _agent()
    a.pacing.stress = 0.6
    a.pacing.commitment = 0.8  # 0.6 * 0.64 = 0.384
    a.pacing.composure = 0.2
    a.pacing.recovery_timer = 0
    assert pacing.check_catastrophe(a) is True


def test_check_catastrophe_respects_overridden_thresholds() -> None:
    a = _agent()
    a.pacing.stress = 0.5
    a.pacing.commitment = 0.8  # potential = 0.32
    a.pacing.composure = 0.28
    a.pacing.recovery_timer = 0

    assert pacing.check_catastrophe(a) is False
    assert pacing.check_catastrophe(a, catastrophe_threshold=0.30, composure_gate=0.29) is True


def test_apply_social_masking() -> None:
    a = _agent()
    a.pacing.composure = 0.5
    public = _loc(privacy=0.1)
    private = _loc(privacy=0.7)

    assert pacing.apply_social_masking(1.0, a, public) == 0.5
    assert pacing.apply_social_masking(1.0, a, private) == 1.0

    a.pacing.composure = 0.3
    assert pacing.apply_social_masking(1.0, a, public) == 1.0


def test_update_recovery_timer_decrements_after_setting() -> None:
    a = _agent()
    a.pacing.recovery_timer = 0

    conflict = Event(
        id="evt_0001",
        sim_time=0.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CONFLICT,
        source_agent=a.id,
        target_agents=["b"],
        location_id=a.location,
        causal_links=[],
        deltas=[],
        description="",
        metrics=EventMetrics(),
    )
    assert pacing.update_recovery_timer(a, [conflict]) == 3  # major sets 4, then -1


def test_update_suppression_count_increments_only_when_stress_high_and_no_dramatic_action() -> None:
    a = _agent()
    a.pacing.stress = 0.61
    a.pacing.suppression_count = 2
    assert pacing.update_suppression_count(a, []) == 3

    dramatic = Event(
        id="evt_0002",
        sim_time=0.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CONFIDE,
        source_agent=a.id,
        target_agents=["b"],
        location_id=a.location,
        causal_links=[],
        deltas=[],
        description="",
        metrics=EventMetrics(),
    )
    assert pacing.update_suppression_count(a, [dramatic]) == 0

    # Below the stress threshold, the suppression counter persists until a release.
    a.pacing.stress = 0.20
    a.pacing.suppression_count = 3
    assert pacing.update_suppression_count(a, []) == 3
