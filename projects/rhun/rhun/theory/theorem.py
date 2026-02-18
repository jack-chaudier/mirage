"""
Prefix-Constraint Impossibility Theorem for Greedy Extraction.

THEOREM (informal):
    Let G be a temporally ordered causal graph with n events.
    Let w: Events -> R+ be a weight function.
    Let GREEDY be a search that selects the max-w event as the turning point.
    Let GRAMMAR require min_prefix_elements >= k development-phase events
    before the turning point.

    If the max-w event occurs at temporal position j (0-indexed in sorted order),
    and the search injects max-w into every candidate pool, then:

    GREEDY produces zero valid sequences when j < k
    (i.e., there are fewer than k events before the max-w element).

    More precisely: if the max-w event is in the first epsilon-fraction
    of the timeline (position j <= epsilon * n), and the grammar requires
    k >= 1 development elements before TP, then for epsilon < k/n,
    every candidate pool containing the max-w event will have the max-w
    event classified as TP with fewer than k preceding elements in
    the development phase.

COROLLARY (shear invariance):
    The failure is independent of how w is decomposed into per-actor
    components. Whether w is a global scalar or a projection of an
    actor-specific vector, the max-w element's temporal position alone
    determines failure.

COROLLARY (search invariance):
    The failure is independent of anchor selection strategy, because
    injection guarantees the max-w element enters every candidate pool
    regardless of which anchors are chosen.

ABSORBING STATE CHARACTERIZATION:
    Model the greedy search as a finite state machine where the state is
    (current_phase, n_selected, development_count). The monotonic phase
    rule means phase transitions are irreversible: once the search enters
    TURNING_POINT phase (by selecting the max-w event as TP), it cannot
    return to DEVELOPMENT.

    If development_count < k at the moment of TP assignment, the search
    enters an ABSORBING STATE: no sequence of future event additions can
    ever satisfy the prefix constraint. The search is deterministically
    trapped — not approximately bad, but provably unable to produce any
    valid completion.

    This is qualitatively different from standard greedy approximation
    failures (where greedy gets a suboptimal-but-feasible solution). Here,
    greedy produces zero feasible solutions. The prefix constraint creates
    the absorbing state; without it (k=0), no absorbing state exists.
"""

from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.schemas import CausalGraph, ExtractedSequence, Phase


def _actor_relevant_events(graph: CausalGraph, focal_actor: str):
    actor_events = graph.events_for_actor(focal_actor)
    return actor_events if actor_events else graph.events


def _argmax_event(events):
    return max(events, key=lambda event: (event.weight, -event.timestamp))


def check_precondition(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> dict:
    """
    Check whether the theorem's precondition holds for this graph/actor/grammar.

    Returns a dict with:
        - max_weight_position: float (normalized 0-1)
        - max_weight_index: int (0-indexed in temporal sort)
        - k: int (grammar.min_prefix_elements)
        - precondition_met: bool (max_weight is early enough to trigger failure)
        - predicted_failure: bool (theorem predicts greedy will fail)
        - events_before_max: int
    """
    if not graph.events:
        return {
            "max_weight_position": 0.0,
            "max_weight_index": -1,
            "k": grammar.min_prefix_elements,
            "precondition_met": False,
            "predicted_failure": False,
            "events_before_max": 0,
        }

    relevant = _actor_relevant_events(graph, focal_actor)
    max_weight_event = _argmax_event(relevant)
    max_weight_index = next(
        (idx for idx, event in enumerate(graph.events) if event.id == max_weight_event.id),
        -1,
    )
    n = len(graph.events)
    max_weight_position = (max_weight_index / (n - 1)) if n > 1 and max_weight_index >= 0 else 0.0

    k = grammar.min_prefix_elements
    events_before_max = max(0, max_weight_index)
    precondition_met = (k > 0) and (events_before_max < k)

    return {
        "max_weight_position": max_weight_position,
        "max_weight_index": max_weight_index,
        "k": k,
        "precondition_met": precondition_met,
        "predicted_failure": precondition_met,
        "events_before_max": events_before_max,
    }


def verify_prediction(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    actual_result: ExtractedSequence,
) -> dict:
    """
    Compare theorem prediction against actual extraction result.

    Returns:
        - predicted_failure: bool
        - actual_failure: bool (not actual_result.valid)
        - prediction_correct: bool
        - mismatch_reason: str | None
    """
    precondition = check_precondition(graph, focal_actor, grammar)
    predicted_failure = bool(precondition["predicted_failure"])
    actual_failure = not actual_result.valid
    prediction_correct = predicted_failure == actual_failure

    mismatch_reason: str | None
    if prediction_correct:
        mismatch_reason = None
    elif predicted_failure and not actual_failure:
        mismatch_reason = "predicted_failure_but_valid_sequence_found"
    else:
        mismatch_reason = "predicted_success_but_no_valid_sequence_found"

    return {
        "predicted_failure": predicted_failure,
        "actual_failure": actual_failure,
        "prediction_correct": prediction_correct,
        "mismatch_reason": mismatch_reason,
    }


def diagnose_absorption(
    sequence: ExtractedSequence,
    grammar: GrammarConfig,
) -> dict:
    """
    Analyze whether the extraction entered an absorbing state.

    Walks the phase assignments in order and detects the moment
    (if any) where the search irrecoverably committed to failure.

    Returns:
        - absorbed: bool (did the search enter an absorbing state?)
        - absorption_step: int | None (0-indexed position where TP was assigned
          with insufficient preceding development elements)
        - development_at_absorption: int (development count when TP was assigned)
        - required_development: int (grammar.min_prefix_elements)
        - events_after_absorption: int (how many events were added after
          the search was already doomed — wasted computation)
    """
    required = grammar.min_prefix_elements

    if required <= 0:
        return {
            "absorbed": False,
            "absorption_step": None,
            "development_at_absorption": 0,
            "required_development": required,
            "events_after_absorption": 0,
        }

    for idx, phase in enumerate(sequence.phases):
        if phase != Phase.TURNING_POINT:
            continue

        development_count = sum(
            1 for prefix_phase in sequence.phases[:idx] if prefix_phase == Phase.DEVELOPMENT
        )
        if development_count < required:
            return {
                "absorbed": True,
                "absorption_step": idx,
                "development_at_absorption": development_count,
                "required_development": required,
                "events_after_absorption": max(0, len(sequence.events) - idx - 1),
            }

        return {
            "absorbed": False,
            "absorption_step": None,
            "development_at_absorption": development_count,
            "required_development": required,
            "events_after_absorption": 0,
        }

    return {
        "absorbed": False,
        "absorption_step": None,
        "development_at_absorption": 0,
        "required_development": required,
        "events_after_absorption": 0,
    }
