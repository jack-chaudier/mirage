from __future__ import annotations

from collections import defaultdict

from .schemas import GuardResult, SceneProposal, Violation


class MirageGuard:
    """Hard-invariant guard for scene proposals."""

    def validate_scene(self, proposal: SceneProposal, engine) -> GuardResult:
        violations: list[Violation] = []
        actor_event_locations: dict[str, set[str]] = defaultdict(set)

        for idx, event in enumerate(proposal.events):
            actor = engine.get_character(event.actor_id)
            event_id = event.id

            if actor is None or not actor.alive:
                violations.append(
                    Violation(
                        type="dead_character_action",
                        description=f"Actor {event.actor_id} is unavailable for action.",
                        severity="hard",
                        event_index=idx,
                        event_id=event_id,
                    )
                )
                continue

            resolved_location = event.location_id or actor.location
            actor_event_locations[event.actor_id].add(resolved_location)
            if len(actor_event_locations[event.actor_id]) > 1:
                violations.append(
                    Violation(
                        type="location_conflict",
                        description=(
                            f"Actor {event.actor_id} appears in multiple locations in one proposal: "
                            f"{sorted(actor_event_locations[event.actor_id])}"
                        ),
                        severity="hard",
                        event_index=idx,
                        event_id=event_id,
                    )
                )

            for secret_id in event.required_knowledge_ids:
                if secret_id not in actor.known_secrets and not engine.is_secret_revealed(secret_id):
                    violations.append(
                        Violation(
                            type="impossible_knowledge",
                            description=(
                                f"{event.actor_id} references {secret_id} without having learned it."
                            ),
                            severity="hard",
                            event_index=idx,
                            event_id=event_id,
                        )
                    )

            for secret_id in event.regresses_secret_ids:
                if engine.is_secret_revealed(secret_id):
                    violations.append(
                        Violation(
                            type="secret_regression",
                            description=f"Secret {secret_id} cannot be un-revealed once public.",
                            severity="hard",
                            event_index=idx,
                            event_id=event_id,
                        )
                    )

            for target_id in event.target_ids:
                target = engine.get_character(target_id)
                if target is None or not target.alive:
                    violations.append(
                        Violation(
                            type="dead_character_action",
                            description=f"Target {target_id} is unavailable.",
                            severity="hard",
                            event_index=idx,
                            event_id=event_id,
                        )
                    )
                    continue

                target_location = target.location
                if not engine.are_adjacent(resolved_location, target_location):
                    violations.append(
                        Violation(
                            type="interaction_distance",
                            description=(
                                f"{event.actor_id} at {resolved_location} cannot interact with "
                                f"{target_id} at {target_location}."
                            ),
                            severity="hard",
                            event_index=idx,
                            event_id=event_id,
                        )
                    )

            if event.pivot_actor_id and engine.state.active_pivot_actor:
                if event.pivot_actor_id != engine.state.active_pivot_actor:
                    # Guard against abrupt pivot substitution unless scene introduces
                    # evidence through an explicit reveal in this same event.
                    has_reveal = bool(event.reveals_secret_ids)
                    if not has_reveal:
                        violations.append(
                            Violation(
                                type="pivot_shift",
                                description=(
                                    f"Pivot changed from {engine.state.active_pivot_actor} "
                                    f"to {event.pivot_actor_id} without new evidence."
                                ),
                                severity="hard",
                                event_index=idx,
                                event_id=event_id,
                            )
                        )

        valid = not any(v.severity == "hard" for v in violations)
        return GuardResult(valid=valid, violations=violations, repaired=False)

    def attempt_repair(self, proposal: SceneProposal, engine) -> tuple[SceneProposal, GuardResult]:
        first_pass = self.validate_scene(proposal=proposal, engine=engine)
        if first_pass.valid:
            return proposal, first_pass

        repaired = proposal.model_copy(deep=True)

        drop_indices: set[int] = set()
        for violation in first_pass.violations:
            idx = violation.event_index
            if idx is None or idx >= len(repaired.events):
                continue

            event = repaired.events[idx]
            if violation.type in {"dead_character_action", "secret_regression", "pivot_shift", "location_conflict"}:
                drop_indices.add(idx)
            elif violation.type == "impossible_knowledge":
                event.required_knowledge_ids = []
            elif violation.type == "interaction_distance":
                actor = engine.get_character(event.actor_id)
                event.target_ids = []
                if actor is not None:
                    event.location_id = actor.location
            elif violation.type == "location_conflict":
                actor = engine.get_character(event.actor_id)
                if actor is not None:
                    event.location_id = actor.location

        if drop_indices:
            repaired.events = [
                event for idx, event in enumerate(repaired.events) if idx not in drop_indices
            ]

        second_pass = self.validate_scene(proposal=repaired, engine=engine)
        second_pass.repaired = second_pass.valid
        return repaired, second_pass
