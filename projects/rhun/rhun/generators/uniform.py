"""Uniform random DAG generator (null model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhun.generators.base import GeneratorConfig, GraphGenerator
from rhun.schemas import CausalGraph, Event


@dataclass(frozen=True)
class UniformConfig(GeneratorConfig):
    """Configuration for the uniform random DAG generator."""

    causal_density: float = 0.05
    causal_window: float = 0.2
    weight_distribution: str = "uniform"  # "uniform", "normal", "exponential"


class UniformGenerator(GraphGenerator):
    """Generate temporally uniform causal DAGs with configurable weight draws."""

    def generate(self, config: GeneratorConfig) -> CausalGraph:
        if not isinstance(config, UniformConfig):
            raise TypeError("UniformGenerator expects UniformConfig")

        rng = np.random.default_rng(config.seed)
        actor_ids = tuple(f"actor_{i}" for i in range(config.n_actors))

        timestamps = np.sort(rng.uniform(0.0, 1.0, size=config.n_events))
        weights = self._draw_weights(rng, config)

        events: list[Event] = []
        event_ids = [f"e{i:04d}" for i in range(config.n_events)]

        for i, (event_id, timestamp, weight) in enumerate(
            zip(event_ids, timestamps, weights, strict=True)
        ):
            n_participants = int(
                rng.integers(1, min(3, config.n_actors) + 1)
                if config.n_actors > 0
                else 0
            )
            participants = (
                frozenset(rng.choice(actor_ids, size=n_participants, replace=False).tolist())
                if actor_ids
                else frozenset()
            )

            parent_ids = self._sample_parents(
                rng=rng,
                event_index=i,
                timestamps=timestamps,
                event_ids=event_ids,
                causal_density=config.causal_density,
                causal_window=config.causal_window,
            )

            events.append(
                Event(
                    id=event_id,
                    timestamp=float(timestamp),
                    weight=float(weight),
                    actors=participants,
                    causal_parents=parent_ids,
                    metadata={"generator": "uniform"},
                )
            )

        return CausalGraph(
            events=tuple(events),
            actors=frozenset(actor_ids),
            seed=config.seed,
            metadata={
                "generator": "uniform",
                "config": {
                    "n_events": config.n_events,
                    "n_actors": config.n_actors,
                    "seed": config.seed,
                    "causal_density": config.causal_density,
                    "causal_window": config.causal_window,
                    "weight_distribution": config.weight_distribution,
                },
            },
        )

    @staticmethod
    def _draw_weights(rng: np.random.Generator, config: UniformConfig) -> np.ndarray:
        if config.weight_distribution == "uniform":
            weights = rng.uniform(0.0, 1.0, size=config.n_events)
        elif config.weight_distribution == "normal":
            weights = rng.normal(loc=0.5, scale=0.2, size=config.n_events)
        elif config.weight_distribution == "exponential":
            weights = rng.exponential(scale=0.3, size=config.n_events)
        else:
            raise ValueError(
                "weight_distribution must be one of: uniform, normal, exponential"
            )
        return np.clip(weights, 0.0, 1.0)

    @staticmethod
    def _sample_parents(
        rng: np.random.Generator,
        event_index: int,
        timestamps: np.ndarray,
        event_ids: list[str],
        causal_density: float,
        causal_window: float,
    ) -> tuple[str, ...]:
        if event_index == 0:
            return ()

        current_time = timestamps[event_index]
        candidates = [
            idx
            for idx in range(event_index)
            if (current_time - timestamps[idx]) <= causal_window
        ]
        if not candidates:
            return ()

        mask = rng.random(len(candidates)) < causal_density
        selected = [candidates[idx] for idx, keep in enumerate(mask) if keep]
        if len(selected) > 3:
            selected = rng.choice(selected, size=3, replace=False).tolist()

        selected.sort()
        return tuple(event_ids[idx] for idx in selected)
