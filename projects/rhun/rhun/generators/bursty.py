"""Bursty temporal DAG generator with front-loaded high-weight events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhun.generators.base import GeneratorConfig, GraphGenerator
from rhun.schemas import CausalGraph, Event


@dataclass(frozen=True)
class BurstyConfig(GeneratorConfig):
    epsilon: float = 0.3  # Front-loading parameter. THE experimental variable.
    n_bursts: int = 3  # Number of activity bursts
    burst_width: float = 0.1  # Width of each burst (fraction of total time)
    hub_fraction: float = 0.05  # Fraction of events that are hub events (involve all actors)
    causal_density: float = 0.08  # Base causal link probability
    weight_heavy_tail: float = 2.0  # Pareto shape parameter for front-loaded weights


class BurstyGenerator(GraphGenerator):
    """Generate bursty graphs with controllable temporal front-loading of high weights."""

    def generate(self, config: GeneratorConfig) -> CausalGraph:
        if not isinstance(config, BurstyConfig):
            raise TypeError("BurstyGenerator expects BurstyConfig")
        if not 0.0 <= config.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        rng = np.random.default_rng(config.seed)
        actor_ids = tuple(f"actor_{i}" for i in range(config.n_actors))

        timestamps = self._generate_bursty_timestamps(rng, config)
        weights = self._assign_weights(rng, timestamps, config)

        hub_count = int(round(config.hub_fraction * config.n_events))
        hub_indices: set[int] = set()
        if hub_count > 0:
            sorted_by_weight = np.argsort(weights)[::-1]
            hub_indices = set(int(idx) for idx in sorted_by_weight[:hub_count])

        event_ids = [f"e{i:04d}" for i in range(config.n_events)]
        events: list[Event] = []

        for i, (event_id, timestamp, weight) in enumerate(
            zip(event_ids, timestamps, weights, strict=True)
        ):
            if i in hub_indices and actor_ids:
                participants = frozenset(actor_ids)
            else:
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

            parent_ids = self._sample_preferential_parents(
                rng=rng,
                event_index=i,
                event_ids=event_ids,
                timestamps=timestamps,
                weights=weights,
                causal_density=config.causal_density,
            )

            events.append(
                Event(
                    id=event_id,
                    timestamp=float(timestamp),
                    weight=float(weight),
                    actors=participants,
                    causal_parents=parent_ids,
                    metadata={"generator": "bursty", "epsilon": config.epsilon},
                )
            )

        return CausalGraph(
            events=tuple(events),
            actors=frozenset(actor_ids),
            seed=config.seed,
            metadata={
                "generator": "bursty",
                "config": {
                    "n_events": config.n_events,
                    "n_actors": config.n_actors,
                    "seed": config.seed,
                    "epsilon": config.epsilon,
                    "n_bursts": config.n_bursts,
                    "burst_width": config.burst_width,
                    "hub_fraction": config.hub_fraction,
                    "causal_density": config.causal_density,
                    "weight_heavy_tail": config.weight_heavy_tail,
                },
            },
        )

    @staticmethod
    def _generate_bursty_timestamps(
        rng: np.random.Generator,
        config: BurstyConfig,
    ) -> np.ndarray:
        """Sample event times from burst-centered mixtures plus background noise."""
        burst_centers = np.sort(rng.uniform(0.05, 0.95, size=config.n_bursts))
        burst_mass = 0.8
        timestamps: list[float] = []

        std = max(1e-3, config.burst_width / 2.0)
        for _ in range(config.n_events):
            if rng.random() < burst_mass:
                center = float(rng.choice(burst_centers))
                timestamp = float(rng.normal(loc=center, scale=std))
            else:
                timestamp = float(rng.uniform(0.0, 1.0))
            timestamps.append(min(1.0, max(0.0, timestamp)))

        return np.sort(np.asarray(timestamps, dtype=float))

    @staticmethod
    def _assign_weights(
        rng: np.random.Generator,
        timestamps: np.ndarray,
        config: BurstyConfig,
    ) -> np.ndarray:
        """
        Assign weights with controllable front-loading.

        epsilon=0.0 -> uniform over the whole timeline (null model behavior).
        epsilon=1.0 -> high-tail weights constrained to the first ~10% of time.
        """
        if config.epsilon <= 0.0:
            return rng.uniform(0.0, 1.0, size=config.n_events)

        front_window = 1.0 - (0.9 * config.epsilon)  # maps 0->1.0, 1->0.1
        early_mask = timestamps <= front_window

        weights = rng.uniform(0.0, 0.5, size=config.n_events)
        n_early = int(np.sum(early_mask))
        if n_early > 0:
            pareto_raw = rng.pareto(config.weight_heavy_tail, size=n_early) + 1.0
            heavy = 0.5 + 0.5 * (1.0 - (1.0 / pareto_raw))
            weights[early_mask] = np.clip(heavy, 0.0, 1.0)

        return weights

    @staticmethod
    def _sample_preferential_parents(
        rng: np.random.Generator,
        event_index: int,
        event_ids: list[str],
        timestamps: np.ndarray,
        weights: np.ndarray,
        causal_density: float,
    ) -> tuple[str, ...]:
        if event_index == 0:
            return ()

        if rng.random() >= causal_density:
            return ()

        earlier_indices = np.arange(event_index)
        if earlier_indices.size == 0:
            return ()

        # Slight recency preference so very old events are less likely parents.
        age = timestamps[event_index] - timestamps[earlier_indices]
        recency = 1.0 / (1.0 + age)
        parent_scores = (weights[earlier_indices] + 1e-9) * recency
        probs = parent_scores / parent_scores.sum()

        n_links = int(rng.integers(1, min(3, event_index) + 1))
        chosen = rng.choice(earlier_indices, size=n_links, replace=False, p=probs)
        chosen = np.sort(chosen)
        return tuple(event_ids[int(idx)] for idx in chosen)
