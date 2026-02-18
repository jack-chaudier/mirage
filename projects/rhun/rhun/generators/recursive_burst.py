"""Recursive multi-scale burst generator (bursts-within-bursts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhun.generators.base import GeneratorConfig, GraphGenerator
from rhun.schemas import CausalGraph, Event


@dataclass(frozen=True)
class RecursiveBurstConfig(GeneratorConfig):
    """Configuration for hierarchical burst generation."""

    epsilon: float = 0.6
    levels: int = 3
    branch_factor: int = 2
    base_width: float = 0.30
    burst_mass: float = 0.85
    causal_density: float = 0.08
    hub_fraction: float = 0.05
    weight_heavy_tail: float = 2.0


class RecursiveBurstGenerator(GraphGenerator):
    """Generate causal DAGs with recursive temporal concentration."""

    def generate(self, config: GeneratorConfig) -> CausalGraph:
        if not isinstance(config, RecursiveBurstConfig):
            raise TypeError("RecursiveBurstGenerator expects RecursiveBurstConfig")
        if config.n_events < 0:
            raise ValueError("n_events must be >= 0")
        if config.n_actors < 0:
            raise ValueError("n_actors must be >= 0")
        if not 0.0 <= config.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if config.levels <= 0:
            raise ValueError("levels must be >= 1")
        if config.branch_factor <= 0:
            raise ValueError("branch_factor must be >= 1")
        if config.base_width <= 0.0:
            raise ValueError("base_width must be > 0")

        rng = np.random.default_rng(config.seed)
        actor_ids = tuple(f"actor_{i}" for i in range(config.n_actors))

        centers_by_level = self._hierarchical_centers(
            rng=rng,
            levels=config.levels,
            branch_factor=config.branch_factor,
            base_width=float(config.base_width),
        )
        timestamps = self._generate_timestamps(rng=rng, config=config, centers_by_level=centers_by_level)
        weights = self._assign_weights(
            rng=rng,
            timestamps=timestamps,
            config=config,
            centers_by_level=centers_by_level,
        )

        hub_count = int(round(config.hub_fraction * config.n_events))
        hub_indices: set[int] = set()
        if hub_count > 0 and config.n_events > 0:
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
                causal_density=float(config.causal_density),
            )

            events.append(
                Event(
                    id=event_id,
                    timestamp=float(timestamp),
                    weight=float(weight),
                    actors=participants,
                    causal_parents=parent_ids,
                    metadata={
                        "generator": "recursive_burst",
                        "epsilon": float(config.epsilon),
                        "levels": int(config.levels),
                        "branch_factor": int(config.branch_factor),
                    },
                )
            )

        return CausalGraph(
            events=tuple(events),
            actors=frozenset(actor_ids),
            seed=config.seed,
            metadata={
                "generator": "recursive_burst",
                "config": {
                    "n_events": config.n_events,
                    "n_actors": config.n_actors,
                    "seed": config.seed,
                    "epsilon": config.epsilon,
                    "levels": config.levels,
                    "branch_factor": config.branch_factor,
                    "base_width": config.base_width,
                    "burst_mass": config.burst_mass,
                    "causal_density": config.causal_density,
                    "hub_fraction": config.hub_fraction,
                    "weight_heavy_tail": config.weight_heavy_tail,
                },
            },
        )

    @staticmethod
    def _hierarchical_centers(
        rng: np.random.Generator,
        levels: int,
        branch_factor: int,
        base_width: float,
    ) -> list[np.ndarray]:
        centers_by_level: list[np.ndarray] = []
        current = np.asarray([0.5], dtype=float)
        centers_by_level.append(current)

        for level in range(1, levels):
            children: list[float] = []
            offset_scale = base_width / (2.0**level)
            for parent in current:
                for _ in range(branch_factor):
                    offset = float(rng.normal(loc=0.0, scale=max(1e-3, offset_scale / 2.5)))
                    children.append(float(np.clip(parent + offset, 0.0, 1.0)))
            current = np.asarray(sorted(children), dtype=float)
            centers_by_level.append(current)
        return centers_by_level

    @staticmethod
    def _level_probs(levels: int, epsilon: float) -> np.ndarray:
        # Higher epsilon pushes more mass to finer levels.
        if levels <= 1:
            return np.asarray([1.0], dtype=float)
        raw = np.asarray([(1.0 - epsilon) + epsilon * (lvl / (levels - 1)) for lvl in range(levels)], dtype=float)
        raw = np.maximum(raw, 1e-9)
        return raw / raw.sum()

    @classmethod
    def _generate_timestamps(
        cls,
        rng: np.random.Generator,
        config: RecursiveBurstConfig,
        centers_by_level: list[np.ndarray],
    ) -> np.ndarray:
        if config.n_events == 0:
            return np.asarray([], dtype=float)

        n = int(config.n_events)
        timestamps = np.zeros(n, dtype=float)
        level_probs = cls._level_probs(config.levels, float(config.epsilon))
        level_choices = rng.choice(np.arange(config.levels), size=n, p=level_probs)

        for i in range(n):
            if rng.random() > float(config.burst_mass):
                timestamps[i] = float(rng.uniform(0.0, 1.0))
                continue

            level = int(level_choices[i])
            centers = centers_by_level[level]
            center = float(rng.choice(centers))
            std = max(1e-3, float(config.base_width) / (2.0 ** (level + 1)))
            timestamps[i] = float(np.clip(rng.normal(loc=center, scale=std), 0.0, 1.0))

        return np.sort(timestamps)

    @staticmethod
    def _assign_weights(
        rng: np.random.Generator,
        timestamps: np.ndarray,
        config: RecursiveBurstConfig,
        centers_by_level: list[np.ndarray],
    ) -> np.ndarray:
        n = int(timestamps.size)
        if n == 0:
            return np.asarray([], dtype=float)

        base = rng.uniform(0.05, 0.45, size=n)

        # Multi-scale burst affinity.
        affinity = np.zeros(n, dtype=float)
        for level, centers in enumerate(centers_by_level):
            sigma = max(1e-3, float(config.base_width) / (2.0 ** (level + 1)))
            if centers.size == 0:
                continue
            level_aff = np.zeros(n, dtype=float)
            for center in centers:
                dist2 = (timestamps - float(center)) ** 2
                level_aff = np.maximum(level_aff, np.exp(-0.5 * dist2 / (sigma * sigma)))
            affinity += level_aff / max(1, len(centers_by_level))

        # Front-loading control reused from bursty settings.
        epsilon = float(config.epsilon)
        front_window = 1.0 - (0.9 * epsilon)
        early_mask = timestamps <= front_window
        heavy = np.zeros(n, dtype=float)
        n_early = int(np.sum(early_mask))
        if n_early > 0:
            pareto_raw = rng.pareto(float(config.weight_heavy_tail), size=n_early) + 1.0
            heavy[early_mask] = 1.0 - (1.0 / pareto_raw)

        front_profile = np.clip(
            (front_window - timestamps) / max(1e-6, front_window),
            0.0,
            1.0,
        )
        weights = base + 0.30 * affinity
        weights += (0.15 + 0.55 * epsilon) * heavy
        weights += 0.45 * epsilon * front_profile
        if np.any(~early_mask):
            weights[~early_mask] *= max(0.35, 1.0 - 0.45 * epsilon)
        return np.clip(weights, 1e-6, 1.0)

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

        earlier = np.arange(event_index)
        if earlier.size == 0:
            return ()

        age = timestamps[event_index] - timestamps[earlier]
        recency = 1.0 / (1.0 + age)
        parent_scores = (weights[earlier] + 1e-9) * recency
        probs = parent_scores / parent_scores.sum()

        n_links = int(rng.integers(1, min(3, event_index) + 1))
        chosen = rng.choice(earlier, size=n_links, replace=False, p=probs)
        chosen = np.sort(chosen)
        return tuple(event_ids[int(idx)] for idx in chosen)
