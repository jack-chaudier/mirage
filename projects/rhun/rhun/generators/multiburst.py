"""Multi-burst temporal DAG generator with two high-weight temporal peaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhun.generators.base import GeneratorConfig, GraphGenerator
from rhun.schemas import CausalGraph, Event


@dataclass(frozen=True)
class MultiBurstConfig(GeneratorConfig):
    """Configuration for two-burst temporal/weight concentration."""

    burst_centers: tuple[float, float] = (0.2, 0.8)
    burst_width: float = 0.1
    burst_weight_boost: float = 3.0
    inter_burst_density: float = 0.3
    hub_fraction: float = 0.05
    causal_density: float = 0.08
    weight_noise_std: float = 0.01


class MultiBurstGenerator(GraphGenerator):
    """Generate causal DAGs with high-weight events around two temporal bursts."""

    def generate(self, config: GeneratorConfig) -> CausalGraph:
        if not isinstance(config, MultiBurstConfig):
            raise TypeError("MultiBurstGenerator expects MultiBurstConfig")
        if config.n_events < 0:
            raise ValueError("n_events must be >= 0")
        if config.n_actors < 0:
            raise ValueError("n_actors must be >= 0")
        if len(config.burst_centers) != 2:
            raise ValueError("burst_centers must contain exactly two centers")
        if config.burst_width <= 0.0:
            raise ValueError("burst_width must be > 0")
        if config.burst_weight_boost <= 0.0:
            raise ValueError("burst_weight_boost must be > 0")
        if config.hub_fraction < 0.0 or config.hub_fraction > 1.0:
            raise ValueError("hub_fraction must be in [0, 1]")
        if config.causal_density < 0.0 or config.causal_density > 1.0:
            raise ValueError("causal_density must be in [0, 1]")

        rng = np.random.default_rng(config.seed)
        actor_ids = tuple(f"actor_{i}" for i in range(config.n_actors))

        timestamps = self._generate_multiburst_timestamps(rng, config)
        weights = self._assign_weights(rng, timestamps, config)

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
                causal_density=config.causal_density,
            )

            events.append(
                Event(
                    id=event_id,
                    timestamp=float(timestamp),
                    weight=float(weight),
                    actors=participants,
                    causal_parents=parent_ids,
                    metadata={
                        "generator": "multiburst",
                        "region": self._region_for_timestamp(float(timestamp), config),
                    },
                )
            )

        return CausalGraph(
            events=tuple(events),
            actors=frozenset(actor_ids),
            seed=config.seed,
            metadata={
                "generator": "multiburst",
                "config": {
                    "n_events": config.n_events,
                    "n_actors": config.n_actors,
                    "seed": config.seed,
                    "burst_centers": tuple(float(center) for center in config.burst_centers),
                    "burst_width": config.burst_width,
                    "burst_weight_boost": config.burst_weight_boost,
                    "inter_burst_density": config.inter_burst_density,
                    "hub_fraction": config.hub_fraction,
                    "causal_density": config.causal_density,
                    "weight_noise_std": config.weight_noise_std,
                },
            },
        )

    @staticmethod
    def _sorted_centers(config: MultiBurstConfig) -> tuple[float, float]:
        left, right = sorted(float(c) for c in config.burst_centers)
        return left, right

    @staticmethod
    def _window(center: float, width: float) -> tuple[float, float]:
        return max(0.0, center - width), min(1.0, center + width)

    @classmethod
    def _burst_windows(cls, config: MultiBurstConfig) -> tuple[tuple[float, float], tuple[float, float]]:
        left_center, right_center = cls._sorted_centers(config)
        width = float(config.burst_width)
        return cls._window(left_center, width), cls._window(right_center, width)

    @classmethod
    def _interburst_window(cls, config: MultiBurstConfig) -> tuple[float, float] | None:
        (_b1_start, b1_end), (b2_start, _b2_end) = cls._burst_windows(config)
        if b1_end >= b2_start:
            return None
        return (b1_end, b2_start)

    @classmethod
    def _generate_multiburst_timestamps(
        cls,
        rng: np.random.Generator,
        config: MultiBurstConfig,
    ) -> np.ndarray:
        if config.n_events == 0:
            return np.asarray([], dtype=float)

        burst1, burst2 = cls._burst_windows(config)
        inter = cls._interburst_window(config)

        len_b1 = max(0.0, burst1[1] - burst1[0])
        len_b2 = max(0.0, burst2[1] - burst2[0])
        len_inter = max(0.0, (inter[1] - inter[0])) if inter is not None else 0.0

        density_ratio = max(0.0, float(config.inter_burst_density))

        mass_b1 = len_b1
        mass_b2 = len_b2
        mass_inter = density_ratio * len_inter

        total_mass = mass_b1 + mass_b2 + mass_inter
        if total_mass <= 1e-12:
            # Degenerate config: fall back to uniform timeline sampling.
            return np.sort(rng.uniform(0.0, 1.0, size=config.n_events))

        probs = np.asarray([mass_b1, mass_b2, mass_inter], dtype=float) / total_mass
        counts = rng.multinomial(config.n_events, probs)

        samples: list[np.ndarray] = []

        c1 = int(counts[0])
        if c1 > 0:
            if len_b1 <= 1e-12:
                samples.append(np.full(c1, fill_value=(burst1[0] + burst1[1]) / 2.0))
            else:
                samples.append(rng.uniform(burst1[0], burst1[1], size=c1))

        c2 = int(counts[1])
        if c2 > 0:
            if len_b2 <= 1e-12:
                samples.append(np.full(c2, fill_value=(burst2[0] + burst2[1]) / 2.0))
            else:
                samples.append(rng.uniform(burst2[0], burst2[1], size=c2))

        c_inter = int(counts[2])
        if c_inter > 0:
            if inter is None or len_inter <= 1e-12:
                samples.append(rng.uniform(0.0, 1.0, size=c_inter))
            else:
                samples.append(rng.uniform(inter[0], inter[1], size=c_inter))

        timestamps = np.concatenate(samples) if samples else np.asarray([], dtype=float)
        return np.sort(np.clip(timestamps.astype(float), 0.0, 1.0))

    @classmethod
    def _assign_weights(
        cls,
        rng: np.random.Generator,
        timestamps: np.ndarray,
        config: MultiBurstConfig,
    ) -> np.ndarray:
        n = timestamps.size
        if n == 0:
            return np.asarray([], dtype=float)

        burst1, burst2 = cls._burst_windows(config)
        in_burst1 = (timestamps >= burst1[0]) & (timestamps <= burst1[1])
        in_burst2 = (timestamps >= burst2[0]) & (timestamps <= burst2[1])
        in_burst = in_burst1 | in_burst2

        base = rng.uniform(0.1, 1.0, size=n)
        weights = base * np.where(in_burst, float(config.burst_weight_boost), 1.0)

        noise_std = max(0.0, float(config.weight_noise_std))
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=n)
            weights = weights + noise

        return np.maximum(weights, 1e-6)

    @classmethod
    def _region_for_timestamp(cls, timestamp: float, config: MultiBurstConfig) -> str:
        burst1, burst2 = cls._burst_windows(config)
        if burst1[0] <= timestamp <= burst1[1]:
            return "burst1"
        if burst2[0] <= timestamp <= burst2[1]:
            return "burst2"

        inter = cls._interburst_window(config)
        if inter is not None and inter[0] <= timestamp <= inter[1]:
            return "inter"
        return "outside"

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

        age = timestamps[event_index] - timestamps[earlier_indices]
        recency = 1.0 / (1.0 + age)
        parent_scores = (weights[earlier_indices] + 1e-9) * recency
        probs = parent_scores / parent_scores.sum()

        n_links = int(rng.integers(1, min(3, event_index) + 1))
        chosen = rng.choice(earlier_indices, size=n_links, replace=False, p=probs)
        chosen = np.sort(chosen)
        return tuple(event_ids[int(idx)] for idx in chosen)
