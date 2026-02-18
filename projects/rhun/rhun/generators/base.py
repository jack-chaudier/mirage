"""Abstract base for causal graph generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rhun.schemas import CausalGraph


@dataclass(frozen=True)
class GeneratorConfig:
    """Common parameters for all generators."""

    n_events: int = 200
    n_actors: int = 6
    seed: int = 42
    # Subclasses add their own parameters


class GraphGenerator(ABC):
    @abstractmethod
    def generate(self, config: GeneratorConfig) -> CausalGraph:
        ...
