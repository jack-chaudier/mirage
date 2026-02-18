"""Common experiment infrastructure: deterministic, verification-first, $0 compute."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "output"


@dataclass
class ExperimentMetadata:
    name: str
    timestamp: str
    runtime_seconds: float
    n_graphs: int
    n_extractions: int
    seed_range: tuple[int, int]
    parameters: dict


class ExperimentTimer:
    """Simple wall-clock timer for experiment metadata."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_results(
    name: str,
    data: dict,
    metadata: ExperimentMetadata,
    summary_formatter: Callable[[dict, ExperimentMetadata], str] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "metadata": asdict(metadata),
        "results": data,
    }
    json_path = OUTPUT_DIR / f"{name}.json"
    md_path = OUTPUT_DIR / f"{name}_summary.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    if summary_formatter is not None:
        markdown = summary_formatter(data, metadata)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)

    print(f"Results saved to {json_path}")
