"""Tally violation types for dev-present, non-prefix failure cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def violation_type(violation: str) -> str:
    return violation.split(":", maxsplit=1)[0].strip()


def main() -> None:
    path = Path(__file__).resolve().parent / "output" / "extraction_internals_diagnosis.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    cases = [
        case
        for case in data["results"]["per_case"]
        if case.get("mechanism_bucket") == "dev_present_other_violation"
    ]

    full_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()

    for case in cases:
        for violation in case["greedy_candidate"]["violations"]:
            full_counter[violation] += 1
            type_counter[violation_type(violation)] += 1

    print(f"cases={len(cases)}")
    print("violation_types:")
    for key, count in sorted(type_counter.items()):
        print(f"  {key}: {count}")

    print("full_violations:")
    for key, count in sorted(full_counter.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
