# Pre-check on oracle scaling:
# - Reviewed `rhun/extraction/search.py::oracle_extract` before writing this script.
# - It is not full combinatorial subsequence enumeration.
# - It evaluates one bounded candidate per eligible forced turning point and scores/validates it.
# - Quick benchmark at n_events=200 (20 seeds, epsilon=0.95, focal actor actor_0):
#   mean ~0.0044s and max ~0.0053s per graph for oracle_extract.
# - Conclusion: n_events=200 is safely tractable; no reduction required.

from __future__ import annotations

from rhun.experiments.oracle_diff import run_oracle_diff


if __name__ == "__main__":
    result = run_oracle_diff()
    print(result["results"]["table"])
