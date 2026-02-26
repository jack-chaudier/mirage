# tropical-compactor

`stdio` MCP server that applies L2 tropical-algebra context compaction before context-window eviction.

## What It Does

This server provides three MCP tools:

- `inspect(messages, k)`: runs the L2 frontier scan and reports feasibility plus load-bearing chunks.
- `compact(messages, token_budget, policy, k)`: compacts context with either plain recency or `l2_guarded` protection.
- `tag(messages)`: shows heuristic role classification (`pivot`, `predecessor`, `noise`).

Additional automation helpers:

- `inspect_horizon(messages, k_max)`: shows which `k` slots are feasible from `0..k_max`.
- `compact_auto(messages, token_budget, k_target, mode)`: picks a feasible `k` automatically (`adaptive`) or fails closed (`strict`).
- `retention_floor(messages, k, horizon, failure_prob)`: estimates an operational retention floor from predecessor density.

## Math Contract

The core composition follows the L2 shift-and-max rule:

`W_result[j] = max(W_A[j], W_B[max(0, j - d_A)])`

This is the same rule documented in `papers/sources/paper_i/main.tex` (`eq:shift-and-max`, section "L2: The Tropical Vector").

Implementation details:

- `algebra.py` executes the L2 scan and tracks full provenance (pivot + predecessor IDs).
- `protected_set(...)` returns the exact chunks that must survive for a feasible `k`-arc.
- `compactor.py` enforces protected-first retention under token budgets and logs any contract breach.
- `l2_iterative_guarded` policy uses iterative removal to preserve `W[k]` (and optionally pivot identity) before allowing breaches.

## Semantics Clarifications

- The L1 scalar `d_pre` is not, in general, equal to the highest finite slot index in `W`.
  That equality only holds in restricted cases (for example, when a single pivot dominates all
  frontier slots). In the general case, different slots can be realized by different pivots.
- `contract_satisfied` (alias: `protection_satisfied`) means only that protected IDs were not
  dropped by the token budget. It does not by itself imply a valid `k`-arc exists.
- Use `guard_effective` for the strong operational check: it is true only when both
  `feasible == true` and `protection_satisfied == true`.

## Project Layout

```text
tropical-compactor/
├── server.py
├── algebra.py
├── tagger.py
├── compactor.py
├── pyproject.toml
├── README.md
└── tests/
```

## Install

```bash
cd /absolute/path/to/tropical-compactor
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Test

```bash
pytest
```

Comprehensive product validation:

```bash
./scripts/full_validation.sh
```

or run the functional checks only:

```bash
uv run python scripts/run_full_validation.py
```

## Run

```bash
python server.py
```

or with `uv`:

```bash
uv run server.py
```

## Claude Code Registration

```bash
claude mcp add tropical-compactor --scope user -- \
  uv --directory /absolute/path/to/tropical-compactor run server.py
```

Verify:

```bash
claude mcp list
# expect: tropical-compactor ... ✓ Connected
```

## Codex Registration

`~/.codex/config.toml`

```toml
[[mcp_servers]]
name = "tropical-compactor"
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-compactor", "run", "server.py"]
```

or CLI:

```bash
codex mcp add tropical-compactor -- \
  uv --directory /absolute/path/to/tropical-compactor run server.py
```

## CyberOps Replay Harness

Run a deterministic replay that pushes CyberOps-style logs through `compact()` and reports policy curves:

```bash
tropical-compactor-replay \
  --fractions 1.0,0.8,0.65,0.5,0.4 \
  --policies recency,l2_guarded \
  --k 3 \
  --output-dir artifacts/cyberops_mcp_replay
```

Outputs:

- `artifacts/cyberops_mcp_replay/replay_rows.csv`
- `artifacts/cyberops_mcp_replay/replay_summary.csv`
- `artifacts/cyberops_mcp_replay/replay_summary.json`

## Operational Notes

- `stdout` must stay clean for JSON-RPC transport. Logging goes to `stderr` only.
- Tagging is heuristic; algebra is exact once tags are correct.
- Preferred workflow: `tag()` -> optional `role_hint` overrides -> `inspect_horizon()` -> `compact_auto()` (or `compact()` for manual control).
- Policy guide:
  - `l2_guarded`: deterministic protected-set preservation (pivot + k predecessors).
  - `l2_iterative_guarded`: attempts additional safe removals while preserving `W[k]`, then falls back to breach-aware guarded eviction if needed.
  - `recency`: baseline newest-first policy.
