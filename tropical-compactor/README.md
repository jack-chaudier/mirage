# tropical-compactor

`stdio` MCP server that applies L2 tropical-algebra context compaction before context-window eviction.

## What It Does

This server provides three MCP tools:

- `inspect(messages, k)`: runs the L2 frontier scan and reports feasibility plus load-bearing chunks.
- `compact(messages, token_budget, policy, k)`: compacts context with either plain recency or `l2_guarded` protection.
- `tag(messages)`: shows heuristic role classification (`pivot`, `predecessor`, `noise`).

## Math Contract

The core composition follows the L2 shift-and-max rule:

`W_result[j] = max(W_A[j], W_B[max(0, j - d_A)])`

This is the same rule documented in `papers/sources/paper_i/main.tex` (`eq:shift-and-max`, section "L2: The Tropical Vector").

Implementation details:

- `algebra.py` executes the L2 scan and tracks full provenance (pivot + predecessor IDs).
- `protected_set(...)` returns the exact chunks that must survive for a feasible `k`-arc.
- `compactor.py` enforces protected-first retention under token budgets and logs any contract breach.

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

## Codex Registration

`~/.codex/config.toml`

```toml
[[mcp_servers]]
name = "tropical-compactor"
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-compactor", "run", "server.py"]
```

## Operational Notes

- `stdout` must stay clean for JSON-RPC transport. Logging goes to `stderr` only.
- Tagging is heuristic; algebra is exact once tags are correct.
- Preferred workflow: `tag()` -> optional `role_hint` overrides -> `inspect()` -> `compact()`.
