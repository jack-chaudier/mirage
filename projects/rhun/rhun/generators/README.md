# Graph Generators

This package contains deterministic causal DAG generators for constrained extraction research.

## Uniform Generator (`uniform.py`)
Null model where event importance has no systematic temporal bias.

Parameters (`UniformConfig`):
- `n_events` (int, default `200`): number of events.
- `n_actors` (int, default `6`): number of actors.
- `seed` (int, default `42`): random seed.
- `causal_density` (float, default `0.05`): probability of retaining each candidate parent link.
- `causal_window` (float, default `0.2`): max backward temporal distance for parent candidates.
- `weight_distribution` (`"uniform" | "normal" | "exponential"`, default `"uniform"`).

Expected properties:
- Event timestamps are roughly uniform in `[0, 1]`.
- Max-weight event position is broadly distributed over time.
- Theorem precondition (`max index < k`) is uncommon for small `k`.

## Bursty Generator (`bursty.py`)
Bursty temporal process with front-loaded high-weight events.

Parameters (`BurstyConfig`):
- `epsilon` (float in `[0, 1]`, default `0.3`): front-loading control.
- `n_bursts` (int, default `3`): number of activity bursts.
- `burst_width` (float, default `0.1`): burst spread in normalized time.
- `hub_fraction` (float, default `0.05`): top-weight fraction involving all actors.
- `causal_density` (float, default `0.08`): base chance an event receives parent links.
- `weight_heavy_tail` (float, default `2.0`): Pareto shape for front-loaded high weights.

Relationship between `epsilon` and theorem precondition:
- As `epsilon` increases, high weights are concentrated earlier in time.
- Early max-weight events increase probability that `j < k` and therefore greedy failure.
- With `epsilon > 0.5`, max-weight position should often be early enough to trigger failures for `k >= 1`.

Verification checks:
- Compute max-weight normalized position per graph and average over seeds.
- For larger `epsilon`, mean max-weight position should decrease.
- For `epsilon > 0.5`, a practical target is `P(max in first epsilon-fraction) > 0.8` across many seeds.

## Example

```python
from rhun.generators.uniform import UniformConfig, UniformGenerator
from rhun.generators.bursty import BurstyConfig, BurstyGenerator

uniform_graph = UniformGenerator().generate(UniformConfig(seed=7))
bursty_graph = BurstyGenerator().generate(BurstyConfig(seed=7, epsilon=0.7))

print(uniform_graph.n_events, bursty_graph.n_events)
```
