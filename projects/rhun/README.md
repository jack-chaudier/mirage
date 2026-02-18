# rhun

Research question: **Under what conditions does greedy extraction from causal DAGs with sequential phase constraints provably fail?**

## Motivation
This framework originates from findings in the Lorien/NarrativeField project and generalizes them into a domain-agnostic research instrument.

Key predecessor findings (domain-neutral):
- A single sequential prefix constraint (`k`: minimum pre-turning-point development elements) accounted for effectively all regularization pressure.
- Failure was driven by temporal position of the highest-weight event, not by decomposition of importance across actors.
- Search strategy variants did not remove the core failure mode when high-weight events were injected into every candidate pool.

## Architecture
Pipeline:
1. Generators (`rhun/generators`) produce causal DAGs with controllable temporal and weight structure.
2. Extraction (`rhun/extraction`) proposes subsequences and assigns sequential phases.
3. Grammar (`GrammarConfig` + validator) enforces phase and structural constraints.
4. Theory (`rhun/theory`) predicts impossibility conditions and diagnoses absorbing-state failure.
5. Experiments (`rhun/experiments`, root `experiments/`) run deterministic sweeps and persist results.

## Installation

```bash
cd ~/rhun
python -m pip install -e ".[dev]"
```

## Test

```bash
cd ~/rhun
pytest
```

## Run First Experiment

```bash
cd ~/rhun
python experiments/run_position_sweep.py
```

## Organization Quick Map

- Paper entrypoints: `paper/README.md`
- Experiment entrypoints: `experiments/README.md`
- Topic views (non-destructive symlinks): `experiments/by_topic/`
- Streaming outputs + canonical/alias summary names: `experiments/output/streaming/README.md`

## Cross-Project Workspace

An umbrella workspace is available at `~/research` for cross-project indexing and paper discovery:

```bash
~/research/scripts/research_status.sh
~/research/scripts/refresh_workspace.sh
~/research/scripts/run_tests_all.sh
```

## Attribution
The research question was born in Lorien/NarrativeField (`~/lorien`) and is studied here as a separate, domain-agnostic framework.
