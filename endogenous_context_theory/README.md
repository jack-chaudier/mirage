# Endogenous Context Theory Validation Suite

A publication-oriented Python test bench for validating or falsifying the **Tropical Endogenous Context Semiring** and companion claims on absorbing states, streaming oscillation traps, divergence, compression mirage, and pivot-margin stability.

## Project Layout

```text
endogenous_context_theory/
├── README.md
├── scripts/
│   ├── run_all.py
│   ├── run_miragebench_ollama.py
│   ├── run_miragebench_api.py
│   ├── build_release_assets.py
│   ├── generate_training_data.py
│   ├── make_balanced_train.py
│   ├── eval_mirage_aware.py
│   └── train_mirage_aware.sh
├── notebooks/
│   └── miragebench_experiments_colab.ipynb
├── data/
│   ├── processed/
│   └── smoke/
├── src/
│   ├── tropical_semiring.py
│   ├── holographic_tree.py
│   ├── generators.py
│   ├── streaming.py
│   ├── compression.py
│   └── pivot_margin.py
├── tests/
│   ├── test_01_exactness.py
│   ├── test_02_associativity.py
│   ├── test_03_monoid_subsumption.py
│   ├── test_04_absorbing_ideal.py
│   ├── test_05_holographic_exactness.py
│   ├── test_06_incremental_consistency.py
│   ├── test_07_scaling.py
│   ├── test_08_divergence.py
│   ├── test_09_record_process.py
│   ├── test_10_tropical_shield.py
│   ├── test_11_compression_mirage.py
│   ├── test_12_deterministic_witness.py
│   ├── test_13_contract_compression.py
│   ├── test_14_margin_correlation.py
│   ├── test_15_adaptive_compression.py
│   ├── test_16_organic_traps.py
│   └── test_17_tropical_streaming.py
├── results/
│   ├── summary_report.md
│   ├── figures/
│   └── raw/
├── release/
│   ├── README.md
│   ├── notebooks/
│   ├── miragebench_tasks/
│   ├── results/
│   ├── adapters/
│   └── figures/
├── run_all.py                # legacy wrapper -> scripts/run_all.py
├── run_miragebench_ollama.py # legacy wrapper -> scripts/run_miragebench_ollama.py
├── run_miragebench_api.py    # legacy wrapper -> scripts/run_miragebench_api.py
├── generate_training_data.py # legacy wrapper -> scripts/generate_training_data.py
├── make_balanced_train.py    # legacy wrapper -> scripts/make_balanced_train.py
├── eval_mirage_aware.py      # legacy wrapper -> scripts/eval_mirage_aware.py
├── train_mirage_aware.sh     # legacy wrapper -> scripts/train_mirage_aware.sh
├── training_data -> data/processed     # compatibility symlink
└── training_data_smoke -> data/smoke   # compatibility symlink
└── requirements.txt
```

## Curated Public-Release Bundle

For the clean paper-facing artifact package (tasks + notebooks + CSVs + adapters), see:

- `release/README.md`

This release bundle is designed so external researchers can map artifacts directly to Paper 03 sections and reproduce headline tables quickly.

## Setup

```bash
cd endogenous_context_theory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Everything

```bash
python scripts/run_all.py
```

This will:
1. Execute all 17 experiments with deterministic seeds.
2. Print a table and PASS/FAIL verdict for each test.
3. Save raw CSV outputs under `results/raw/`.
4. Save publication-style figures under `results/figures/`.
5. Write `results/summary_report.md` with overall scientific assessment.

## Legacy Compatibility

Top-level script names are preserved as wrappers. Existing commands like
`python run_all.py` and `python generate_training_data.py` still work and
delegate to `scripts/` entrypoints.

## Notes

- `NEG_INF` is represented by `float('-inf')` across all algebra code.
- Tropical composition strictly applies the right-shift rule: pivots from the right block gain the full left `d_total`.
- Holographic forest root composition follows reversed index order to preserve temporal semantics.
- If any critical test fails (exactness, associativity, tree consistency), treat it as a potential theory falsification until investigated.
