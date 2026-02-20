# Context Algebra and the Validity Mirage

A self-contained textbook on endogenous semantics, structural failures, and what validity metrics miss.

## Build

Requires a TeX Live installation with `pdflatex`. The following packages must be available: `amsmath`, `amsthm`, `amssymb`, `mathtools`, `graphicx`, `xcolor`, `booktabs`, `longtable`, `float`, `enumitem`, `tikz`, `hyperref`, `cleveref`, `natbib`, `algorithm`, `algpseudocode`, `mathpazo`, `microtype`, `geometry`.

```bash
cd book/
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex   # resolve cross-refs
pdflatex -interaction=nonstopmode -halt-on-error main.tex   # stabilise page numbers
```

Output: `main.pdf` (~120 pages).

## Figures

Figures live in `book/figures/`. Most are copied from `endogenous_context_theory/results/figures/`. Two are generated from CSV data:

```bash
# Requires Python 3 with matplotlib and numpy.
# Use the project's existing venv:
../endogenous_context_theory/.venv/bin/python scripts/generate_figures.py
```

This produces `figures/kj_heatmap.pdf` and `figures/test_48c_pareto_curve.png`.

## Structure

```
book/
  main.tex                          # Root document
  preface.tex                       # Preface with dependency diagram
  latexmkrc                         # latexmk configuration
  refs.bib                          # Merged bibliography
  chapters/
    01_endogenous_pivot_problem.tex  # Ch1: Motivation
    02_formal_problem.tex            # Ch2: Formal problem setup
    03_absorbing_states.tex          # Ch3: Absorbing states theorem
    04_failure_taxonomy_and_hierarchy.tex  # Ch4: Failure taxonomy
    05_context_algebra.tex           # Ch5: Context monoid
    06_tropical_lift.tex             # Ch6: Tropical semiring lift
    07_absorbing_ideal_and_contracts.tex  # Ch7: Absorbing ideal
    08_streaming_traps.tex           # Ch8: Streaming oscillation traps
    09_validity_mirage.tex           # Ch9: The validity mirage
    10_narrative_origin_story.tex    # Ch10: Narrative origin
    11_manifesto.tex                 # Ch11: Design principles
    12_discussion_future_work.tex    # Ch12: Open problems
    appendices.tex                   # Notation, glossary, determinism, artifacts
  figures/                           # Generated and copied figures
  scripts/
    generate_figures.py              # Figure generation from repo CSVs
```
