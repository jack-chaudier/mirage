# Build Notes

## Style / Template
- `neurips_2026.sty` was not available at `media.neurips.cc` (404).
- Used official NeurIPS style fallback: `paper/neurips_2023.sty` from `https://media.neurips.cc/Conferences/NeurIPS2023/Styles/neurips_2023.sty`.

## Figure Provenance
All manuscript figures were generated from real experiment outputs (no placeholders):
- `paper/figures/kj_heatmap.pdf` from `experiments/output/kj_boundary_final.json`
- `paper/figures/position_sweep.pdf` from `experiments/output/position_sweep.json`
- `paper/figures/hierarchy_comparison.pdf` from `experiments/output/tp_solver_evaluation_m25.json`
- `paper/figures/antagonism.pdf` conceptual companion figure (drawn programmatically, no synthetic empirical values)

Generation script:
- `paper/generate_figures.py`

## Citations
- `paper/references.bib` populated with verified entries for:
  - Korte & Lovasz (1981)
  - Bjorner & Ziegler (1992)
  - van der Aalst (2016)
  - Scholak et al. (2021)
  - Willard & Louf (2023)
  - Irnich & Desaulniers (2005)
  - Gange & Stuckey (2020)
- No `[CITE]` placeholders remain.

## Compilation
Commands run:
```bash
cd ~/rhun/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output:
- `paper/main.pdf` generated successfully.
- Page count: `7` pages.
- No unresolved references/citations (`??` not present; no undefined-reference warnings in final log).

## Environment Notes
- `neurips_2023.sty` dependencies were missing in the local TeX user tree.
- Installed user-mode TeX packages:
  - `environ`
  - `trimspaces`
  - `units` (provides `nicefrac.sty`)

## Remaining Formatting Warnings
- Minor overfull box warnings remain in dense taxonomy table layout (`main.log`), but PDF compiles and renders all figures/tables correctly.
