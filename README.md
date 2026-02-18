# The Validity Mirage

LLM outputs under context compression can score high on fluency, coherence, and
format compliance while silently substituting the specific facts that determine
whether the answer is actually correct. We call this failure mode the
**validity mirage**: the answer looks valid but its semantic pivot has shifted.

This repository contains MirageBench (a diagnostic benchmark that detects pivot
substitution), multi-model and KV-cache experiments reproducing the effect, and a
mirage-aware LoRA adapter that learns to flag its own evidence degradation.

## The core result

Across five instruction-tuned models, raw validity scores remain above 0.83
while pivot preservation drops as low as 0.42. The gap is the mirage.

![Validity-preservation gap across 5 frontier models. Raw validity stays high while pivot preservation collapses.](endogenous_context_theory/release/figures/blackbox_validity_vs_pivot_preservation.png)

Models tested: Gemma-2 9B, Llama-3.1 8B, Mistral 7B v0.3, Phi-3-Medium 14B,
Qwen-2.5 14B. All bf16, greedy decoding, MirageBench 12-task set at compression
levels 0.4/0.5/0.6.

### KV-cache eviction

The mirage also appears at the representation level. When KV-cache entries are
evicted (retaining 70% down to 10% of keys), pivot preservation drops to 0% at
10% retention — even though all prerequisite information remains present in the
input text. This isolates the failure to internal attention, not input truncation.

![KV eviction sweep on Llama-3.1 8B. Pivot preservation drops to 0% at 10% retention despite full input context.](endogenous_context_theory/release/figures/kv_retention_protocol_vs_pivot.png)

## What's in this repo

| Directory | Contents |
|---|---|
| `endogenous_context_theory/release/miragebench_tasks/` | 12-task MirageBench set (JSON + index CSV) |
| `endogenous_context_theory/release/notebooks/` | Blackbox 5-model sweep notebook, KV-cache eviction notebook, generator methods notebook |
| `endogenous_context_theory/release/results/` | Raw CSVs for blackbox and KV-cache experiments |
| `endogenous_context_theory/release/adapters/mirage_aware_v1/` | Mirage-aware LoRA weights (Qwen-2.5 14B base) |
| `endogenous_context_theory/release/figures/` | Release figures |
| `endogenous_context_theory/src/` | Tropical semiring algebra, compression, pivot-margin code |
| `endogenous_context_theory/tests/` | 17 synthetic validation experiments |
| `endogenous_context_theory/results/ntsb/` | Real-incident NTSB benchmark (external validation) |
| `papers/` | Canonical paper sources (`papers/sources/*`) and release PDFs (`papers/paper_0*.pdf`) |

## Quick start

```bash
# Setup
cd endogenous_context_theory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all 17 synthetic validation experiments
python scripts/run_all.py

# Rebuild release figures and summary tables
python scripts/build_release_assets.py
```

The blackbox and KV-cache experiments require GPU access. Open the notebooks in
`release/notebooks/` on Colab or a local GPU machine:

- `miragebench_blackbox_bf16_5models_colab.ipynb` — reproduces the 5-model sweep
- `kv_cache_eviction_mirage_colab.ipynb` — reproduces the KV retention curve

To load the mirage-aware adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(base, "endogenous_context_theory/release/adapters/mirage_aware_v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
```

## Reproducibility

See `endogenous_context_theory/release/README.md` for the full artifact map
(paper section to file), integrity checksums, and inference protocol details.
See `docs/reproducibility-checklist.md` for the step-by-step checklist.

Paper publishing workflow:

```bash
./scripts/publish_papers_from_sources.sh
```

## Citation

```bibtex
@article{validity_mirage_2026,
  title   = {The Validity Mirage: How Context Compression Preserves Fluency While Destroying Semantic Pivots},
  author  = {Jack Chaudier Gaffney},
  year    = {2026},
  journal = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

## License

See individual directories for licensing details.
