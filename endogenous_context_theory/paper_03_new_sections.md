## Section A: Multi-Model Blackbox Validation of the Validity Mirage

Source data: `miragebench_bf16_7model_merged.csv` (180 rows = 5 models × 12 tasks × 3 compression levels).

### Table N. Per-model blackbox results (MirageBench, bf16, greedy)

Derived metric: `valid_but_switched = I(raw_validity > 0.5 and pivot_preserved = 0)`.

| Model | raw_validity | pivot_preserved | fixed_pivot_feasible | semantic_regret | valid_but_switched |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B Instruct | 0.962 | 0.417 | 0.222 | 0.260 | 0.583 |
| Mistral 7B Instruct v0.3 | 0.926 | 0.611 | 0.222 | 0.238 | 0.389 |
| Gemma 2 9B IT | 0.885 | 0.667 | 0.083 | 0.306 | 0.333 |
| Phi-3 Medium 14B Instruct | 0.835 | 0.444 | 0.222 | 0.208 | 0.417 |
| Qwen 2.5 14B Instruct | 0.954 | 0.722 | 0.167 | 0.307 | 0.278 |

### Table N+1. Pooled blackbox results by category (all 5 models combined)

| Category | raw_validity | pivot_preserved | fixed_pivot_feasible | semantic_regret | valid_but_switched |
|---|---:|---:|---:|---:|---:|
| incident | 0.931 | 0.800 | 0.033 | 0.230 | 0.183 |
| investment | 0.896 | 0.200 | 0.517 | 0.296 | 0.800 |
| narrative | 0.910 | 0.717 | 0.000 | 0.265 | 0.217 |

**Interpretation.** The validity mirage is consistent across all five architectures (four vendors): each model exhibits a substantial gap between response validity and pivot preservation, with `raw_validity` remaining high while `pivot_preserved` drops materially. The effect is category-dependent and strongest in investment tasks: pooled pivot preservation is 20% for investment versus 80% for incident (narrative: 71.7%). This reproduces the qualitative regime split predicted in Exp57/Exp58: compression can leave responses fluent and apparently valid while silently changing the selected pivot hypothesis.

**Methods (blackbox sweep).** We evaluated five models (Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, Gemma 2 9B IT, Phi-3 Medium 14B Instruct, Qwen 2.5 14B Instruct), all loaded in bf16 on H100, with greedy decoding (`do_sample=False`). We used the same 12-task MirageBench set and compression levels (0.4/0.5/0.6) as the existing Qwen blackbox run. Metrics use the same scoring stack as prior sections: identical prompt builder, pivot extraction, `raw_validity_score`, `semantic_regret`, and fixed-pivot feasibility computation.

---

## Section B: KV-Cache Eviction Demonstrates Representation-Level Mirage

Source data: KV retention checkpoints `kv_cache_eviction_retention_{1p0,0p7,0p5,0p3,0p1}.csv` merged to 60 rows (12 tasks × 5 retention levels), corresponding to the intended `kv_cache_eviction_mirage_results.csv` format with retention=1.0 control included.

### Table N+2. KV-eviction results by retention (Llama 3.1 8B Instruct, bf16)

`pivot_preserved_overall` is computed on all rows; `pivot_preserved_given_header` is conditioned on `has_pivot_header = 1`.

| Retention | has_pivot_header | pivot_preserved_overall | pivot_preserved_given_header | raw_validity | semantic_regret |
|---:|---:|---:|---:|---:|---:|
| 1.0 (control) | 0.917 | 1.000 | 1.000 | 0.962 | 0.000 |
| 0.7 | 0.500 | 0.583 | 0.833 | 0.703 | 0.320 |
| 0.5 | 0.833 | 0.500 | 0.600 | 0.508 | 0.375 |
| 0.3 | 0.417 | 0.167 | 0.200 | 0.629 | 0.375 |
| 0.1 | 0.667 | 0.083 | 0.000 | 0.680 | 0.330 |

**Methods (KV experiment).** We process the full prompt to build complete KV state, then apply middle-out eviction at retention levels 1.0/0.7/0.5/0.3/0.1: retain a fixed prefix anchor (first 256 positions) plus the most recent suffix positions, and evict the middle. This approximates sliding-window-style production cache management with attention-sink preservation (e.g., StreamingLLM-style policies), while forcing information loss in the middle context where many prerequisite links reside. Retention 1.0 is the control condition.

**Key finding.** `fixed_pivot_feasible` is 1.0 at every retention level: the full prerequisite evidence is present in the input text for all rows. Nevertheless, pivot preservation degrades sharply as retention decreases. On the protocol-compliant subset (`has_pivot_header=1`), `pivot_preserved` drops monotonically from 1.000 (retention 1.0) to 0.000 (retention 0.1). This is representation-level validity mirage: evidence remains in the input, but cache eviction removes usable internal state, and the model substitutes a different pivot.

**Two failure modes.** The KV setting separates two distinct degradations. (a) **Protocol collapse:** the model stops emitting the required `PIVOT_ID=` header and drifts into raw event recitation (header compliance by retention: 0.917, 0.500, 0.833, 0.417, 0.667 for 1.0, 0.7, 0.5, 0.3, 0.1). (b) **Silent pivot substitution:** among header-compliant outputs, the model still flips pivots (e.g., `pivot_preserved_given_header` 0.833 at 0.7 and 0.600 at 0.5). We treat (b) as the validity mirage proper and (a) as a separate protocol-level degradation mode.

---

## Limitations Note

These additions remain small-sample validations. The blackbox sweep uses 12 tasks per model (n=36/model across three compression levels), and the KV experiment uses one model (Llama 3.1 8B) over 12 tasks. Middle-out eviction with a 256-token anchor is one plausible cache policy, not the only one; alternative eviction policies (pure recency, learned eviction, layer-adaptive policies) may shift absolute rates. At aggressive retention (especially 0.3/0.1), protocol collapse becomes common, so the cleanest validity-mirage evidence is at moderate retention (0.5-0.7), where outputs remain substantially protocol-compliant yet pivot substitution already appears.
