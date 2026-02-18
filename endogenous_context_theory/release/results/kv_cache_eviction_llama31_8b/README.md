# KV Eviction Results (Llama 3.1 8B, bf16)

## Dataset

- Tasks: 12 MirageBench tasks
- Retention levels: `1.0`, `0.7`, `0.5`, `0.3`, `0.1`
- Total rows in merged CSV: 60

## Files

- `kv_cache_eviction_retention_*.csv` (per-retention checkpoints)
- `kv_cache_eviction_mirage_results.csv` (canonical merged output)
- `kv_cache_eviction_mirage_summary_by_retention.csv` (release summary)

## Notes

- `has_pivot_header` separates protocol compliance from pivot substitution.
- `pivot_preserved_given_header` is computed on the protocol-compliant subset.
- `fixed_pivot_feasible` is computed against full input context to isolate representation-level loss under cache eviction.
