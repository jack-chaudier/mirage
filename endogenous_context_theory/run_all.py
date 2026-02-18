from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

TEST_MODULES = [
    "tests.test_01_exactness",
    "tests.test_02_associativity",
    "tests.test_03_monoid_subsumption",
    "tests.test_04_absorbing_ideal",
    "tests.test_05_holographic_exactness",
    "tests.test_06_incremental_consistency",
    "tests.test_07_scaling",
    "tests.test_08_divergence",
    "tests.test_09_record_process",
    "tests.test_10_tropical_shield",
    "tests.test_11_compression_mirage",
    "tests.test_12_deterministic_witness",
    "tests.test_13_contract_compression",
    "tests.test_14_margin_correlation",
    "tests.test_15_adaptive_compression",
    "tests.test_16_organic_traps",
    "tests.test_17_tropical_streaming",
]


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_summary_report(run_df: pd.DataFrame, results_dir: Path) -> None:
    raw = results_dir / "raw"

    t1 = _safe_read_csv(raw / "test_01_exactness.csv")
    t2 = _safe_read_csv(raw / "test_02_associativity.csv")
    t3 = _safe_read_csv(raw / "test_03_monoid_subsumption.csv")
    t4 = _safe_read_csv(raw / "test_04_absorbing_ideal.csv")
    t8fit = _safe_read_csv(raw / "test_08_divergence_fit.csv")
    t11 = _safe_read_csv(raw / "test_11_mirage_summary.csv")
    t13 = _safe_read_csv(raw / "test_13_contract_compression_summary.csv")
    t14 = _safe_read_csv(raw / "test_14_margin_quartiles.csv")
    t17 = _safe_read_csv(raw / "test_17_tropical_streaming_overall.csv")

    lines: List[str] = []
    lines.append("# Tropical Endogenous Context Semiring: Experimental Summary")
    lines.append("")
    lines.append("## 1. Validation Status")

    for _, row in run_df.iterrows():
        lines.append(
            f"- **{row['test']}**: **{row['verdict']}** "
            f"(runtime={row['runtime_s']:.2f}s, key metric={row['key_metric']})"
        )

    if t1 is not None:
        v = int(t1["violations"].sum())
        lines.append(f"- Exactness violations (Test 1): **{v}**")
    if t2 is not None:
        v = int(t2["violations"].sum())
        lines.append(f"- Associativity violations (Test 2): **{v}**")

    lines.append("")
    lines.append("## 2. Validated vs Falsified Claims")

    critical = run_df[run_df["test"].str.contains("Test 01|Test 02|Test 05|Test 06|Test 17")]
    if (critical["verdict"] == "PASS").all():
        lines.append("- Core algebraic soundness claims are validated in this run (exactness, associativity, tree consistency, streaming equivalence).")
    else:
        lines.append("- At least one core algebraic soundness claim is falsified in this run; inspect failing critical tests before relying on the theory.")

    failed = run_df[run_df["verdict"] == "FAIL"]
    if len(failed) == 0:
        lines.append("- No tests were falsified in this run.")
    else:
        lines.append("- Falsified tests in this run:")
        for _, row in failed.iterrows():
            lines.append(f"  - {row['test']}")

    lines.append("")
    lines.append("## 3. Surprising Findings")

    if t3 is not None:
        m = float(t3["multi_slot_rate"].mean())
        lines.append(f"- Mean fraction of instances with multiple occupied tropical slots (Test 3): **{m:.3f}**.")
    if t4 is not None:
        esc = float(t4["uncommitted_escape_rate"].mean())
        lines.append(f"- Mean uncommitted escape rate from absorbed prefixes (Test 4): **{esc:.3f}**.")

    lines.append("")
    lines.append("## 4. Divergence Characterization")

    if t8fit is not None and len(t8fit) > 0:
        pooled = t8fit[t8fit["epsilon"] == "pooled"]
        if len(pooled) == 0:
            pooled = t8fit.tail(1)
        p = pooled.iloc[0]
        b = float(p["b"])
        lo = float(p["b_ci_low"])
        hi = float(p["b_ci_high"])
        lines.append(
            f"- Estimated power-law exponent: **b = {b:.3f}** (95% CI [{lo:.3f}, {hi:.3f}])."
        )
        if b > 1.0:
            lines.append("- Interpretation: super-linear reassignment growth supports thermodynamic divergence concerns for irrevocable commitment.")
        else:
            lines.append("- Interpretation: near-linear growth suggests weaker divergence than hypothesized under current generator assumptions.")

    lines.append("")
    lines.append("## 5. Mirage Severity and Contract Fix")

    if t11 is not None:
        high = t11[t11["retention"] <= 0.5]
        if len(high) > 0:
            gap = float((high["raw_validity"] - high["pivot_preservation"]).mean())
            reg = float(high["semantic_regret"].mean())
            lines.append(
                f"- Mean mirage gap at retention <= 0.5: **{gap:.3f}** (raw validity - preservation)."
            )
            lines.append(f"- Mean semantic regret at retention <= 0.5: **{reg:.3f}**.")

    if t13 is not None:
        naive = t13[t13["method"] == "naive"]
        guard = t13[t13["method"] == "contract"]
        if len(naive) > 0 and len(guard) > 0:
            n_pres = float(naive["pivot_preservation"].mean())
            g_pres = float(guard["pivot_preservation"].mean())
            n_ret = float(naive["achieved_retention"].mean())
            g_ret = float(guard["achieved_retention"].mean())
            lines.append(
                f"- Contract guard pivot preservation: **{g_pres:.3f}** vs naive **{n_pres:.3f}**."
            )
            lines.append(
                f"- Contract guard achieved retention: **{g_ret:.3f}** vs naive **{n_ret:.3f}** (higher means less aggressive compression)."
            )

    lines.append("")
    lines.append("## 6. Margin Hypothesis")

    t14raw = _safe_read_csv(raw / "test_14_margin_correlation_raw.csv")
    if t14raw is not None and len(t14raw) > 0:
        corr = float(t14raw[["margin_finite", "preserved"]].corr().iloc[0, 1])
        if pd.isna(corr):
            lines.append("- Margin-preservation correlation (Pearson proxy): **undefined** (constant-input regime).")
        else:
            lines.append(f"- Margin-preservation correlation (Pearson proxy): **{corr:.3f}**.")

    if t14 is not None:
        q = t14.sort_values("quartile")
        quartile_str = ", ".join(
            [f"{r['quartile']}={r['pivot_preservation']:.3f}" for _, r in q.iterrows()]
        )
        lines.append(f"- Quartile preservation rates: {quartile_str}.")

    lines.append("")
    lines.append("## 7. Overall Assessment")

    critical_ok = (critical["verdict"] == "PASS").all() if len(critical) > 0 else False
    streaming_ok = False
    if t17 is not None and len(t17) > 0:
        trop = float(t17[t17["policy"] == "tropical"]["validity"].iloc[0])
        fin = float(t17[t17["policy"] == "finite"]["validity"].iloc[0])
        streaming_ok = abs(trop - fin) <= 1e-12

    if critical_ok and streaming_ok:
        lines.append(
            "The current empirical suite supports the Tropical Endogenous Context Semiring as a sound computational foundation for endogenous pivot tracking, with exact composition and tree compatibility under tested regimes."
        )
    else:
        lines.append(
            "The current run reveals unresolved issues in core or streaming equivalence tests; the theory should be treated as provisional until those failures are resolved."
        )

    report_path = results_dir / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []

    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        start = time.perf_counter()
        try:
            out = module.run(results_dir=RESULTS)
            verdict = str(out.get("verdict", "UNKNOWN"))
            key_metric = "; ".join(
                [
                    f"{k}={v}"
                    for k, v in out.items()
                    if k not in {"name", "verdict"}
                ]
            )
            test_name = str(out.get("name", mod_name))
        except Exception as exc:
            verdict = "FAIL"
            key_metric = f"exception={type(exc).__name__}: {exc}"
            test_name = mod_name
            print(f"\n[ERROR] {mod_name} failed with exception: {exc}")

        elapsed = time.perf_counter() - start
        run_rows.append(
            {
                "test": test_name,
                "module": mod_name,
                "verdict": verdict,
                "runtime_s": elapsed,
                "key_metric": key_metric,
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(RESULTS / "raw" / "run_all_summary.csv", index=False)

    print("\n" + "#" * 96)
    print("RUN SUMMARY")
    print("#" * 96)
    print(run_df.to_string(index=False))

    write_summary_report(run_df, RESULTS)
    print(f"\nWrote summary report: {RESULTS / 'summary_report.md'}")


if __name__ == "__main__":
    main()
