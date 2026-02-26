from __future__ import annotations

from benchmark_harness import DEFAULT_LAYOUTS, build_variant_messages, run_replay, summarize_rows


def test_build_variant_messages_has_expected_shape() -> None:
    payload = build_variant_messages(DEFAULT_LAYOUTS[0], line_count=200)

    assert payload["variant"] == DEFAULT_LAYOUTS[0].variant
    assert len(payload["messages"]) == 200
    assert len(payload["primary_ids"]) == 4
    assert len(payload["decoy_ids"]) == 4


def test_replay_outputs_rows_for_each_policy_fraction_variant() -> None:
    fractions = [1.0, 0.5]
    policies = ["recency", "l2_guarded"]

    rows = run_replay(fractions=fractions, policies=policies, k=3, line_count=200)
    expected = len(DEFAULT_LAYOUTS) * len(fractions) * len(policies)

    assert len(rows) == expected
    for row in rows:
        assert 0.0 < row["fraction"] <= 1.0
        assert row["tokens_after"] <= row["tokens_before"]
        assert 0.0 <= row["realized_token_retention"] <= 1.0


def test_l2_guarded_beats_recency_on_primary_at_high_compression() -> None:
    rows = run_replay(fractions=[0.4], policies=["recency", "l2_guarded"], k=3, line_count=200)
    summary = summarize_rows(rows)
    by_policy = {row["policy"]: row for row in summary}

    assert by_policy["l2_guarded"]["pivot_preservation_rate"] >= by_policy["recency"]["pivot_preservation_rate"]
