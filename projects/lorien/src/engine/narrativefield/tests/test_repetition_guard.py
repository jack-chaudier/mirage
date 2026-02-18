"""Tests for repetition detection and removal."""

from __future__ import annotations

from narrativefield.storyteller.repetition_guard import detect_and_remove_repetition


def test_no_repetition_passes_through() -> None:
    """Clean prose should pass through unchanged."""
    prose = "Diana reached for her glass. Victor watched from across the table."
    cleaned, reps = detect_and_remove_repetition(prose)
    assert cleaned == prose
    assert len(reps) == 0


def test_sentence_repetition_detected() -> None:
    """Same sentence appearing twice should be flagged."""
    prose = (
        "Victor studied the room. The candles flickered. "
        "Marcus shifted uncomfortably. Victor studied the room. "
        "Elena watched in silence."
    )
    cleaned, reps = detect_and_remove_repetition(prose)
    assert "Victor studied the room" in cleaned
    # Should appear only once in cleaned text
    assert cleaned.count("Victor studied the room") == 1


def test_paragraph_repetition_removed() -> None:
    """Same paragraph repeated across scene breaks should be cleaned."""
    prose = (
        "The dining table gleamed under the chandelier. Six faces, six intentions.\n\n"
        "* * *\n\n"
        "The dining table gleamed under the chandelier. Six faces, six intentions.\n\n"
        "* * *\n\n"
        "The dining table gleamed under the chandelier. Six faces, six intentions."
    )
    cleaned, reps = detect_and_remove_repetition(prose)
    assert cleaned.count("Six faces, six intentions") == 1
    assert len(reps) > 0


def test_short_repeats_ignored() -> None:
    """Short common phrases should not be flagged."""
    prose = "He said something. She replied. He said something else. She replied again."
    cleaned, reps = detect_and_remove_repetition(prose, min_repeat_length=30)
    assert cleaned == prose  # no changes, "She replied" is too short


def test_golden_run_bug() -> None:
    """Reproduce the exact bug from the seed-42 live run."""
    prose = (
        "Diana reached out and took her hand. It was something. Not enough. But something."
        "Victor studied the room. The candlelight played across faces that revealed "
        "nothing and everything at once.\n\n* * *\n\n"
        "Victor studied the room. The candlelight played across faces that revealed "
        "nothing and everything at once.\n\n* * *\n\n"
        "Victor studied the room. The candlelight played across faces that revealed "
        "nothing and everything at once.\n\n* * *\n\n"
        "Victor studied the room. The candlelight played across faces that revealed "
        "nothing and everything at once."
    )
    cleaned, reps = detect_and_remove_repetition(prose)
    # The repeated line should appear at most once
    assert cleaned.count("Victor studied the room") <= 1
    # The story content should be preserved
    assert "Diana reached out" in cleaned
    assert len(reps) > 0


def test_orphaned_scene_breaks_cleaned() -> None:
    """Scene break with no content after removal should be cleaned."""
    prose = "Scene one content.\n\n* * *\n\n"
    # After removing what came after * * *, the break should go too
    cleaned, reps = detect_and_remove_repetition(prose)
    assert not cleaned.endswith("* * *")


def test_returns_repetition_metadata() -> None:
    """Repetition dict should contain count and position info."""
    prose = "A long enough sentence to be detected. " * 3
    cleaned, reps = detect_and_remove_repetition(prose, min_repeat_length=20)
    assert len(reps) > 0
    rep = reps[0]
    assert "count" in rep
    assert rep["count"] >= 3


def test_sentence_starter_repetition_flagged() -> None:
    prose = (
        "Diana watched the doorway. "
        "Diana watched the wine tremble in her glass. "
        "Diana watched Marcus swallow before speaking. "
        "Diana watched Elena avoid Thorne's eyes."
    )
    _, reps = detect_and_remove_repetition(prose)
    assert any(rep.get("kind") == "sentence_starter" for rep in reps)


def test_structure_repetition_flagged() -> None:
    prose = (
        "It was the kind that made Diana flinch. "
        "It was the kind that made Victor look away. "
        "It was the kind that made Marcus laugh too loudly. "
        "It was the kind that made Elena flatten her napkin."
    )
    _, reps = detect_and_remove_repetition(prose)
    assert any(rep.get("kind") == "structure" for rep in reps)
