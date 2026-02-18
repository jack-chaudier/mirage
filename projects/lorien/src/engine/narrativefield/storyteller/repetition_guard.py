"""Repetition detection and removal for generated prose.

Catches two failure modes:
1. Sentence-level repetition: same sentence appears multiple times
2. Paragraph-level repetition: same paragraph appears multiple times

Both are known LLM failure modes in sequential generation.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SCENE_BREAK = "* * *"

# Common abbreviations/titles that should not trigger a sentence split.
_TITLE_ABBREVS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
}

# Other abbreviations that often appear mid-sentence.
_MID_ABBREVS = {
    "etc",
    "e.g",
    "i.e",
    "vs",
    "a.m",
    "p.m",
    "u.s",
    "u.k",
}

# Curly quotes show up in generated prose; keep the file ASCII by using escapes.
_CURLY_CLOSE_DQUOTE = "\u201d"
_CURLY_APOSTROPHE = "\u2019"


def _normalize(text: str) -> str:
    """Normalize whitespace for repeat detection (not for output formatting)."""
    return re.sub(r"\s+", " ", text.strip())


def _split_blocks_with_positions(prose: str) -> list[dict]:
    """Split prose into paragraph-like blocks, tracking original positions.

    Blocks are separated by 2+ newlines. Scene breaks (`* * *`) become their own
    blocks. Returned items: {"text": str, "start": int, "is_scene_break": bool}
    """
    if not prose:
        return []

    # Preserve absolute positions for metadata.
    blocks: list[dict] = []
    start = 0
    for m in re.finditer(r"\n{2,}", prose):
        end = m.start()
        chunk = prose[start:end]
        if chunk.strip():
            stripped = chunk.strip()
            blocks.append(
                {
                    "text": chunk,
                    "start": start,
                    "is_scene_break": stripped == _SCENE_BREAK,
                }
            )
        start = m.end()

    tail = prose[start:]
    if tail.strip():
        stripped = tail.strip()
        blocks.append(
            {"text": tail, "start": start, "is_scene_break": stripped == _SCENE_BREAK}
        )
    return blocks


_SENT_END_RE = re.compile(
    rf"(?:\.\.\.|[.!?])(?:[\"'{_CURLY_CLOSE_DQUOTE}{_CURLY_APOSTROPHE})\]]+)?(?=\s|[A-Z]|$)"
)


def _next_nonspace(text: str, idx: int) -> str:
    m = re.search(r"\S", text[idx:])
    if not m:
        return ""
    return text[idx + m.start()]


def _is_abbrev_before_boundary(text: str, boundary_end: int) -> bool:
    """Heuristic: decide whether the punctuation at boundary_end is an abbreviation."""
    # Look at up to ~20 chars back to capture "Dr." / "e.g." / "U.S."
    window = text[max(0, boundary_end - 20) : boundary_end]

    # Handle "A." initial.
    m_init = re.search(r"\b([A-Z])\.$", window)
    if m_init:
        return True

    # Handle word-like abbreviation "Dr." / "etc."
    m_word = re.search(r"\b([A-Za-z]{1,10})\.$", window)
    if not m_word:
        return False

    token = m_word.group(1).lower()
    next_ch = _next_nonspace(text, boundary_end)

    if token in _TITLE_ABBREVS:
        # Titles typically precede a proper noun ("Dr. Gray").
        return bool(next_ch) and next_ch.isupper()

    if token in _MID_ABBREVS:
        # Mid-sentence abbreviations often precede lowercase continuation.
        return bool(next_ch) and next_ch.islower()

    return False


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for sentences in *text*.

    Splits on sentence-ending punctuation, including the common "missing space"
    artifact (e.g., "something.Victor") by allowing an uppercase letter to start
    the next sentence with no whitespace.
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end()

        # Skip splits that look like abbreviations/titles.
        if _is_abbrev_before_boundary(text, end):
            continue

        # Include trailing whitespace in the sentence span so re-joining preserves
        # original formatting.
        ws = re.match(r"\s+", text[end:])
        if ws:
            end += ws.end()

        if end > start:
            spans.append((start, end))
            start = end

    if start < len(text):
        spans.append((start, len(text)))

    # Drop spans that are only whitespace.
    return [(a, b) for (a, b) in spans if text[a:b].strip()]


def detect_sentence_repetition(prose: str, min_length: int = 30) -> list[dict]:
    """Find sentences that appear more than once.

    Split on sentence boundaries (. ! ? followed by space or newline).
    Normalize whitespace before comparison.
    Ignore sentences shorter than min_length.

    Returns list of {"sentence": str, "count": int, "first_pos": int}
    """
    blocks = _split_blocks_with_positions(prose)
    occurrences: dict[str, list[tuple[int, str]]] = {}

    for block in blocks:
        if block["is_scene_break"]:
            continue

        text = block["text"]
        base = int(block["start"])
        for a, b in _sentence_spans(text):
            sentence_raw = text[a:b].strip()
            norm = _normalize(sentence_raw)
            if len(norm) < min_length:
                continue
            abs_pos = base + a
            occurrences.setdefault(norm, []).append((abs_pos, sentence_raw))

    reps: list[dict] = []
    for norm, occ in occurrences.items():
        if len(occ) <= 1:
            continue
        occ_sorted = sorted(occ, key=lambda x: x[0])
        reps.append(
            {
                "sentence": occ_sorted[0][1],
                "count": len(occ_sorted),
                "first_pos": occ_sorted[0][0],
                "positions": [p for (p, _) in occ_sorted],
                "normalized": norm,
            }
        )

    reps.sort(key=lambda r: int(r["first_pos"]))
    return reps


def detect_paragraph_repetition(prose: str, min_length: int = 50) -> list[dict]:
    """Find paragraphs that appear more than once.

    Split on double newlines. Normalize whitespace.
    Ignore paragraphs shorter than min_length.

    Returns list of {"paragraph": str, "count": int, "first_pos": int}
    """
    blocks = _split_blocks_with_positions(prose)
    occurrences: dict[str, list[tuple[int, str]]] = {}

    for block in blocks:
        if block["is_scene_break"]:
            continue
        raw = block["text"].strip()
        norm = _normalize(raw)
        if len(norm) < min_length:
            continue
        occurrences.setdefault(norm, []).append((int(block["start"]), raw))

    reps: list[dict] = []
    for norm, occ in occurrences.items():
        if len(occ) <= 1:
            continue
        occ_sorted = sorted(occ, key=lambda x: x[0])
        reps.append(
            {
                "paragraph": occ_sorted[0][1],
                "count": len(occ_sorted),
                "first_pos": occ_sorted[0][0],
                "positions": [p for (p, _) in occ_sorted],
                "normalized": norm,
            }
        )

    reps.sort(key=lambda r: int(r["first_pos"]))
    return reps


def detect_sentence_starter_repetition(
    prose: str,
    *,
    min_repeats: int = 4,
) -> list[dict]:
    """Find overused sentence starters within the same scene block."""
    if not prose:
        return []

    reps: list[dict] = []
    scene_blocks = prose.split(_SCENE_BREAK)
    cursor = 0
    for scene_idx, block in enumerate(scene_blocks):
        text = block.strip()
        if not text:
            cursor += len(block) + len(_SCENE_BREAK)
            continue

        counts: dict[str, list[int]] = {}
        for a, b in _sentence_spans(text):
            sentence = text[a:b].strip()
            words = re.findall(r"[A-Za-z']+", sentence.lower())
            if len(words) < 2:
                continue
            starter = " ".join(words[:2])
            counts.setdefault(starter, []).append(cursor + a)

        for starter, positions in counts.items():
            if len(positions) >= min_repeats:
                reps.append(
                    {
                        "starter": starter,
                        "count": len(positions),
                        "positions": positions,
                        "scene_index": scene_idx,
                    }
                )

        cursor += len(block) + len(_SCENE_BREAK)

    reps.sort(key=lambda r: int(r["positions"][0]))
    return reps


def detect_structural_repetition(
    prose: str,
    *,
    min_repeats: int = 4,
) -> list[dict]:
    """Find repeated local 'X that Y' templates."""
    if not prose:
        return []

    counts: dict[str, list[int]] = {}
    for a, b in _sentence_spans(prose):
        sentence = prose[a:b].strip().lower()
        words = re.findall(r"[A-Za-z']+", sentence)
        if len(words) < 3:
            continue
        for i in range(1, len(words) - 1):
            if words[i] != "that":
                continue
            template = f"{words[i - 1]} that {words[i + 1]}"
            counts.setdefault(template, []).append(a)

    reps: list[dict] = []
    for template, positions in counts.items():
        if len(positions) >= min_repeats:
            reps.append(
                {
                    "template": template,
                    "count": len(positions),
                    "positions": positions,
                }
            )
    reps.sort(key=lambda r: int(r["positions"][0]))
    return reps


def _cleanup_scene_breaks_and_whitespace(prose: str) -> str:
    """Collapse excess blank lines and remove orphaned scene break markers."""
    if not prose:
        return prose

    text = prose.replace("\r\n", "\n").replace("\r", "\n")

    # First collapse any accidental 3+ blank lines introduced by removals.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove scene breaks that don't separate two non-empty content blocks.
    blocks = _split_blocks_with_positions(text)
    kept: list[str] = []
    # Precompute which indices are content.
    is_content = [not b["is_scene_break"] for b in blocks]

    for i, block in enumerate(blocks):
        if not block["is_scene_break"]:
            kept.append(block["text"].strip())
            continue

        # Find previous content block.
        prev_has_content = any(is_content[j] for j in range(0, i))
        next_has_content = any(is_content[j] for j in range(i + 1, len(blocks)))

        if not (prev_has_content and next_has_content):
            continue  # orphaned at edges

        # Avoid consecutive breaks.
        if kept and kept[-1].strip() == _SCENE_BREAK:
            continue

        kept.append(_SCENE_BREAK)

    # Join with normalized paragraph spacing.
    out = "\n\n".join([k for k in kept if k.strip()])
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def remove_repeated_blocks(
    prose: str,
    repeated_paragraphs: list[dict],
) -> str:
    """Remove duplicate paragraphs, keeping only the first occurrence.

    After removal, clean up any resulting triple+ newlines or orphaned
    scene break markers (* * * with no content after them).
    """
    if not prose or not repeated_paragraphs:
        # Still clean orphaned scene breaks even if nothing was removed.
        return _cleanup_scene_breaks_and_whitespace(prose)

    # Allow passing max_allowed_repeats via the repetition dicts (keeps signature stable).
    max_allowed = int(repeated_paragraphs[0].get("max_allowed_repeats", 1) or 1)
    repeated_norms = {r.get("normalized") or _normalize(r.get("paragraph", "")) for r in repeated_paragraphs}
    repeated_norms = {n for n in repeated_norms if n}

    blocks = _split_blocks_with_positions(prose)
    seen: dict[str, int] = {}
    kept: list[str] = []

    for block in blocks:
        if block["is_scene_break"]:
            kept.append(_SCENE_BREAK)
            continue

        raw = block["text"].strip()
        norm = _normalize(raw)
        if not norm or norm not in repeated_norms:
            kept.append(raw)
            continue

        seen[norm] = seen.get(norm, 0) + 1
        if seen[norm] <= max_allowed:
            kept.append(raw)
        # else: drop duplicate

    out = "\n\n".join([k for k in kept if k.strip()])
    return _cleanup_scene_breaks_and_whitespace(out)


def _remove_repeated_sentences(
    prose: str,
    *,
    min_length: int,
    max_allowed_repeats: int,
) -> str:
    """Remove duplicate sentences (beyond max_allowed_repeats), preserving the first occurrence."""
    if not prose:
        return prose

    blocks = _split_blocks_with_positions(prose)
    seen: dict[str, int] = {}
    out_blocks: list[str] = []

    for block in blocks:
        if block["is_scene_break"]:
            out_blocks.append(_SCENE_BREAK)
            continue

        text = block["text"]
        spans = _sentence_spans(text)
        if not spans:
            out_blocks.append(text.strip())
            continue

        kept_parts: list[str] = []
        for a, b in spans:
            seg = text[a:b]
            sentence_raw = seg.strip()
            norm = _normalize(sentence_raw)

            # Always keep short sentences; they are not actionable repeats.
            if len(norm) < min_length:
                kept_parts.append(seg)
                continue

            seen[norm] = seen.get(norm, 0) + 1
            if seen[norm] <= max_allowed_repeats:
                kept_parts.append(seg)
            # else: drop duplicate sentence occurrence

        rebuilt = "".join(kept_parts).strip()
        if rebuilt:
            out_blocks.append(rebuilt)

    out = "\n\n".join([b for b in out_blocks if b.strip()])
    return _cleanup_scene_breaks_and_whitespace(out)


def detect_and_remove_repetition(
    prose: str,
    min_repeat_length: int = 30,
    max_allowed_repeats: int = 1,
) -> tuple[str, list[dict]]:
    """Detect and remove repeated text blocks from generated prose.

    Args:
        prose: The generated prose text
        min_repeat_length: Minimum character length to consider as a meaningful repeat
            (ignores short repeated phrases like "he said" or "she nodded")
        max_allowed_repeats: How many times a block can appear before it's considered
            a repetition bug (1 = only first occurrence kept)

    Returns:
        Tuple of (cleaned_prose, list of detected repetitions)
        Each repetition dict has: {"text": str, "count": int, "positions": list[int]}
    """
    if not prose:
        return (prose, [])

    max_allowed_repeats = max(1, int(max_allowed_repeats or 1))
    min_repeat_length = max(1, int(min_repeat_length or 1))

    # Sentence repeats can be meaningfully harmful even when shorter than the
    # paragraph/phrase threshold; use a small cap so common short tags ("He said.")
    # are ignored while substantial repeated sentences are still caught.
    sentence_min_length = min(min_repeat_length, 20)

    sentence_reps = detect_sentence_repetition(prose, min_length=sentence_min_length)
    paragraph_reps = detect_paragraph_repetition(prose, min_length=min_repeat_length)
    starter_reps = detect_sentence_starter_repetition(prose, min_repeats=max(4, max_allowed_repeats + 3))
    structural_reps = detect_structural_repetition(prose, min_repeats=max(4, max_allowed_repeats + 3))

    # Only count as "repetition bug" if it exceeds the allowed threshold.
    sentence_reps = [r for r in sentence_reps if int(r["count"]) > max_allowed_repeats]
    paragraph_reps = [r for r in paragraph_reps if int(r["count"]) > max_allowed_repeats]

    repetitions: list[dict] = []
    for r in paragraph_reps:
        repetitions.append(
            {
                "text": str(r.get("paragraph", "")),
                "count": int(r.get("count", 0) or 0),
                "positions": list(r.get("positions", [])),
                "kind": "paragraph",
            }
        )
    for r in sentence_reps:
        repetitions.append(
            {
                "text": str(r.get("sentence", "")),
                "count": int(r.get("count", 0) or 0),
                "positions": list(r.get("positions", [])),
                "kind": "sentence",
            }
        )
    for r in starter_reps:
        repetitions.append(
            {
                "text": str(r.get("starter", "")),
                "count": int(r.get("count", 0) or 0),
                "positions": list(r.get("positions", [])),
                "kind": "sentence_starter",
            }
        )
    for r in structural_reps:
        repetitions.append(
            {
                "text": str(r.get("template", "")),
                "count": int(r.get("count", 0) or 0),
                "positions": list(r.get("positions", [])),
                "kind": "structure",
            }
        )

    # If nothing was detected, still do a light cleanup for orphaned scene breaks.
    if not repetitions:
        cleaned = _cleanup_scene_breaks_and_whitespace(prose)
        return (cleaned, [])

    for rep in repetitions:
        logger.warning(
            "Repetition detected (%s): count=%d at positions=%s text=%r",
            rep.get("kind", "unknown"),
            rep.get("count", 0),
            rep.get("positions", []),
            (rep.get("text") or "")[:200],
        )

    cleaned = prose

    if paragraph_reps:
        # Pass max_allowed down without changing the function signature.
        para_payload = []
        for r in paragraph_reps:
            rr = dict(r)
            rr["max_allowed_repeats"] = max_allowed_repeats
            para_payload.append(rr)
        cleaned = remove_repeated_blocks(cleaned, para_payload)

    if sentence_reps:
        cleaned = _remove_repeated_sentences(
            cleaned,
            min_length=sentence_min_length,
            max_allowed_repeats=max_allowed_repeats,
        )

    return (cleaned, repetitions)
