"""Tests for parsing + formatting helpers."""
from __future__ import annotations
import pytest
from generate_script import _join_natural, _clip_summary
from tts import parse_dialogue, _chunk_turns


def test_join_natural_one():
    assert _join_natural(["A"]) == "A"


def test_join_natural_two():
    assert _join_natural(["A", "B"]) == "A and B"


def test_join_natural_three():
    assert _join_natural(["A", "B", "C"]) == "A, B, and C"


def test_join_natural_empty():
    assert _join_natural([]) == ""


def test_clip_summary_short_unchanged():
    s = "short text"
    assert _clip_summary(s, max_chars=100) == s


def test_clip_summary_long_clipped_on_word_boundary():
    s = "This is a very long summary " * 20
    out = _clip_summary(s, max_chars=50)
    assert len(out) <= 51
    assert out.endswith("…")
    # ends on a word boundary, not mid-word
    assert not out[:-1].endswith(("y", "i", "h"))  # last char before … is space-trimmed


def test_parse_dialogue_basic():
    text = "JAMIE: hello\nALEX: world\n"
    turns = parse_dialogue(text)
    assert turns == [("JAMIE", "hello"), ("ALEX", "world")]


def test_chunk_turns_under_limit_one_chunk():
    turns = [("JAMIE", "short"), ("ALEX", "also short")]
    chunks = _chunk_turns(turns)
    assert len(chunks) == 1


def test_chunk_turns_splits_on_size():
    from tts import V3_MAX_CHARS_PER_REQUEST
    big = "x " * 1000  # 2000 chars — over the per-request limit on its own
    turns = [("JAMIE", big), ("ALEX", big)]
    chunks = _chunk_turns(turns)
    # Oversized turns are now split, so we get MORE than one chunk and
    # every chunk stays under the v3 request limit.
    assert len(chunks) >= 2
    for chunk in chunks:
        assert sum(len(t) for _, t in chunk) <= V3_MAX_CHARS_PER_REQUEST
