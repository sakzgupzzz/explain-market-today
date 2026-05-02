"""Tests for sanitize.py post-processing guardrails."""
from __future__ import annotations
import pytest
from sanitize import (
    sanitize_script, _strip_banned_openers, _fix_wrong_name_intros,
    _space_tickers, _enforce_jamie_cap, _parse,
)


def test_strip_banned_openers_basic():
    text, removed = _strip_banned_openers("Well folks, the market opened down.")
    assert "Well folks" not in text
    assert "the market opened down" in text
    assert len(removed) >= 1


def test_strip_banned_openers_chained():
    text, removed = _strip_banned_openers(
        "Welcome to the show! Good morning everyone. Let's dive in. Apple was up."
    )
    assert "Welcome to the show" not in text
    assert "Good morning" not in text
    assert "Let's dive in" not in text
    assert "Apple was up" in text


def test_strip_banned_openers_idempotent():
    clean = "Apple beat earnings."
    text, removed = _strip_banned_openers(clean)
    assert text == clean
    assert removed == []


def test_fix_wrong_name_intros_alex_says_jamie():
    fixed, count = _fix_wrong_name_intros("ALEX", "Jamie here, the market is wild today.")
    assert "Alex" in fixed
    assert "Jamie here" not in fixed
    assert count == 1


def test_fix_wrong_name_intros_self_intro_passes():
    fixed, count = _fix_wrong_name_intros("ALEX", "Alex here, the market is wild.")
    assert fixed.startswith("Alex here")
    assert count == 0


def test_fix_wrong_name_intros_mid_sentence_unchanged():
    fixed, count = _fix_wrong_name_intros(
        "ALEX", "Anyway, Jamie made a good point earlier."
    )
    # "Jamie made a good point" — no self-intro pattern, not modified
    assert "Jamie" in fixed
    assert count == 0


def test_space_tickers_basic():
    fixed, count = _space_tickers("Apple (AAPL) and Microsoft (MSFT) ripped today.")
    assert "A A P L" in fixed
    assert "M S F T" in fixed
    assert "(AAPL)" not in fixed
    assert count == 2


def test_space_tickers_skips_common_acronyms():
    fixed, count = _space_tickers("The (CEO) said the (IPO) flopped.")
    assert "(CEO)" in fixed
    assert "(IPO)" in fixed
    assert count == 0


def test_enforce_jamie_cap_drops_short_jamie_turns():
    turns = [
        ("JAMIE", "Cold open with substance about the market today."),
        ("ALEX", "Markets analyst report on tech."),
        ("JAMIE", "Right."),
        ("MAYA", "Tech beat report."),
        ("JAMIE", "Wow."),
        ("CAM", "Macro report."),
        ("JAMIE", "Okay."),
        ("KAI", "Odd thing of the day."),
        ("JAMIE", "Sign-off line."),
    ]
    out, dropped = _enforce_jamie_cap(turns, cap_ratio=1 / 3)
    jamie_count = sum(1 for n, _ in out if n == "JAMIE")
    total = len(out)
    assert jamie_count <= total // 3 + 1
    assert dropped >= 1
    # cold open (turn 0) preserved
    assert out[0][0] == "JAMIE"


def test_enforce_jamie_cap_under_limit_unchanged():
    turns = [
        ("JAMIE", "Cold open."),
        ("ALEX", "Markets."),
        ("MAYA", "Tech."),
        ("CAM", "Macro."),
        ("KAI", "Odd."),
    ]
    out, dropped = _enforce_jamie_cap(turns, cap_ratio=1 / 3)
    assert dropped == 0
    assert len(out) == len(turns)


def test_parse_basic():
    text = "JAMIE: Hello.\nALEX: World.\n"
    turns = _parse(text)
    assert turns == [("JAMIE", "Hello."), ("ALEX", "World.")]


def test_parse_unknown_speaker_falls_back_to_default():
    text = "BOGUS: Should map to default.\n"
    turns = _parse(text)
    assert len(turns) == 1
    # falls back to JAMIE (DEFAULT_CHARACTER)
    assert turns[0][0] == "JAMIE"


def test_parse_continuation_line_glues_to_previous_speaker():
    text = "JAMIE: First sentence.\nstill talking.\nALEX: Now me.\n"
    turns = _parse(text)
    assert len(turns) == 2
    assert "still talking" in turns[0][1]


def test_sanitize_script_full_pipeline():
    raw = (
        "JAMIE: Well folks, welcome to the show! Apple (AAPL) ripped.\n"
        "ALEX: Jamie here, big news from (NVDA) today.\n"
        "JAMIE: Right.\n"
        "MAYA: Cool.\n"
        "JAMIE: Wow.\n"
        "CAM: Macro stuff.\n"
        "JAMIE: Okay.\n"
        "KAI: Odd thing.\n"
    )
    out = sanitize_script(raw, verbose=False)
    assert "Well folks" not in out
    assert "welcome to the show" not in out.lower()
    assert "A A P L" in out
    assert "N V D A" in out
    # ALEX line shouldn't say "Jamie here"
    alex_line = next(line for line in out.splitlines() if line.startswith("ALEX:"))
    assert "Jamie here" not in alex_line
    assert "Alex here" in alex_line or "Alex" in alex_line


def test_sanitize_empty_script():
    assert sanitize_script("", verbose=False) == ""
