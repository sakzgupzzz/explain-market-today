"""Tests for sanitize.py post-processing guardrails."""
from __future__ import annotations
import pytest
from sanitize import (
    sanitize_script, _strip_banned_openers, _fix_wrong_name_intros,
    _space_tickers, _enforce_jamie_cap, _parse, _collapse_same_speaker_streaks,
)


# ─────────── beat-seam handling: no inline tags, no topic cram ───────────

def test_collapse_leaves_full_seam_turns_separate():
    # Two full same-speaker turns (a beat seam) must NOT fuse — that crammed
    # topics and pulled the second turn's [tag] inline.
    turns = [
        ("MAYA", "[mischievously] SentinelOne lost the market's faith entirely."),
        ("MAYA", "[rushed] Wikipedia editors are threatening to strike over AI."),
    ]
    out, merges = _collapse_same_speaker_streaks(turns)
    assert merges == 0
    assert len(out) == 2
    # No inline tag leaked into any merged body.
    assert "[rushed]" not in out[0][1]


def test_collapse_folds_short_fragment_and_strips_tag():
    turns = [
        ("ALEX", "The S&P closed up a third of a percent on thin volume."),
        ("ALEX", "[sighs] Exactly."),  # short continuation fragment
    ]
    out, merges = _collapse_same_speaker_streaks(turns)
    assert merges == 1
    assert len(out) == 1
    assert "[sighs]" not in out[0][1]   # tag stripped, not inlined
    assert out[0][1].endswith("Exactly.")


def test_disclaimer_signature_ignores_casual_investment_advice():
    from sanitize import _disclaimer_signature
    # Real markets talk mentioning "investment advice" must NOT be flagged
    # (else _dedup_disclaimer deletes it as a stray disclaimer).
    assert not _disclaimer_signature("Honestly the best investment advice is buy low, sell high.")
    assert _disclaimer_signature("This show is for entertainment and education only — nothing here is investment advice.")


def test_dedup_disclaimer_mid_episode_casual_mention_preserved():
    from sanitize import _dedup_disclaimer
    # A casual "not investment advice" at turn 12 of 30 must NOT anchor the
    # drop-everything-after behavior — that silently deleted turns 13-30.
    hosts = ["JAMIE", "ALEX", "MAYA"]
    turns = [
        (hosts[i % 3], f"Turn {i} covers a distinct market story in real detail.")
        for i in range(30)
    ]
    turns[12] = ("ALEX", "Look, that's not investment advice, anyway — the chart is just ugly.")
    out, drops = _dedup_disclaimer(turns)
    assert drops == 0
    assert len(out) == 30
    assert out[29][1] == "Turn 29 covers a distinct market story in real detail."
    # And through the full pipeline: back half of the episode survives.
    script = "\n".join(f"{n}: {t}" for n, t in turns)
    sanitized = sanitize_script(script, verbose=False)
    assert "Turn 29 covers" in sanitized


def test_dedup_disclaimer_canonical_still_dedups_and_ends_script():
    from sanitize import _dedup_disclaimer
    from config import DISCLAIMER_SHORT
    turns = [
        ("JAMIE", "Cold open with substance."),
        ("MAYA", "This show is for entertainment and education only — nothing here is investment advice."),
        ("ALEX", "Markets stuff."),
        ("JAMIE", "Not investment advice, folks."),  # near-end → allowed to anchor
        ("MAYA", "Trailing chatter that should drop."),
    ]
    out, drops = _dedup_disclaimer(turns)
    assert out[-1][1] == DISCLAIMER_SHORT
    assert drops == 2  # early canonical dropped + trailing turn dropped
    assert [n for n, _ in out] == ["JAMIE", "ALEX", "JAMIE"]


def test_unknown_acronyms_pass_through_unspaced():
    from sanitize import _space_standalone_tickers
    # Unknown all-caps words must NOT be letter-spaced — old default read
    # "NASA" as "N A S A" on air.
    out, fixes = _space_standalone_tickers("NASA and OSHA both weighed in on the SPAC.")
    assert out == "NASA and OSHA both weighed in on the SPAC."
    assert fixes == 0
    # Known map entries still resolve to company names.
    out, fixes = _space_standalone_tickers("NVDA ripped again.")
    assert out == "Nvidia ripped again."
    assert fixes == 1


def test_ambiguous_tickers_not_expanded():
    from sanitize import _space_standalone_tickers
    # NOW / MA / HD / KO collide with common words — must pass through untouched.
    for s in ["NOW we turn to the Fed.", "She has an MA in economics.",
              "Watching the game in HD tonight.", "He was out cold, KO in round two."]:
        out, _ = _space_standalone_tickers(s)
        assert "ServiceNow" not in out and "Mastercard" not in out
        assert "Home Depot" not in out and "Coca-Cola" not in out


def test_crocs_ipo_template_scrubbed():
    script = (
        "JAMIE: Victoria's Secret jumped forty percent today.\n"
        "ALEX: That's the Crocs IPO of corporate governance, frankly.\n"
        "JAMIE: This show is for entertainment and education only — nothing here is investment advice."
    )
    out = sanitize_script(script, verbose=False)
    assert "crocs ipo of" not in out.lower()


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
    """Known tickers in parens get DROPPED (the LLM almost always writes the
    company name immediately before, making the paren redundant)."""
    fixed, count = _space_tickers("Apple (AAPL) and Microsoft (MSFT) ripped today.")
    assert "(AAPL)" not in fixed
    assert "(MSFT)" not in fixed
    # Company names already present in the input — final string mentions them
    assert "Apple" in fixed
    assert "Microsoft" in fixed
    assert count == 2


def test_space_tickers_unknown_ticker_falls_back_to_letters():
    """Tickers not in _TICKER_TO_NAME still get letter-spaced for TTS."""
    fixed, count = _space_tickers("New IPO trading as (XYZA) today.")
    assert "X Y Z A" in fixed
    assert "(XYZA)" not in fixed
    assert count == 1


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
    # Known tickers now resolve to company names instead of letter-spacing
    assert "(AAPL)" not in out
    assert "(NVDA)" not in out
    assert "Nvidia" in out
    # ALEX line shouldn't say "Jamie here"
    alex_line = next(line for line in out.splitlines() if line.startswith("ALEX:"))
    assert "Jamie here" not in alex_line
    assert "Alex here" in alex_line or "Alex" in alex_line


def test_sanitize_empty_script():
    assert sanitize_script("", verbose=False) == ""
