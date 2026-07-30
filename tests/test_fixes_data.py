"""Tests for data-layer fixes: feed fetch timeout + failure cooldown, UTC
timestamp parsing, word-boundary keyword matching, mover sign filtering,
seed-token clustering, NYSE-aware staleness, calendar warnings, and the
verify-pass sanity gates."""
from __future__ import annotations
import time
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import requests

import calendar_events
import fetch_market
import fetch_news
import verify_facts
from cluster import cluster_headlines
from score import _keyword_score, score_clusters


# ─── fetch_news: HTTP timeout + failure accounting ──────────────────────────

_RSS = (
    b'<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>'
    b"<item><title>Apple earnings beat</title><link>http://x</link></item>"
    b"</channel></rss>"
)


def _cutoff():
    return datetime.now(timezone.utc) - timedelta(hours=24)


def test_fetch_one_feed_fetches_with_timeout():
    resp = MagicMock(content=_RSS)
    resp.raise_for_status.return_value = None
    with patch.object(fetch_news.requests, "get", return_value=resp) as get:
        out = fetch_news._fetch_one_feed("http://feed", _cutoff(), "markets", health={})
    assert get.call_args.kwargs["timeout"] == fetch_news.FEED_TIMEOUT_SEC
    assert out and out[0]["title"] == "Apple earnings beat"


def test_http_error_counts_as_feed_failure():
    health: dict = {}
    with patch.object(fetch_news.requests, "get", side_effect=requests.ConnectionError("boom")):
        out = fetch_news._fetch_one_feed("http://feed", _cutoff(), "markets", health=health)
    assert out == []
    assert health["http://feed"]["fail_streak"] == 1


# ─── fetch_news: disable cooldown ───────────────────────────────────────────

def test_disabled_feed_retried_after_cooldown():
    url = "http://feed"
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    health = {url: {"fail_streak": fetch_news.FAIL_THRESHOLD, "retry_after": future}}
    assert fetch_news._is_feed_disabled(health, url) is True
    health[url]["retry_after"] = past
    assert fetch_news._is_feed_disabled(health, url) is False
    # legacy record without a cooldown timestamp gets a retry too
    assert fetch_news._is_feed_disabled({url: {"fail_streak": 9}}, url) is False


def test_failure_at_threshold_sets_cooldown():
    health: dict = {}
    for _ in range(fetch_news.FAIL_THRESHOLD):
        fetch_news._record_outcome(health, "u", ok=False, err="e")
    assert "retry_after" in health["u"]


def test_success_resets_streak_and_clears_cooldown():
    health = {"u": {"fail_streak": 7, "retry_after": "2099-01-01T00:00:00+00:00",
                    "last_error": "x", "last_seen": ""}}
    fetch_news._record_outcome(health, "u", ok=True)
    assert health["u"]["fail_streak"] == 0
    assert "retry_after" not in health["u"]


# ─── fetch_news: UTC timestamps ─────────────────────────────────────────────

def test_entry_dt_treats_struct_time_as_utc():
    ts = 1750000000
    entry = {"published_parsed": time.gmtime(ts)}
    dt = fetch_news._entry_dt(entry)
    assert dt == datetime.fromtimestamp(ts, tz=timezone.utc)


# ─── score: word-boundary keyword matching ──────────────────────────────────

def test_keyword_score_ignores_substring_hits():
    # "war" in software, "ban" in banking, "miss" in commission, "cuts" in haircuts
    assert _keyword_score("New software update for banking commission haircuts") == 0.0


def test_keyword_score_still_matches_words_and_phrases():
    assert _keyword_score("War fears deepen") >= 4.0
    assert _keyword_score("Fed weighs interest rate move") >= 4.0  # multi-word phrase


def test_watchlist_keyword_requires_word_boundary():
    clusters = [{"id": "a", "title": "Airlines merger talks", "summary": "",
                 "sources": ["AP"], "categories": ["business"], "published": "",
                 "cluster_size": 1, "headlines": []}]
    base = score_clusters(clusters, market={}, interests={})[0]["score"]
    boosted = score_clusters(
        clusters, market={}, interests={"watchlist": {"keywords": ["ai"]}}
    )[0]["score"]
    assert boosted == base  # "ai" must not match inside "Airlines"


# ─── fetch_market: mover sign filter ────────────────────────────────────────

def _rows(pcts):
    return [{"symbol": f"S{i}", "name": f"S{i}", "close": 1.0, "prev_close": 1.0, "pct": p}
            for i, p in enumerate(pcts)]


def test_fetch_movers_red_day_has_no_fake_gainers():
    with patch.object(fetch_market, "_snapshot", return_value=(_rows([-0.5, -1.2, -3.0]), "2026-07-29")):
        gainers, losers, _ = fetch_market.fetch_movers(n=8)
    assert gainers == []
    assert [r["pct"] for r in losers] == [-3.0, -1.2, -0.5]


def test_fetch_movers_lists_do_not_overlap():
    with patch.object(fetch_market, "_snapshot", return_value=(_rows([2.0, -1.0, 0.5, 0.0]), "2026-07-29")):
        gainers, losers, _ = fetch_market.fetch_movers(n=8)
    assert all(r["pct"] > 0 for r in gainers)
    assert all(r["pct"] < 0 for r in losers)
    assert not {r["symbol"] for r in gainers} & {r["symbol"] for r in losers}


# ─── fetch_market: NYSE-aware staleness ─────────────────────────────────────

_NY = ZoneInfo("America/New_York")


def test_last_expected_session_weekend_rolls_to_friday():
    sat = datetime(2026, 8, 1, 12, 0, tzinfo=_NY)
    sun = datetime(2026, 8, 2, 12, 0, tzinfo=_NY)
    assert fetch_market._last_expected_session(sat) == date(2026, 7, 31)
    assert fetch_market._last_expected_session(sun) == date(2026, 7, 31)


def test_last_expected_session_premarket_monday_expects_friday():
    mon_early = datetime(2026, 8, 3, 8, 0, tzinfo=_NY)
    assert fetch_market._last_expected_session(mon_early) == date(2026, 7, 31)


def test_last_expected_session_midday_weekday_is_today():
    wed = datetime(2026, 7, 29, 12, 0, tzinfo=_NY)
    assert fetch_market._last_expected_session(wed) == date(2026, 7, 29)


# ─── cluster: no single-link chaining via grown token set ───────────────────

def test_cluster_no_chaining_via_grown_token_set():
    headlines = [
        {"title": "Apple iPhone sales fall in China on weakness", "source": "Reuters",
         "category": "tech", "summary": "", "published": ""},
        {"title": "Apple iPhone sales fall sharply in China analysts stunned", "source": "AP",
         "category": "tech", "summary": "", "published": ""},
        # Overlaps only the tokens the SECOND member added — must not chain in.
        {"title": "Analysts stunned by sharply higher oil prices", "source": "Bloomberg",
         "category": "markets", "summary": "", "published": ""},
    ]
    clusters = cluster_headlines(headlines)
    assert sorted(c["cluster_size"] for c in clusters) == [1, 2]


# ─── calendar_events: year rollover warning + parallel earnings ─────────────

def test_upcoming_macro_warns_when_year_missing(capsys):
    stale = [("2020-01-28", "FOMC rate decision")]
    with patch.object(calendar_events, "MACRO_CALENDAR_2026", stale):
        out = calendar_events.upcoming_macro()
    assert out == []
    assert f"has no {date.today().year} entries" in capsys.readouterr().out


def test_upcoming_macro_no_warning_for_current_year(capsys):
    fixture = [(date.today().isoformat(), "CPI release")]
    with patch.object(calendar_events, "MACRO_CALENDAR_2026", fixture):
        out = calendar_events.upcoming_macro(days_ahead=5)
    assert ("CPI release" in [n for _, n in out])
    assert "update MACRO_CALENDAR" not in capsys.readouterr().out


def test_upcoming_earnings_filters_and_prints_summary(capsys):
    soon = date.today() + timedelta(days=2)
    far = date.today() + timedelta(days=30)

    def fake(sym):
        if sym == "AAPL":
            return soon
        if sym == "MSFT":
            return far
        raise RuntimeError("yfinance flake")

    with patch.object(calendar_events, "_earnings_date", side_effect=fake):
        out = calendar_events.upcoming_earnings(["AAPL", "MSFT", "BROKEN"])
    assert out == [(soon.isoformat(), "AAPL")]
    assert "2 fetched, 1 skipped" in capsys.readouterr().out


def test_upcoming_earnings_caps_universe():
    calls: list[str] = []

    def fake(sym):
        calls.append(sym)
        return None

    tickers = [f"T{i}" for i in range(calendar_events.MAX_EARNINGS_TICKERS * 2)]
    with patch.object(calendar_events, "_earnings_date", side_effect=fake):
        calendar_events.upcoming_earnings(tickers)
    assert len(calls) == calendar_events.MAX_EARNINGS_TICKERS


# ─── verify_facts: sanity gates + split verification ────────────────────────

_NAMES = list(verify_facts.CHARACTERS.keys())


def _script(n_turns: int = 10, with_disclaimer: bool = True) -> str:
    lines = [
        f"{_NAMES[i % len(_NAMES)]}: This is turn number {i} with a bit of padding text."
        for i in range(n_turns)
    ]
    if with_disclaimer:
        lines.append(f"{_NAMES[0]}: {verify_facts.DISCLAIMER_SHORT}")
    return "\n".join(lines)


def test_output_sane_accepts_faithful_output():
    s = _script()
    assert verify_facts._output_sane(s, s) is True


def test_output_sane_rejects_empty_and_truncated():
    s = _script()
    assert verify_facts._output_sane(s, "") is False
    assert verify_facts._output_sane(s, s[: len(s) // 3]) is False


def test_output_sane_rejects_dropped_disclaimer():
    s = _script()
    out = s.replace(verify_facts.DISCLAIMER_SHORT, "something else entirely instead")
    assert verify_facts._output_sane(s, out) is False


def test_output_sane_rejects_preamble_without_turns():
    s = _script()
    preamble = "Sure! Here is the verified script you asked for.\n" * 30
    assert verify_facts._output_sane(s, preamble) is False


def test_split_at_turn_boundary_preserves_content():
    s = _script(n_turns=12)
    first, second = verify_facts._split_at_turn_boundary(s)
    assert first + second == s
    assert verify_facts._TURN_RE.match(second)  # second half starts on a turn
    assert verify_facts._count_turns(first) >= 3
    assert verify_facts._count_turns(second) >= 3


def test_split_returns_none_for_single_turn():
    assert verify_facts._split_at_turn_boundary(f"{_NAMES[0]}: only turn") is None


def test_verify_keeps_original_when_llm_output_insane(monkeypatch):
    s = _script()
    monkeypatch.setattr(verify_facts, "_llm_call", lambda *a, **k: "OK")
    monkeypatch.setattr(verify_facts, "GROQ_API_KEY", "")
    monkeypatch.setattr(verify_facts, "ANTHROPIC_API_KEY", "")
    assert verify_facts.verify(s, {}, []) == s


def test_verify_returns_sane_output(monkeypatch):
    s = _script()
    verified = s.replace("turn number 3", "turn number three")
    monkeypatch.setattr(verify_facts, "_llm_call", lambda *a, **k: verified)
    monkeypatch.setattr(verify_facts, "GROQ_API_KEY", "")
    monkeypatch.setattr(verify_facts, "ANTHROPIC_API_KEY", "")
    assert verify_facts.verify(s, {}, []) == verified


def _chunk_from_prompt(prompt: str) -> str:
    return prompt.split("==== SCRIPT TO VERIFY ====\n", 1)[1].rsplit(
        "\n\n==== VERIFIED SCRIPT ====", 1
    )[0]


def test_verify_oversized_script_split_and_verified(monkeypatch):
    s = _script(n_turns=20)
    full_prompt_len = len(verify_facts._verify_prompt(s, {}, [], None))
    overhead = full_prompt_len - len(s)
    # Cap sits between half-script and full-script prompt sizes → whole script
    # is over the cap, each half is under it.
    monkeypatch.setattr(verify_facts, "VERIFY_MAX_PROMPT_CHARS", overhead + int(len(s) * 0.75))
    monkeypatch.setattr(verify_facts, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(verify_facts, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    calls: list[str] = []

    def fake_llm(prompt, *a, **k):
        chunk = _chunk_from_prompt(prompt)
        calls.append(chunk)
        return chunk  # echo the half back verbatim

    monkeypatch.setattr(verify_facts, "_llm_call", fake_llm)
    out = verify_facts.verify(s, {}, [])
    assert len(calls) == 2  # both halves went through the LLM
    assert out == s  # rejoin reproduces the script


def test_verify_oversized_unsplittable_falls_back_with_warning(monkeypatch, capsys):
    single = f"{_NAMES[0]}: one enormous turn " + "x" * 500
    monkeypatch.setattr(verify_facts, "VERIFY_MAX_PROMPT_CHARS", 100)
    monkeypatch.setattr(verify_facts, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(verify_facts, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(verify_facts, "_llm_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    out = verify_facts.verify(single, {}, [])
    assert out == single
    assert "too large" in capsys.readouterr().out
