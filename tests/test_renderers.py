"""Smoke tests for the deterministic (no-LLM) renderers."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import render_email
import render_thread


def _market():
    return {
        "indices": [{"symbol": "^GSPC", "name": "S&P 500", "close": 5400.0, "prev_close": 5350.0, "pct": 0.93}],
        "sectors": [],
        "macro": [],
        "gainers": [{"symbol": "AAPL", "name": "AAPL", "close": 200.0, "prev_close": 190.0, "pct": 5.26}],
        "losers": [{"symbol": "TSLA", "name": "TSLA", "close": 180.0, "prev_close": 200.0, "pct": -10.0}],
    }


def _ranked():
    return [
        {"id": "a", "title": "Apple beats earnings", "summary": "Quarterly results crushed estimates.", "sources": ["Reuters", "AP"], "categories": ["business"], "published": "", "cluster_size": 2, "headlines": [], "link": "https://example.com/a", "score": 12.0},
        {"id": "b", "title": "Tesla recall", "summary": "", "sources": ["AP"], "categories": ["business"], "published": "", "cluster_size": 1, "headlines": [], "link": "https://example.com/b", "score": 6.0},
    ]


def test_email_digest_contains_top_story_title():
    body = render_email.render_email_digest(_market(), _ranked(), "2026-05-05")
    assert "Apple beats earnings" in body
    assert "Tesla recall" in body


def test_email_digest_includes_market_indices():
    body = render_email.render_email_digest(_market(), _ranked(), "2026-05-05")
    assert "S&P 500" in body
    assert "+0.93%" in body


def test_email_digest_links_to_audio():
    body = render_email.render_email_digest(_market(), _ranked(), "2026-05-05")
    assert "/episodes/2026-05-05.mp3" in body
    assert "/express/2026-05-05.mp3" in body


def test_email_digest_includes_disclaimer():
    body = render_email.render_email_digest(_market(), _ranked(), "2026-05-05")
    assert "investment, financial, legal, or tax advice" in body


def test_thread_renderer_produces_at_least_3_tweets():
    tweets = render_thread.render_thread(_market(), _ranked(), "2026-05-05")
    assert len(tweets) >= 3


def test_thread_tweets_under_280_chars():
    tweets = render_thread.render_thread(_market(), _ranked(), "2026-05-05")
    for t in tweets:
        assert len(t) <= 280


def test_thread_first_tweet_mentions_top_story():
    tweets = render_thread.render_thread(_market(), _ranked(), "2026-05-05")
    assert "Apple beats earnings" in tweets[0]


def test_thread_closing_tweet_links_the_episode():
    from config import PODCAST_BASE_URL
    tweets = render_thread.render_thread(_market(), _ranked(), "2026-05-05")
    # Closing tweet must link TODAY'S episode page, not just the site root.
    assert f"{PODCAST_BASE_URL}/episodes/2026-05-05.html" in tweets[-1]


# ─── build_feed smoke ───────────────────────────────────────────────────────


def test_build_feed_smoke_with_and_without_meta_sidecars(tmp_path, monkeypatch):
    import publish
    docs = tmp_path / "docs"
    eps = docs / "episodes"
    eps.mkdir(parents=True)
    # Episode WITH sidecars: meta (duration + pubDate source) and script.
    (eps / "2026-07-01.mp3").write_bytes(b"\x00" * 256)
    (eps / "2026-07-01.txt").write_text(
        "JAMIE: Markets went up today because of vibes and momentum trades.\n"
    )
    (eps / "2026-07-01.meta.json").write_text(json.dumps({
        "generated_at": "2026-07-01T21:05:00Z",
        "duration_sec": 312.5,
        "turns": 20,
        "char_usage_estimate": 4000,
    }))
    # Episode WITHOUT any sidecars (bare mp3, unreadable by ffprobe) — feed
    # build must degrade (duration 0) instead of crashing.
    (eps / "2026-07-02.mp3").write_bytes(b"\x00" * 256)

    monkeypatch.setattr(publish, "EPISODES_DIR", eps)
    monkeypatch.setattr(publish, "FEED_PATH", docs / "feed.xml")
    monkeypatch.setattr(publish, "DOCS", docs)
    publish.build_feed()

    feed = (docs / "feed.xml").read_text()
    assert "2026-07-01.mp3" in feed
    assert "2026-07-02.mp3" in feed
    # duration came from the meta sidecar (312.5s → 00:05:12), no ffprobe
    assert "00:05:12" in feed
    # meta generated_at drives the pubDate for the sidecar'd episode
    assert "1 Jul 2026 21:05:00" in feed


# ── express narrator rotation ────────────────────────────────────────────────

def test_express_narrator_rotation_deterministic():
    from render_express import pick_express_narrator, EXPRESS_NARRATORS
    # Same date → same narrator (resume/--force safe)
    assert pick_express_narrator("2026-08-12") == pick_express_narrator("2026-08-12")
    # Three consecutive days cover all three hosts
    got = {pick_express_narrator(d) for d in ("2026-08-10", "2026-08-11", "2026-08-12")}
    assert got == set(EXPRESS_NARRATORS)


def test_express_prompt_uses_narrator_name():
    from render_express import build_express_prompt
    market = {"indices": [], "gainers": [], "losers": []}
    prompt = build_express_prompt(market, [], "Wednesday, August 12, 2026", narrator="ALEX")
    assert "`ALEX: text`" in prompt
    assert "`JAMIE: text`" not in prompt
    assert "starts with `ALEX:`" in prompt


def test_jamie_cap_skipped_for_single_narrator():
    from sanitize import _enforce_jamie_cap
    # 10 solo JAMIE turns incl. short ones — cap must not delete any
    turns = [("JAMIE", f"line {i} ok")for i in range(10)]
    out, dropped = _enforce_jamie_cap(turns)
    assert dropped == 0
    assert len(out) == 10
