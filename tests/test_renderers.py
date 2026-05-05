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
