"""Twitter/X / Bluesky thread renderer — JSON artifact, no autopost.

Generates a 5-tweet thread from the same ranked story list. Output:
docs/threads/YYYY-MM-DD.json. Each tweet ≤ 280 chars. Hand-post or wire to
a posting agent later.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from config import DOCS, PODCAST_BASE_URL


THREAD_DIR = DOCS / "threads"
TWEET_LIMIT = 280


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1].rsplit(" ", 1)[0] + "…"


def _market_summary(market: dict) -> str:
    parts = []
    for r in (market.get("indices") or [])[:3]:
        sign = "▲" if r["pct"] >= 0 else "▼"
        parts.append(f"{r['name']} {sign}{abs(r['pct']):.2f}%")
    return " · ".join(parts)


def render_thread(market: dict, ranked: list[dict], date_str: str) -> list[str]:
    tweets: list[str] = []

    # 1. Hook — top story
    if ranked:
        top = ranked[0].get("title", "")
        tweets.append(_truncate(f"📍 {date_str}: {top}", TWEET_LIMIT))
    else:
        tweets.append(f"📍 {date_str}: Daily roundup")

    # 2. Market summary
    summary = _market_summary(market)
    if summary:
        tweets.append(_truncate(f"Markets: {summary}", TWEET_LIMIT))

    # 3-4. Next two stories
    for c in ranked[1:3]:
        title = c.get("title", "")
        sources = " · ".join((c.get("sources") or [])[:2])
        body = f"{title}\n\n— {sources}" if sources else title
        tweets.append(_truncate(body, TWEET_LIMIT))

    # 5. Listen link
    audio_url = f"{PODCAST_BASE_URL}/episodes/{date_str}.mp3"
    tweets.append(_truncate(f"Full show + transcript: {PODCAST_BASE_URL}", TWEET_LIMIT))

    return tweets


def write_thread(market: dict, ranked: list[dict], date_str: str) -> Path:
    THREAD_DIR.mkdir(parents=True, exist_ok=True)
    tweets = render_thread(market, ranked, date_str)
    out = THREAD_DIR / f"{date_str}.json"
    out.write_text(json.dumps({"date": date_str, "tweets": tweets}, indent=2))
    return out


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines, flatten
    from cluster import cluster_headlines
    from score import score_clusters
    from interests_loader import load_interests
    today = datetime.now().strftime("%Y-%m-%d")
    m = fetch_all()
    h = fetch_headlines()
    ranked = score_clusters(cluster_headlines(flatten(h)), m, load_interests())
    for i, tw in enumerate(render_thread(m, ranked, today), 1):
        print(f"--- tweet {i} ({len(tw)} chars) ---")
        print(tw)
        print()
