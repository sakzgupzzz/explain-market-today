"""Score and rank clustered stories by importance.

The scoring formula is intentionally simple and additive so it's debuggable:

    raw = keyword_weight + cluster_size_weight + ticker_mover_weight
        + interest_keyword_weight + interest_sector_weight
    score = raw * source_quality * recency_decay

Personalization (interests.yaml) feeds in three ways:
  1. ticker watchlist → mover_weight gets a hard +3 boost on match
  2. keyword watchlist → +2 per keyword present
  3. sector watchlist → +1.5 per sector mention
  4. blocked topics/sources → cluster dropped entirely

Output is sorted descending by score.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone

# News patterns that signal "this matters more than X went up 0.3%"
_KEYWORD_WEIGHTS = {
    "earnings": 3.0, "guidance": 2.5, "revenue": 2.0, "miss": 2.5, "beat": 2.0,
    "ipo": 4.0, "merger": 4.5, "acquisition": 4.5, "buyout": 4.0, "spin off": 3.0,
    "bankruptcy": 5.0, "chapter 11": 5.0, "shutdown": 4.0, "layoffs": 3.5, "cuts": 2.5,
    "fed": 4.0, "fomc": 4.5, "interest rate": 4.0, "rate hike": 4.0, "rate cut": 4.0,
    "inflation": 3.5, "cpi": 4.0, "ppi": 3.5, "jobs report": 3.5, "unemployment": 3.0,
    "ceo": 2.0, "resigns": 3.0, "fired": 3.0, "departure": 2.5,
    "lawsuit": 2.5, "scandal": 3.0, "fraud": 4.0, "indictment": 4.0,
    "antitrust": 3.5, "regulator": 2.5, "ban": 2.5, "tariff": 3.5, "sanctions": 3.5,
    "fda": 3.0, "approval": 2.5, "recall": 3.0,
    "crash": 4.0, "surge": 2.0, "plunge": 3.0, "rally": 1.5, "selloff": 2.5,
    "outage": 2.5, "breach": 3.0, "hack": 3.0, "ransomware": 3.5,
    "war": 4.0, "strike": 2.5, "election": 3.0, "supreme court": 3.0,
}

# Trusted source baseline; defaults to 0.6 for anything not listed.
_SOURCE_WEIGHTS = {
    "Reuters": 1.0, "Associated Press": 1.0, "AP": 1.0, "AP News": 1.0,
    "Bloomberg": 1.0, "WSJ": 1.0, "Wall Street Journal": 1.0,
    "Federal Reserve": 1.0, "CNBC": 0.9, "Yahoo Finance": 0.75,
    "MarketWatch": 0.85, "TechCrunch": 0.85, "The Verge": 0.85,
    "Ars Technica": 0.85, "BBC": 0.95, "BBC News": 0.95, "NPR": 0.85,
    "Hacker News": 0.6, "Seeking Alpha": 0.7, "Axios": 0.85,
    "The Atlantic": 0.85,
}


def _keyword_score(text: str) -> float:
    text_l = text.lower()
    return sum(w for kw, w in _KEYWORD_WEIGHTS.items() if kw in text_l)


def _source_score(sources: list[str]) -> float:
    if not sources:
        return 0.6
    return max(_SOURCE_WEIGHTS.get(s, 0.6) for s in sources)


def _recency_decay(published: str) -> float:
    if not published:
        return 0.5
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        if hours <= 6:
            return 1.0
        if hours <= 18:
            return 0.85
        if hours <= 36:
            return 0.65
        return 0.4
    except Exception:
        return 0.5


def _ticker_match(text: str, sym: str) -> bool:
    return bool(re.search(rf"\b{re.escape(sym)}\b", text))


def _ticker_mover_boost(cluster: dict, market: dict) -> float:
    """Boost when cluster mentions a ticker that moved >2% today."""
    text_upper = (cluster.get("title", "") + " " + cluster.get("summary", "")).upper()
    boost = 0.0
    for m in (market.get("gainers") or []) + (market.get("losers") or []):
        sym = m.get("symbol") or ""
        if sym and _ticker_match(text_upper, sym):
            boost += min(abs(m.get("pct", 0.0)) / 2.0, 5.0)
    return boost


def score_clusters(
    clusters: list[dict],
    market: dict,
    interests: dict | None = None,
) -> list[dict]:
    """Score and sort. Drops blocked clusters. Returns clusters with `score`."""
    interests = interests or {}
    watchlist = interests.get("watchlist") or {}
    blocked = interests.get("blocked") or {}
    wl_tickers = {t.upper() for t in (watchlist.get("tickers") or [])}
    wl_keywords = [k.lower() for k in (watchlist.get("keywords") or [])]
    wl_sectors = [s.lower() for s in (watchlist.get("sectors") or [])]
    bl_topics = [t.lower() for t in (blocked.get("topics") or [])]
    bl_sources = {s.lower() for s in (blocked.get("sources") or [])}

    out: list[dict] = []
    for c in clusters:
        text = c.get("title", "") + " " + c.get("summary", "")
        text_l = text.lower()
        text_u = text.upper()

        # blocking gates first
        if any(t in text_l for t in bl_topics):
            continue
        if any(s.lower() in bl_sources for s in (c.get("sources") or [])):
            continue

        kw = _keyword_score(text)
        cluster_w = min(c.get("cluster_size", 1), 5) * 0.6
        mover_w = _ticker_mover_boost(c, market)
        wl_ticker_w = sum(3.0 for t in wl_tickers if _ticker_match(text_u, t))
        wl_kw_w = sum(2.0 for k in wl_keywords if k in text_l)
        wl_sec_w = sum(1.5 for s in wl_sectors if s in text_l)

        raw = kw + cluster_w + mover_w + wl_ticker_w + wl_kw_w + wl_sec_w
        modifier = _source_score(c.get("sources") or []) * _recency_decay(c.get("published", ""))
        score = round(raw * modifier, 3)

        c2 = dict(c)
        c2.pop("_tokens", None)
        c2["score"] = score
        c2["_score_components"] = {
            "keywords": round(kw, 2),
            "cluster_size": round(cluster_w, 2),
            "movers": round(mover_w, 2),
            "watchlist_ticker": round(wl_ticker_w, 2),
            "watchlist_keyword": round(wl_kw_w, 2),
            "watchlist_sector": round(wl_sec_w, 2),
            "source_quality": round(_source_score(c.get("sources") or []), 2),
            "recency": round(_recency_decay(c.get("published", "")), 2),
        }
        out.append(c2)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


if __name__ == "__main__":
    from fetch_news import fetch_headlines, flatten
    from fetch_market import fetch_all
    from cluster import cluster_headlines
    market = fetch_all()
    clusters = cluster_headlines(flatten(fetch_headlines()))
    ranked = score_clusters(clusters, market, interests=None)
    print(f"{len(clusters)} clusters → top 12 by score:")
    for c in ranked[:12]:
        sources = ", ".join(c["sources"][:3])
        print(f"  {c['score']:>6.2f}  [{c['cluster_size']}] {c['title'][:75]}  ({sources})")
