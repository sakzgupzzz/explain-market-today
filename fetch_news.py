"""Aggregate RSS headlines across beats (markets, business, tech, world, culture)."""
from __future__ import annotations
import time
import re
from datetime import datetime, timezone, timedelta
import feedparser
from config import RSS_FEEDS_BY_CATEGORY, HEADLINES_PER_CATEGORY


def _entry_dt(e) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        v = getattr(e, key, None) or e.get(key)
        if v:
            return datetime.fromtimestamp(time.mktime(v), tz=timezone.utc)
    return None


def _clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def _fetch_category(category: str, feeds: list[str], hours: int, cap: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items: list[dict] = []
    seen: set[str] = set()
    for url in feeds:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        source = feed.feed.get("title", url)
        for e in feed.entries[:25]:
            title = _clean(e.get("title", ""))
            key = title.lower()
            if not title or key in seen:
                continue
            seen.add(key)
            dt = _entry_dt(e)
            if dt and dt < cutoff:
                continue
            items.append({
                "category": category,
                "title": title,
                "summary": _clean(e.get("summary", ""))[:400],
                "source": source,
                "published": dt.isoformat() if dt else None,
                "link": e.get("link", ""),
            })
    items.sort(key=lambda x: x["published"] or "", reverse=True)
    return items[:cap]


def fetch_headlines(hours: int = 24) -> dict[str, list[dict]]:
    """Return headlines grouped by category."""
    out: dict[str, list[dict]] = {}
    for cat, feeds in RSS_FEEDS_BY_CATEGORY.items():
        cap = HEADLINES_PER_CATEGORY.get(cat, 10)
        out[cat] = _fetch_category(cat, feeds, hours, cap)
    return out


def flatten(headlines: dict[str, list[dict]]) -> list[dict]:
    flat: list[dict] = []
    for cat, items in headlines.items():
        flat.extend(items)
    return flat


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_headlines(), indent=2))
