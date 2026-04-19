"""Aggregate RSS headlines from free finance sources."""
from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta
import feedparser
from config import RSS_FEEDS, HEADLINE_LIMIT


def _entry_dt(e) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        v = getattr(e, key, None) or e.get(key)
        if v:
            return datetime.fromtimestamp(time.mktime(v), tz=timezone.utc)
    return None


def fetch_headlines(hours: int = 24) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items = []
    seen_titles = set()
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        source = feed.feed.get("title", url)
        for e in feed.entries[:30]:
            title = (e.get("title") or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            dt = _entry_dt(e)
            if dt and dt < cutoff:
                continue
            summary = (e.get("summary") or "").strip()
            # strip html crudely
            if "<" in summary:
                import re
                summary = re.sub(r"<[^>]+>", "", summary)
            items.append({
                "title": title,
                "summary": summary[:400],
                "source": source,
                "published": dt.isoformat() if dt else None,
                "link": e.get("link", ""),
            })
    # newest first, keep top N
    items.sort(key=lambda x: x["published"] or "", reverse=True)
    return items[:HEADLINE_LIMIT]


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_headlines(), indent=2))
