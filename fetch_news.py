"""Aggregate RSS headlines across beats (markets, business, tech, world, culture).
Fetches feeds in parallel for speed."""
from __future__ import annotations
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _fetch_one_feed(url: str, cutoff: datetime, category: str) -> list[dict]:
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    source = feed.feed.get("title", url)
    out: list[dict] = []
    for e in feed.entries[:25]:
        title = _clean(e.get("title", ""))
        if not title:
            continue
        dt = _entry_dt(e)
        if dt and dt < cutoff:
            continue
        out.append({
            "category": category,
            "title": title,
            "summary": _clean(e.get("summary", ""))[:400],
            "source": source,
            "published": dt.isoformat() if dt else None,
            "link": e.get("link", ""),
        })
    return out


def _fetch_category(category: str, feeds: list[str], hours: int, cap: int) -> list[dict]:
    """Fetch all feeds in a category in parallel, dedupe by title, cap, sort by recency."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(feeds) or 1)) as pool:
        futures = [pool.submit(_fetch_one_feed, url, cutoff, category) for url in feeds]
        for fut in as_completed(futures):
            try:
                items.extend(fut.result())
            except Exception:
                continue
    seen: set[str] = set()
    deduped: list[dict] = []
    for it in items:
        key = it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    deduped.sort(key=lambda x: x["published"] or "", reverse=True)
    return deduped[:cap]


def fetch_headlines(hours: int = 24) -> dict[str, list[dict]]:
    """Return headlines grouped by category. Categories run in parallel too."""
    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(RSS_FEEDS_BY_CATEGORY)) as pool:
        future_to_cat = {
            pool.submit(_fetch_category, cat, feeds, hours, HEADLINES_PER_CATEGORY.get(cat, 10)): cat
            for cat, feeds in RSS_FEEDS_BY_CATEGORY.items()
        }
        for fut in as_completed(future_to_cat):
            cat = future_to_cat[fut]
            try:
                out[cat] = fut.result()
            except Exception:
                out[cat] = []
    # preserve canonical order
    return {cat: out.get(cat, []) for cat in RSS_FEEDS_BY_CATEGORY.keys()}


def flatten(headlines: dict[str, list[dict]]) -> list[dict]:
    flat: list[dict] = []
    for cat, items in headlines.items():
        flat.extend(items)
    return flat


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_headlines(), indent=2))
