"""Aggregate RSS headlines across beats (markets, business, tech, world, culture).
Fetches feeds in parallel for speed. Tracks per-feed health in
.feed_health.json so persistently-failing feeds get auto-skipped."""
from __future__ import annotations
import calendar
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
import feedparser
import requests
from config import ROOT, RSS_FEEDS_BY_CATEGORY, HEADLINES_PER_CATEGORY

FEED_HEALTH_PATH = ROOT / ".feed_health.json"
FAIL_THRESHOLD = 5  # consecutive failures before a feed is skipped
FEED_TIMEOUT_SEC = 15  # per-feed HTTP timeout so one hung server can't stall the run
RETRY_COOLDOWN_HOURS = 24  # disabled feeds get one retry attempt per cooldown window


def _load_health() -> dict:
    if not FEED_HEALTH_PATH.exists():
        return {}
    try:
        return json.loads(FEED_HEALTH_PATH.read_text())
    except Exception:
        return {}


def _save_health(health: dict) -> None:
    try:
        FEED_HEALTH_PATH.write_text(json.dumps(health, indent=2, sort_keys=True))
    except Exception:
        pass


def _record_outcome(health: dict, url: str, ok: bool, err: str = "") -> None:
    rec = health.setdefault(url, {"fail_streak": 0, "last_error": "", "last_seen": ""})
    if ok:
        rec["fail_streak"] = 0
        rec["last_seen"] = datetime.now(timezone.utc).isoformat()
        rec["last_error"] = ""
        rec.pop("retry_after", None)
    else:
        rec["fail_streak"] = rec.get("fail_streak", 0) + 1
        rec["last_error"] = err[:200]
        if rec["fail_streak"] >= FAIL_THRESHOLD:
            # Cooldown rather than permanent kill: the feed gets one retry
            # attempt per RETRY_COOLDOWN_HOURS so a flaky week can't disable
            # it forever. Success resets the streak and clears the cooldown.
            rec["retry_after"] = (
                datetime.now(timezone.utc) + timedelta(hours=RETRY_COOLDOWN_HOURS)
            ).isoformat()


def _is_feed_disabled(health: dict, url: str) -> bool:
    rec = health.get(url, {})
    if rec.get("fail_streak", 0) < FAIL_THRESHOLD:
        return False
    retry_after = rec.get("retry_after")
    if not retry_after:
        return False  # legacy record without cooldown — allow the retry
    try:
        return datetime.now(timezone.utc) < datetime.fromisoformat(retry_after)
    except ValueError:
        return False


def _entry_dt(e) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        v = getattr(e, key, None) or e.get(key)
        if v:
            # feedparser normalizes struct_times to UTC; timegm reads them as
            # UTC (time.mktime would apply the local offset and shift hours).
            return datetime.fromtimestamp(calendar.timegm(v), tz=timezone.utc)
    return None


def _clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def _fetch_one_feed(url: str, cutoff: datetime, category: str, health: dict | None = None) -> list[dict]:
    if health is not None and _is_feed_disabled(health, url):
        return []
    try:
        # Fetch ourselves with a timeout — feedparser.parse(url) has no network
        # timeout, so one hung server would block a worker thread forever.
        resp = requests.get(
            url, timeout=FEED_TIMEOUT_SEC,
            headers={"User-Agent": "explain-market-today/1.0 (+rss aggregator)"},
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        bozo = bool(getattr(feed, "bozo", False) and not feed.entries)
        if bozo:
            err = str(getattr(feed, "bozo_exception", "feedparser bozo"))
            if health is not None:
                _record_outcome(health, url, ok=False, err=err)
            return []
    except Exception as e:
        if health is not None:
            _record_outcome(health, url, ok=False, err=str(e))
        return []
    if health is not None:
        _record_outcome(health, url, ok=True)
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


def _fetch_category(category: str, feeds: list[str], hours: int, cap: int, health: dict | None = None) -> list[dict]:
    """Fetch all feeds in a category in parallel, dedupe by title, cap, sort by recency."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(feeds) or 1)) as pool:
        futures = [pool.submit(_fetch_one_feed, url, cutoff, category, health) for url in feeds]
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
    """Return headlines grouped by category. Categories run in parallel too.
    Per-feed health is loaded from .feed_health.json, updated with this run's
    outcomes, and saved back. Feeds with FAIL_THRESHOLD consecutive failures
    are skipped, then retried once per RETRY_COOLDOWN_HOURS; a success
    re-enables the feed."""
    health = _load_health()
    disabled = [u for u in (health.keys()) if _is_feed_disabled(health, u)]
    if disabled:
        print(f"[fetch_news] {len(disabled)} feed(s) auto-disabled (>={FAIL_THRESHOLD} fails)")
    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(RSS_FEEDS_BY_CATEGORY)) as pool:
        future_to_cat = {
            pool.submit(_fetch_category, cat, feeds, hours, HEADLINES_PER_CATEGORY.get(cat, 10), health): cat
            for cat, feeds in RSS_FEEDS_BY_CATEGORY.items()
        }
        for fut in as_completed(future_to_cat):
            cat = future_to_cat[fut]
            try:
                out[cat] = fut.result()
            except Exception:
                out[cat] = []
    _save_health(health)
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
