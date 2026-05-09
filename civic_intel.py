"""Civic intelligence wrapper around the civicledger library.

Pulls public-domain US financial signal in one shot and shapes it for
the script-generation pipeline:

  - macro_today / macro_lookahead   — FRED economic event calendar
  - earnings_today / earnings_lookahead — SEC EDGAR 8-K Item 2.02 calendar
  - insider_trades                  — Form 4 filings on watchlist tickers
  - material_events                 — 8-K filings on watchlist tickers
  - congress_trades                 — House congressional trades

All upstream calls are async; this module provides a sync entrypoint
(``fetch_civic_today``) that runs them in parallel via asyncio.gather and
returns a plain dict for downstream consumers.

Watchlist filtering keeps payloads small and on-topic. If FRED_API_KEY is
not set, the macro lanes silently return empty lists — the rest of the
pipeline still works.

Civicledger source: github.com/sakzgupzzz/civicledger
"""
from __future__ import annotations
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from civicledger.economic.fred import fetch_economic_events
    from civicledger.edgar.earnings import fetch_earnings
    from civicledger.edgar.insider_trades import fetch_recent_insider_trades
    from civicledger.edgar.material_events import fetch_material_events
    from civicledger.congress.trades import fetch_all_congressional_trades
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _date_offset(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _filter_to_watchlist(items: list[dict], watchlist: set[str], key: str = "ticker") -> list[dict]:
    if not watchlist:
        return items
    out = []
    for it in items:
        sym = (it.get(key) or "").upper()
        if sym in watchlist:
            out.append(it)
    return out


async def _safe(coro, label: str) -> Any:
    """Wrap one upstream call so a single API outage doesn't kill the lot."""
    try:
        return await coro
    except Exception as e:
        print(f"[civic] {label} failed: {type(e).__name__}: {e}")
        return []


async def _fetch_async(
    watchlist: set[str],
    lookahead_days: int,
    insider_lookback_days: int,
    congress_lookback_days: int,
) -> dict:
    today = _today()
    week_ahead = _date_offset(lookahead_days)
    insider_from = _date_offset(-insider_lookback_days)
    has_fred = bool(os.environ.get("FRED_API_KEY"))

    tasks = []
    labels = []

    if has_fred:
        tasks.append(_safe(fetch_economic_events(today, today), "macro_today"))
        labels.append("macro_today")
        tasks.append(_safe(fetch_economic_events(today, week_ahead), "macro_lookahead"))
        labels.append("macro_lookahead")
    else:
        tasks.append(_noop())
        labels.append("macro_today")
        tasks.append(_noop())
        labels.append("macro_lookahead")

    tasks.append(_safe(fetch_earnings(today, today), "earnings_today"))
    labels.append("earnings_today")
    tasks.append(_safe(fetch_earnings(today, week_ahead), "earnings_lookahead"))
    labels.append("earnings_lookahead")

    tasks.append(_safe(fetch_recent_insider_trades(insider_from, today, limit=200), "insider_trades"))
    labels.append("insider_trades")

    tasks.append(_safe(fetch_material_events(insider_from, today), "material_events"))
    labels.append("material_events")

    year = datetime.now(timezone.utc).year
    tasks.append(_safe(fetch_all_congressional_trades(year=year, limit=500), "congress_trades"))
    labels.append("congress_trades")

    results = await asyncio.gather(*tasks)
    raw = dict(zip(labels, results))

    # Material events: many filings have a ticker, filter to watchlist or
    # fall through to the top of the unfiltered list (8-Ks are interesting
    # even outside the watchlist — e.g. unexpected CEO change at SBNY).
    me_watch = _filter_to_watchlist(raw["material_events"] or [], watchlist)
    raw["material_events"] = (me_watch or (raw["material_events"] or []))[:25]

    # Insider trades: civicledger frequently leaves ticker=None on Form 4
    # (it's parsed from the filing index, not the body). Filter to watchlist
    # but DON'T zero-out when nothing matches — fall back to the most-recent
    # tagged subset so the prompt still has signal.
    it_full = raw["insider_trades"] or []
    it_watch = _filter_to_watchlist(it_full, watchlist)
    it_with_ticker = [t for t in it_full if (t.get("ticker") or "").strip()]
    raw["insider_trades"] = (it_watch or it_with_ticker[:15])

    # Earnings calendar — same fall-through.
    et_watch = _filter_to_watchlist(raw["earnings_today"] or [], watchlist)
    raw["earnings_today"] = et_watch or (raw["earnings_today"] or [])[:8]
    el_watch = _filter_to_watchlist(raw["earnings_lookahead"] or [], watchlist)
    raw["earnings_lookahead"] = el_watch or (raw["earnings_lookahead"] or [])[:8]

    # Congress trades: ticker is also frequently null (PDF-parse upstream).
    # Use disclosure_date for recency, fall back to ticker-tagged sample.
    cg = raw["congress_trades"] or []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=congress_lookback_days)).date()
    def _recent(t: dict) -> bool:
        d = (t.get("disclosure_date") or t.get("transaction_date")
             or t.get("traded") or t.get("trade_date") or "")
        try:
            # Both YYYY-MM-DD and M/D/YYYY appear upstream
            d10 = d[:10]
            for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                try:
                    return datetime.strptime(d10, fmt).date() >= cutoff
                except ValueError:
                    continue
        except (ValueError, TypeError):
            pass
        return False
    cg_recent = [t for t in cg if _recent(t)]
    cg_watch = _filter_to_watchlist(cg_recent, watchlist)
    cg_with_ticker = [t for t in cg_recent if (t.get("ticker") or "").strip()]
    raw["congress_trades"] = (cg_watch or cg_with_ticker or cg_recent)[:25]

    return raw


async def _noop() -> list:
    return []


def fetch_civic_today(
    watchlist: list[str] | set[str] | None = None,
    lookahead_days: int = 7,
    insider_lookback_days: int = 1,
    congress_lookback_days: int = 7,
) -> dict:
    """Synchronous entrypoint. Runs all civicledger lanes in parallel and
    returns a plain dict. Empty dict when civicledger is not installed."""
    if not _AVAILABLE:
        print("[civic] civicledger not installed; civic_intel.fetch_civic_today returning empty")
        return {
            "macro_today": [], "macro_lookahead": [],
            "earnings_today": [], "earnings_lookahead": [],
            "insider_trades": [], "material_events": [], "congress_trades": [],
        }
    wl: set[str] = {t.upper() for t in (watchlist or [])}
    return asyncio.run(
        _fetch_async(wl, lookahead_days, insider_lookback_days, congress_lookback_days)
    )


# ─── prompt-block formatters ────────────────────────────────────────────────


def _fmt_macro(events: list[dict], top: int = 6) -> str:
    if not events:
        return "(no macro releases)"
    out = []
    for e in events[:top]:
        date = e.get("date") or e.get("release_date") or ""
        name = e.get("name") or e.get("release_name") or ""
        out.append(f"- {date} · {name}")
    return "\n".join(out)


def _fmt_earnings(events: list[dict], top: int = 8) -> str:
    if not events:
        return "(no earnings)"
    out = []
    for e in events[:top]:
        date = e.get("filing_date") or e.get("date") or ""
        ticker = e.get("ticker") or ""
        company = (e.get("company") or "")[:40]
        out.append(f"- {date} · {ticker} · {company}")
    return "\n".join(out)


def _fmt_insider(events: list[dict], top: int = 8) -> str:
    if not events:
        return "(no insider activity)"
    out = []
    for e in events[:top]:
        date = e.get("filing_date") or e.get("transaction_date") or ""
        ticker = (e.get("ticker") or "?").upper()
        company = (e.get("company") or "")[:35]
        insider = (e.get("insider_name") or e.get("insider") or e.get("reporter") or "")[:25]
        ttype = e.get("transaction_type") or ""
        meta = f" · {insider}" if insider else ""
        meta += f" · {ttype}" if ttype else ""
        out.append(f"- {date} · {ticker} · {company}{meta}")
    return "\n".join(out)


def _fmt_material(events: list[dict], top: int = 6) -> str:
    if not events:
        return "(no 8-K material events)"
    out = []
    for e in events[:top]:
        date = e.get("filing_date") or ""
        ticker = (e.get("ticker") or "?").upper()
        company = (e.get("company") or "")[:30]
        labels = e.get("item_labels") or []
        items = e.get("items") or []
        if labels:
            label = "/".join(labels)[:50]
        elif items:
            label = "Item " + "/".join(items)[:30]
        else:
            label = (e.get("title") or e.get("description") or "")[:60]
        out.append(f"- {date} · {ticker} · {company} · {label}")
    return "\n".join(out)


def _fmt_congress(events: list[dict], top: int = 6) -> str:
    if not events:
        return "(no congressional trades)"
    out = []
    for e in events[:top]:
        traded = (e.get("transaction_date") or e.get("disclosure_date")
                  or e.get("traded") or e.get("trade_date") or "")
        ticker = (e.get("ticker") or "?").upper()
        politician = (e.get("politician") or e.get("member") or e.get("name") or "")[:30]
        chamber = e.get("chamber") or ""
        ttype = e.get("transaction_type") or e.get("type") or ""
        amt = e.get("amount_range") or e.get("amount") or ""
        bits = [b for b in (politician, chamber, ttype, amt) if b]
        out.append(f"- {traded} · {ticker} · " + " · ".join(bits))
    return "\n".join(out)


def format_for_prompt(civic: dict) -> str:
    """Compact multi-section block ready to drop into an LLM prompt."""
    if not any(civic.values()):
        return ""
    sections = [
        "MACRO TODAY:\n" + _fmt_macro(civic.get("macro_today") or []),
        "MACRO LOOKAHEAD (next 7d):\n" + _fmt_macro(civic.get("macro_lookahead") or []),
        "EARNINGS TODAY:\n" + _fmt_earnings(civic.get("earnings_today") or []),
        "EARNINGS LOOKAHEAD (next 7d):\n" + _fmt_earnings(civic.get("earnings_lookahead") or []),
        "INSIDER TRADES (last 24h):\n" + _fmt_insider(civic.get("insider_trades") or []),
        "8-K MATERIAL EVENTS:\n" + _fmt_material(civic.get("material_events") or []),
        "CONGRESS TRADES (last 7d):\n" + _fmt_congress(civic.get("congress_trades") or []),
    ]
    return "\n\n".join(sections)


def lookahead_block(civic: dict) -> str:
    """Just the lookahead lanes — used by the render_lookahead beat."""
    parts = []
    macro = civic.get("macro_lookahead") or []
    earn = civic.get("earnings_lookahead") or []
    if macro:
        parts.append("MACRO RELEASES (next 7d):\n" + _fmt_macro(macro, top=5))
    if earn:
        parts.append("EARNINGS (next 7d):\n" + _fmt_earnings(earn, top=6))
    return "\n\n".join(parts)


if __name__ == "__main__":
    civic = fetch_civic_today(watchlist=["AAPL", "NVDA", "MSFT"])
    for k, v in civic.items():
        print(f"{k}: {len(v)} item(s)")
    print()
    print(format_for_prompt(civic))
