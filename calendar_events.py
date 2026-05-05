"""Upcoming earnings + macro events for context in episode prompts.

Earnings: yfinance Ticker.calendar (free, no API key). For each ticker in
MOVERS_UNIVERSE + interests.watchlist.tickers, surface the next earnings
date if it's within the next 7 days.

Macro: hardcoded FOMC + key BLS / BEA dates for 2026. Pulled once at module
load. Easy to update — single dict.

Both feeds into the prompt as a small `UPCOMING_EVENTS` block so the LLM
can frame today's coverage with anticipation context.
"""
from __future__ import annotations
from datetime import datetime, date, timezone, timedelta
from config import MOVERS_UNIVERSE


# 2026 macro calendar — sourced from federalreserve.gov/monetarypolicy/fomccalendars
# and bls.gov / bea.gov release schedules. Update annually.
MACRO_CALENDAR_2026: list[tuple[str, str]] = [
    # FOMC meetings (rate decisions on the 2nd day each)
    ("2026-01-28", "FOMC rate decision"),
    ("2026-03-18", "FOMC rate decision"),
    ("2026-04-29", "FOMC rate decision"),
    ("2026-06-17", "FOMC rate decision"),
    ("2026-07-29", "FOMC rate decision"),
    ("2026-09-16", "FOMC rate decision"),
    ("2026-10-28", "FOMC rate decision"),
    ("2026-12-16", "FOMC rate decision"),
    # CPI release dates (BLS, ~10th-15th of month, 8:30 ET)
    ("2026-05-13", "CPI release"),
    ("2026-06-11", "CPI release"),
    ("2026-07-15", "CPI release"),
    ("2026-08-12", "CPI release"),
    ("2026-09-11", "CPI release"),
    ("2026-10-15", "CPI release"),
    ("2026-11-12", "CPI release"),
    ("2026-12-10", "CPI release"),
    # Jobs reports (1st Friday of month)
    ("2026-05-01", "Jobs report"),
    ("2026-06-05", "Jobs report"),
    ("2026-07-02", "Jobs report"),
    ("2026-08-07", "Jobs report"),
    ("2026-09-04", "Jobs report"),
    ("2026-10-02", "Jobs report"),
    ("2026-11-06", "Jobs report"),
    ("2026-12-04", "Jobs report"),
    # GDP releases
    ("2026-04-30", "Q1 GDP advance estimate"),
    ("2026-07-30", "Q2 GDP advance estimate"),
    ("2026-10-29", "Q3 GDP advance estimate"),
]


def upcoming_macro(days_ahead: int = 5) -> list[tuple[str, str]]:
    """Return (date_str, name) tuples within `days_ahead` days from today."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    out = []
    for date_s, name in MACRO_CALENDAR_2026:
        try:
            d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= d <= cutoff:
            out.append((date_s, name))
    return out


def upcoming_earnings(tickers: list[str], days_ahead: int = 7) -> list[tuple[str, str]]:
    """Return (date_str, ticker) for each ticker reporting earnings within
    days_ahead. Uses yfinance Ticker.calendar — free, no API key.
    yfinance flakiness is silently absorbed: a ticker with no calendar
    data just gets skipped."""
    import yfinance as yf
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    out = []
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            cal = t.calendar
            if not cal:
                continue
            # cal can be a dict or a DataFrame depending on yfinance version
            er_date = None
            if isinstance(cal, dict):
                er_date = cal.get("Earnings Date")
            else:
                try:
                    er_date = cal.loc["Earnings Date"]
                except Exception:
                    pass
            # er_date is sometimes a list of two timestamps (range)
            if er_date is None:
                continue
            if isinstance(er_date, list) and er_date:
                er_date = er_date[0]
            try:
                if hasattr(er_date, "date"):
                    er = er_date.date()
                else:
                    er = datetime.fromisoformat(str(er_date)[:10]).date()
            except Exception:
                continue
            if today <= er <= cutoff:
                out.append((er.isoformat(), sym))
        except Exception:
            continue
    return out


def fmt_events_block(macro: list[tuple[str, str]], earnings: list[tuple[str, str]]) -> str:
    """One small block to inject into the prompt. Empty string if nothing
    upcoming."""
    if not macro and not earnings:
        return ""
    lines = ["==== UPCOMING EVENTS (use for anticipation framing only — do not invent details) ===="]
    for date_s, name in macro:
        lines.append(f"  {date_s}: {name}")
    if earnings:
        lines.append("  Earnings this week:")
        # group by date
        by_date: dict[str, list[str]] = {}
        for date_s, sym in earnings:
            by_date.setdefault(date_s, []).append(sym)
        for date_s in sorted(by_date):
            syms = ", ".join(by_date[date_s][:8])
            lines.append(f"    {date_s}: {syms}")
    return "\n".join(lines)


def gather(watchlist_tickers: list[str] | None = None) -> str:
    """Returns a ready-to-inject string block, or '' if nothing upcoming."""
    macro = upcoming_macro()
    tickers = list(set((watchlist_tickers or []) + MOVERS_UNIVERSE))
    earnings = upcoming_earnings(tickers)
    return fmt_events_block(macro, earnings)


if __name__ == "__main__":
    print(gather())
