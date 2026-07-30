"""Pull price data via yfinance. No API key.
Retries on transient failures, returns empty list rather than crashing if all fail."""
from __future__ import annotations
import random
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf
from config import INDICES, SECTOR_ETFS, MACRO, MOVERS_UNIVERSE


def _pct(curr: float, prev: float) -> float:
    if prev == 0 or prev is None:
        return 0.0
    return (curr - prev) / prev * 100


def _download_with_retry(syms: list[str], attempts: int = 3) -> any:
    """yfinance occasionally returns empty/JSON-decode errors. Retry with jitter."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            data = yf.download(
                syms, period="5d", interval="1d", progress=False,
                auto_adjust=False, group_by="ticker", threads=True,
            )
            if data is not None and (hasattr(data, "empty") and not data.empty):
                return data
            last_err = RuntimeError("empty yfinance response")
        except Exception as e:
            last_err = e
        # exponential backoff with jitter
        time.sleep(0.5 * (2 ** i) + random.random() * 0.5)
    print(f"[fetch_market] yfinance failed after {attempts} attempts: {last_err}")
    return None


def _snapshot(tickers: dict[str, str]) -> tuple[list[dict], str | None]:
    """Return (rows, last_trade_date_iso). last_trade_date is the most recent
    bar date seen across all symbols; used by callers to detect stale data."""
    syms = list(tickers.keys())
    if not syms:
        return [], None
    data = _download_with_retry(syms)
    if data is None:
        return [], None
    rows: list[dict] = []
    last_trade: date | None = None
    for sym in syms:
        try:
            df = data[sym].dropna() if len(syms) > 1 else data.dropna()
            if len(df) < 2:
                continue
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            try:
                bar_date = df.index[-1].date() if hasattr(df.index[-1], "date") else None
            except Exception:
                bar_date = None
            if bar_date and (last_trade is None or bar_date > last_trade):
                last_trade = bar_date
            rows.append({
                "symbol": sym,
                "name": tickers[sym],
                "close": close,
                "prev_close": prev,
                "pct": _pct(close, prev),
            })
        except Exception:
            continue
    return rows, (last_trade.isoformat() if last_trade else None)


def fetch_movers(n: int = 8) -> tuple[list[dict], list[dict], str | None]:
    """Return (gainers, losers, last_trade_date_iso) from mega-cap universe."""
    universe = {t: t for t in MOVERS_UNIVERSE}
    rows, last_trade = _snapshot(universe)
    if not rows:
        return [], [], last_trade
    # Sign filter: on a broad red (or green) day the top/bottom N would
    # otherwise include wrong-direction names — a decliner announced as a
    # "top gainer" — and the two lists could overlap.
    gainers = sorted((r for r in rows if r["pct"] > 0), key=lambda r: r["pct"], reverse=True)[:n]
    losers = sorted((r for r in rows if r["pct"] < 0), key=lambda r: r["pct"])[:n]
    return gainers, losers, last_trade


def _last_expected_session(now: datetime | None = None) -> date:
    """Most recent NYSE session date, weekday-aware: Sat/Sun roll back to
    Friday, and before the 9:30 ET open we only expect the prior session's
    bar. Holidays aren't modeled (rare one-day false negative, acceptable)."""
    ny = ZoneInfo("America/New_York")
    now_ny = (now or datetime.now(ny)).astimezone(ny)
    d = now_ny.date()
    if d.weekday() < 5 and (now_ny.hour, now_ny.minute) < (9, 30):
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5 / Sun=6 → Friday
        d -= timedelta(days=1)
    return d


def fetch_all() -> dict:
    gainers, losers, last_trade_movers = fetch_movers()
    indices, last_trade_indices = _snapshot(INDICES)
    sectors, _ = _snapshot(SECTOR_ETFS)
    macro, _ = _snapshot(MACRO)
    # Use the most recent trade date we saw across any data slice.
    candidates = [d for d in (last_trade_movers, last_trade_indices) if d]
    as_of = max(candidates) if candidates else None
    # Stale only if we're missing the last *expected* session — comparing to
    # the local calendar date flagged every weekend and pre-market run.
    is_stale = bool(as_of and as_of < _last_expected_session().isoformat())
    return {
        "indices": indices,
        "sectors": sectors,
        "macro": macro,
        "gainers": gainers,
        "losers": losers,
        "as_of": as_of,
        "is_stale": is_stale,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_all(), indent=2, default=str))
