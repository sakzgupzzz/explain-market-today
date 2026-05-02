"""Pull price data via yfinance. No API key.
Retries on transient failures, returns empty list rather than crashing if all fail."""
from __future__ import annotations
import random
import time
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


def _snapshot(tickers: dict[str, str]) -> list[dict]:
    syms = list(tickers.keys())
    if not syms:
        return []
    data = _download_with_retry(syms)
    if data is None:
        return []
    rows: list[dict] = []
    for sym in syms:
        try:
            df = data[sym].dropna() if len(syms) > 1 else data.dropna()
            if len(df) < 2:
                continue
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            rows.append({
                "symbol": sym,
                "name": tickers[sym],
                "close": close,
                "prev_close": prev,
                "pct": _pct(close, prev),
            })
        except Exception:
            continue
    return rows


def fetch_movers(n: int = 8) -> tuple[list[dict], list[dict]]:
    """Return (gainers, losers) from mega-cap universe."""
    universe = {t: t for t in MOVERS_UNIVERSE}
    rows = _snapshot(universe)
    if not rows:
        return [], []
    rows.sort(key=lambda r: r["pct"])
    losers = rows[:n]
    gainers = list(reversed(rows[-n:]))
    return gainers, losers


def fetch_all() -> dict:
    gainers, losers = fetch_movers()
    return {
        "indices": _snapshot(INDICES),
        "sectors": _snapshot(SECTOR_ETFS),
        "macro": _snapshot(MACRO),
        "gainers": gainers,
        "losers": losers,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_all(), indent=2, default=str))
