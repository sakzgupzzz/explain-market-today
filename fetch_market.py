"""Pull price data via yfinance. No API key."""
from __future__ import annotations
import yfinance as yf
from config import INDICES, SECTOR_ETFS, MACRO, MOVERS_UNIVERSE


def _pct(curr: float, prev: float) -> float:
    if prev == 0 or prev is None:
        return 0.0
    return (curr - prev) / prev * 100


def _snapshot(tickers: dict[str, str]) -> list[dict]:
    syms = list(tickers.keys())
    data = yf.download(
        syms, period="5d", interval="1d", progress=False,
        auto_adjust=False, group_by="ticker", threads=True,
    )
    rows = []
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
    rows.sort(key=lambda r: r["pct"])
    losers = rows[:n]
    gainers = list(reversed(rows[-n:]))
    return gainers, losers


def fetch_all() -> dict:
    return {
        "indices": _snapshot(INDICES),
        "sectors": _snapshot(SECTOR_ETFS),
        "macro": _snapshot(MACRO),
        "gainers": fetch_movers()[0],
        "losers": fetch_movers()[1],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_all(), indent=2, default=str))
