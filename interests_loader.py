"""Load interests.yaml. Returns {} on any error — safe to default."""
from __future__ import annotations
from pathlib import Path
from config import ROOT


def load_interests() -> dict:
    path = ROOT / "interests.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def watchlist_tickers(interests: dict) -> set[str]:
    return {t.upper() for t in (interests.get("watchlist") or {}).get("tickers") or []}
