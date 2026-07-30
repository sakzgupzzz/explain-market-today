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
    except Exception as e:
        # Non-fatal, but say so — a silent {} means default tone/watchlist.
        print(f"[interests] failed to load {path.name} ({type(e).__name__}: {e}); using defaults")
        return {}


def watchlist_tickers(interests: dict) -> set[str]:
    return {t.upper() for t in (interests.get("watchlist") or {}).get("tickers") or []}
