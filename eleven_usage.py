"""ElevenLabs char-usage guard.

Calls /v1/user/subscription to check current month usage. If the API key
lacks `voices_read` / `user_read` scope (returns 401), this is a no-op so
restricted keys don't break the pipeline.

Used by main.py before each run to abort cleanly when usage is over a
configurable threshold (default 95%) — prevents runaway overage charges.
"""
from __future__ import annotations
import os
import urllib.request
import json
from config import ELEVENLABS_API_KEY


def fetch_subscription() -> dict | None:
    """Return {tier, character_count, character_limit, next_reset_unix} or
    None if the call fails (key restricted, network error, etc.)."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        return {
            "tier": data.get("tier", "?"),
            "character_count": data.get("character_count", 0),
            "character_limit": data.get("character_limit", 0),
            "next_reset_unix": data.get("next_character_count_reset_unix", 0),
        }
    except Exception as e:
        print(f"[eleven] subscription check unavailable: {e}")
        return None


def usage_pct(sub: dict | None) -> float | None:
    if not sub:
        return None
    limit = sub.get("character_limit") or 0
    if limit <= 0:
        return None
    return sub.get("character_count", 0) / limit


def check_budget(threshold: float = 0.95) -> tuple[bool, str]:
    """(ok_to_proceed, message). Returns (True, '') if usage is unknown
    (restricted key) — fail-open so the pipeline still runs."""
    sub = fetch_subscription()
    if sub is None:
        return True, "[eleven] usage unknown (key may be TTS-only); proceeding"
    pct = usage_pct(sub)
    if pct is None:
        return True, "[eleven] limit=0 (unlimited tier?); proceeding"
    msg = (
        f"[eleven] tier={sub['tier']} "
        f"used={sub['character_count']:,}/{sub['character_limit']:,} "
        f"({pct*100:.1f}%)"
    )
    if pct >= threshold:
        return False, f"{msg} — over {threshold*100:.0f}% threshold; aborting"
    return True, msg


if __name__ == "__main__":
    ok, msg = check_budget()
    print(msg)
    print(f"proceed: {ok}")
