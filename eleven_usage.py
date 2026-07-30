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


def remaining_chars() -> int | None:
    """Actual chars left this month, or None if unknown. The live
    /v1/user/subscription query is authoritative; the manually-set
    ELEVENLABS_REMAINING_CHARS env var is only a FALLBACK for when the
    API query fails (TTS-only key scope) — a stale env value must never
    shadow live data."""
    sub = fetch_subscription()
    if sub and (sub.get("character_limit") or 0) > 0:
        return max(0, sub["character_limit"] - sub.get("character_count", 0))
    manual = os.environ.get("ELEVENLABS_REMAINING_CHARS", "").strip()
    if manual:
        try:
            return max(0, int(manual.replace(",", "")))
        except ValueError:
            pass
    return None


# Dialogue text → billed chars is slightly above script length (audio tags,
# normalization). Single source of truth lives in eleven_budget; imported
# lazily inside check_prespend to avoid a module-level import cycle
# (eleven_budget imports fetch_subscription from this module).

def check_prespend(script_text: str) -> tuple[bool, str]:
    """(ok_to_proceed, message) for THIS synthesis. Unlike check_budget —
    which only looks at PAST usage and passes at 94.9% before billing a
    full show — this estimates the upcoming spend and blocks it if it
    would exceed the actual remaining budget. Called from tts.synth()
    before any ElevenLabs API call. Fail-open only when neither the API
    nor the env fallback can tell us what's remaining."""
    from eleven_budget import CHAR_OVERHEAD_FACTOR  # lazy — see note above
    est = int(len(script_text) * CHAR_OVERHEAD_FACTOR)
    remaining = remaining_chars()
    if remaining is None:
        return True, f"[eleven] pre-spend: remaining unknown (restricted key, no env fallback); proceeding with est {est:,} chars"
    if est > remaining:
        return False, (
            f"[eleven] pre-spend: est {est:,} chars > {remaining:,} remaining "
            f"— aborting before any billing"
        )
    return True, f"[eleven] pre-spend: est {est:,} chars ≤ {remaining:,} remaining"


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
