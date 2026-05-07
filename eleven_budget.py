"""Dynamic episode-length sizing based on ElevenLabs char budget remaining.

Goal: stretch monthly Eleven char budget evenly across remaining weekday
runs. If we have lots of headroom, episodes can run longer. If we're
running low, episodes get tighter automatically.

Two paths to compute used-chars-this-month:
  1. ElevenLabs /v1/user/subscription — accurate, requires user_read scope
  2. Aggregate .meta.json sidecars from docs/episodes/ + docs/express/
     — works with TTS-only keys, slight undercount (excludes audio-tag
     overhead and any non-script chars)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import EPISODES_DIR, ELEVENLABS_CHAR_BUDGET_MONTHLY
from eleven_usage import fetch_subscription


# Heuristic: dialogue text → spoken chars including audio tags + light prosody
# is roughly 1.05x the script char count. ElevenLabs charges per char of
# input text, so we bill against the script length.
CHAR_OVERHEAD_FACTOR = 1.05

# Each word in dialogue averages ~6 chars including spaces and audio tags.
WORDS_PER_CHAR = 1 / 6

# Safety margin so we don't run out mid-month.
SAFETY_MARGIN = 0.85

# Hard floors and ceilings on what the dynamic preset can produce.
MIN_TARGET_WORDS = 350     # ~2 min audio
MAX_TARGET_WORDS = 2400    # ~14 min audio
MIN_TURNS_FLOOR = 14
MAX_TURNS_CEIL = 36


def _used_chars_this_month_from_meta() -> int:
    """Sum char_usage_estimate across all .meta.json sidecars whose
    generated_at falls in the current calendar month."""
    now = datetime.now(timezone.utc)
    month_prefix = now.strftime("%Y-%m")
    total = 0
    for d in (EPISODES_DIR, EPISODES_DIR.parent / "express"):
        if not d.exists():
            continue
        for p in d.glob("*.meta.json"):
            try:
                m = json.loads(p.read_text())
            except Exception:
                continue
            gen = (m.get("generated_at") or "")
            if gen.startswith(month_prefix):
                total += int(m.get("char_usage_estimate", 0) or 0)
    # apply the overhead factor — meta tracks raw script length, but
    # ElevenLabs bills the synthesized char count which includes audio tags
    # and is slightly higher.
    return int(total * CHAR_OVERHEAD_FACTOR)


def _weekdays_until(end_dt: datetime, now: datetime | None = None) -> int:
    """Count weekday-d days from now (inclusive) up to end_dt (exclusive)."""
    now = now or datetime.now(timezone.utc)
    if end_dt <= now:
        return 0
    days = (end_dt.date() - now.date()).days
    count = 0
    cur = now.date()
    for i in range(days):
        if cur.weekday() < 5:  # Mon=0 .. Fri=4
            count += 1
        cur += timedelta(days=1)
    return max(1, count)


def _next_month_start(now: datetime) -> datetime:
    if now.month == 12:
        return datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)


def compute_dynamic_preset() -> dict | None:
    """Returns {min_words, max_words, min_turns, ...telemetry} or None
    if budget data is unavailable. Caller falls back to interests.yaml
    preferences when None."""
    now = datetime.now(timezone.utc)
    sub = fetch_subscription()

    if sub and sub.get("character_limit"):
        limit = int(sub["character_limit"])
        used = int(sub.get("character_count", 0))
        reset_unix = sub.get("next_reset_unix", 0)
        if reset_unix:
            reset_at = datetime.fromtimestamp(reset_unix, tz=timezone.utc)
        else:
            reset_at = _next_month_start(now)
        source = "elevenlabs_api"
    else:
        # Fall back to local meta-sidecar tally
        limit = int(ELEVENLABS_CHAR_BUDGET_MONTHLY or 130_000)
        used = _used_chars_this_month_from_meta()
        reset_at = _next_month_start(now)
        source = "meta_sidecars"

    remaining = max(0, limit - used)
    weekdays_left = _weekdays_until(reset_at, now)
    if weekdays_left <= 0:
        weekdays_left = 1

    # Each weekday produces a SHOW (this preset) + an EXPRESS (~25% the
    # size of show). Reserve ~20% of remaining for express + variance.
    show_share = 0.65
    per_show_chars = (remaining / weekdays_left) * show_share * SAFETY_MARGIN

    target_words = int(per_show_chars * WORDS_PER_CHAR)
    target_words = max(MIN_TARGET_WORDS, min(MAX_TARGET_WORDS, target_words))

    # Span min..max around the target with ±15% width.
    min_words = int(target_words * 0.85)
    max_words = int(target_words * 1.20)

    # Turn count: target ~28 words/turn average for a comfortable floor.
    # The prompt's HARD MAX is 35 turns so we want headroom below that.
    min_turns = max(MIN_TURNS_FLOOR, min(MAX_TURNS_CEIL, target_words // 28))

    return {
        "min_words": min_words,
        "max_words": max_words,
        "min_turns": min_turns,
        "_source": source,
        "_remaining_chars": remaining,
        "_used_chars": used,
        "_limit": limit,
        "_weekdays_left": weekdays_left,
        "_per_show_chars_budget": int(per_show_chars),
        "_target_words": target_words,
    }


def format_log_line(preset: dict) -> str:
    return (
        f"[budget] {preset['_source']}: used {preset['_used_chars']:,}/"
        f"{preset['_limit']:,} chars ({preset['_remaining_chars']:,} left) · "
        f"{preset['_weekdays_left']} weekdays remaining → "
        f"{preset['_per_show_chars_budget']:,} chars/show "
        f"→ target {preset['_target_words']} words "
        f"({preset['min_words']}-{preset['max_words']}) "
        f"across ≥{preset['min_turns']} turns"
    )


if __name__ == "__main__":
    p = compute_dynamic_preset()
    if p:
        print(format_log_line(p))
        print(json.dumps(p, indent=2))
    else:
        print("(no budget data available)")
