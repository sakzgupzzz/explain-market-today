"""Webhook notifications via ntfy.sh.

Free, no auth required. POST to https://ntfy.sh/<topic>. User subscribes on
their phone via the ntfy app to that topic name.

Set NTFY_TOPIC env var to enable. No-op if unset.
"""
from __future__ import annotations
import os
import urllib.request
import urllib.error

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh")

# Dedup guard. main.py can call notify_failure inside check_budget AND from
# the outer except — without this, the same incident pings twice. Cleared on
# any notify_success since a successful run resets the failure window.
_NOTIFIED_FAILURES: set[tuple[str, str]] = set()


def notify(title: str, body: str, priority: str = "default", tags: list[str] | None = None) -> None:
    """Fire-and-forget POST to ntfy. Silently no-op if NTFY_TOPIC is unset
    or the request fails (notifications shouldn't crash the pipeline)."""
    if not NTFY_TOPIC:
        return
    try:
        url = f"{NTFY_URL}/{NTFY_TOPIC}"
        headers = {
            "Title": title,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = ",".join(tags)
        req = urllib.request.Request(
            url,
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except (urllib.error.URLError, OSError) as e:
        print(f"[notify] failed (non-fatal): {e}")


def notify_success(date_str: str, mode: str, turns: int, words: int, dur_sec: float) -> None:
    _NOTIFIED_FAILURES.discard((date_str, "any"))
    notify(
        title=f"✓ episode {date_str} published",
        body=f"mode={mode} · {turns} turns · {words} words · {int(dur_sec//60)}:{int(dur_sec%60):02d}",
        tags=["white_check_mark", "headphones"],
    )


def notify_failure(date_str: str, where: str, err: str) -> None:
    key = (date_str, where)
    if key in _NOTIFIED_FAILURES:
        return
    _NOTIFIED_FAILURES.add(key)
    # Also dedup any second notify on the same day no matter what stage —
    # check_budget → main re-raise → outer except is the common double-ping.
    if (date_str, "any") in _NOTIFIED_FAILURES:
        return
    _NOTIFIED_FAILURES.add((date_str, "any"))
    notify(
        title=f"✗ episode {date_str} failed",
        body=f"stage: {where}\n{err[:400]}",
        priority="high",
        tags=["x", "warning"],
    )


def notify_warn(date_str: str, where: str, msg: str) -> None:
    """Non-fatal warning ping. Same dedup as failure but lower priority."""
    key = (date_str, f"warn:{where}")
    if key in _NOTIFIED_FAILURES:
        return
    _NOTIFIED_FAILURES.add(key)
    notify(
        title=f"⚠ episode {date_str} warning",
        body=f"stage: {where}\n{msg[:400]}",
        priority="default",
        tags=["warning"],
    )
