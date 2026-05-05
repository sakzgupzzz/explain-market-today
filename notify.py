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
    notify(
        title=f"✓ episode {date_str} published",
        body=f"mode={mode} · {turns} turns · {words} words · {int(dur_sec//60)}:{int(dur_sec%60):02d}",
        tags=["white_check_mark", "headphones"],
    )


def notify_failure(date_str: str, where: str, err: str) -> None:
    notify(
        title=f"✗ episode {date_str} failed",
        body=f"stage: {where}\n{err[:400]}",
        priority="high",
        tags=["x", "warning"],
    )
