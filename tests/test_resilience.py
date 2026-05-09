"""Resilience-primitive tests: lock, notify, eleven_budget, verify_facts.

Covers what's load-bearing for a silent CI failure:
  - lock contention + stale recovery
  - notify dedup window
  - eleven_budget month boundary + override path
  - verify_facts soft-flag path
"""
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── lock.py ────────────────────────────────────────────────────────────────


def test_lock_acquire_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr("config.ROOT", tmp_path)
    import importlib
    import lock
    importlib.reload(lock)
    assert not lock.LOCK_PATH.exists()
    with lock.acquire_lock():
        assert lock.LOCK_PATH.exists()
    assert not lock.LOCK_PATH.exists()


def test_lock_stale_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr("config.ROOT", tmp_path)
    import importlib
    import lock
    importlib.reload(lock)
    # Write a stale lock file from a long-dead pid + ancient timestamp
    lock.LOCK_PATH.write_text(f"99999999:{time.time() - 10 * lock.STALE_LOCK_SEC}")
    with lock.acquire_lock():
        # Stale lock should have been taken over
        text = lock.LOCK_PATH.read_text()
        assert text.startswith(f"{os.getpid()}:")


def test_lock_pid_alive_check_for_self():
    import lock
    assert lock._is_pid_alive(os.getpid()) is True
    assert lock._is_pid_alive(0) is False


def test_lock_read_malformed(tmp_path):
    import lock
    bad = tmp_path / ".run.lock.bad"
    bad.write_text("not-a-pid")
    assert lock._read_lock(bad) is None
    bad.write_text("")
    assert lock._read_lock(bad) is None


# ─── notify.py ──────────────────────────────────────────────────────────────


def test_notify_noop_when_topic_unset(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    import importlib
    import notify
    importlib.reload(notify)
    # Should not raise, should not POST
    with patch("urllib.request.urlopen") as m:
        notify.notify("title", "body")
        assert m.call_count == 0


def test_notify_failure_dedups(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    import importlib
    import notify
    importlib.reload(notify)
    with patch("urllib.request.urlopen") as m:
        notify.notify_failure("2026-05-09", "stage-a", "boom")
        notify.notify_failure("2026-05-09", "stage-b", "boom")  # same date, different stage
        notify.notify_failure("2026-05-09", "stage-a", "boom")  # exact dup
        # First fires; second is blocked by the (date, "any") sentinel; dup also blocked.
        assert m.call_count == 1


def test_notify_warn_separate_dedup(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    import importlib
    import notify
    importlib.reload(notify)
    with patch("urllib.request.urlopen") as m:
        notify.notify_warn("2026-05-09", "verify_facts", "msg")
        notify.notify_warn("2026-05-09", "verify_facts", "msg")  # dup
        notify.notify_warn("2026-05-09", "eleven_budget", "msg")  # different stage
        # warn dedup keys on (date, "warn:stage") so two distinct stages fire.
        assert m.call_count == 2


# ─── eleven_budget.py ───────────────────────────────────────────────────────


def test_eleven_budget_manual_override(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_REMAINING_CHARS", "50000")
    monkeypatch.setenv("ELEVENLABS_CHAR_BUDGET_MONTHLY", "100000")
    import importlib
    import config
    import eleven_budget
    importlib.reload(config)
    importlib.reload(eleven_budget)
    p = eleven_budget.compute_dynamic_preset()
    assert p is not None
    assert p["_source"] == "manual_env"
    assert p["_remaining_chars"] == 50000
    assert p["_used_chars"] == 50000
    assert p["min_words"] > 0 and p["max_words"] > p["min_words"]


def test_eleven_budget_falls_back_to_meta(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_REMAINING_CHARS", raising=False)
    import importlib
    import eleven_budget
    importlib.reload(eleven_budget)
    # Force fetch_subscription to return None (no API access)
    with patch.object(eleven_budget, "fetch_subscription", return_value=None):
        p = eleven_budget.compute_dynamic_preset()
    assert p is not None
    assert p["_source"] == "meta_sidecars"


def test_eleven_budget_weekdays_until():
    import eleven_budget
    # Mon → Fri = 5 weekdays inclusive of start, exclusive of end
    mon = datetime(2026, 5, 4, tzinfo=timezone.utc)  # Monday
    sat = datetime(2026, 5, 9, tzinfo=timezone.utc)  # Saturday
    assert eleven_budget._weekdays_until(sat, now=mon) == 5
    # Sat → Mon = 0 weekdays in window (excluding Sat+Sun)
    sun = datetime(2026, 5, 10, tzinfo=timezone.utc)
    mon2 = datetime(2026, 5, 11, tzinfo=timezone.utc)
    assert eleven_budget._weekdays_until(mon2, now=sun) >= 1


def test_eleven_budget_next_month_start():
    import eleven_budget
    mid = datetime(2026, 5, 15, tzinfo=timezone.utc)
    nxt = eleven_budget._next_month_start(mid)
    assert nxt.month == 6 and nxt.day == 1
    dec = datetime(2026, 12, 15, tzinfo=timezone.utc)
    nxt = eleven_budget._next_month_start(dec)
    assert nxt.year == 2027 and nxt.month == 1


# ─── verify_facts.py ────────────────────────────────────────────────────────


def test_verify_facts_flags_unscheduled_macro(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    import importlib
    import notify
    import verify_facts
    importlib.reload(notify)
    importlib.reload(verify_facts)
    civic = {"macro_today": []}  # nothing scheduled
    script = "JAMIE: Today's CPI just released and came in hot at three percent."
    with patch("urllib.request.urlopen") as m:
        verify_facts._flag_unscheduled_macro_claims(script, civic)
        assert m.call_count == 1


def test_verify_facts_no_flag_when_macro_scheduled(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    import importlib
    import notify
    import verify_facts
    importlib.reload(notify)
    importlib.reload(verify_facts)
    civic = {"macro_today": [{"name": "Consumer Price Index"}]}
    script = "JAMIE: Today's CPI just released and came in hot."
    with patch("urllib.request.urlopen") as m:
        verify_facts._flag_unscheduled_macro_claims(script, civic)
        assert m.call_count == 0


def test_verify_facts_no_flag_when_no_proximity(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    import importlib
    import notify
    import verify_facts
    importlib.reload(notify)
    importlib.reload(verify_facts)
    civic = {"macro_today": []}
    script = "JAMIE: CPI prints in a few weeks."  # no 'today'
    with patch("urllib.request.urlopen") as m:
        verify_facts._flag_unscheduled_macro_claims(script, civic)
        assert m.call_count == 0
