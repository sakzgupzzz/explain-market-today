"""File-based concurrency lock for main.run().

Belt-and-suspenders alongside the mp3-exists guard. Prevents two simultaneous
local invocations (manual + launchd) from racing on the same minute. Uses a
PID-stamped lock file with a stale-lock recovery window — if the holder
process is gone or the lock is older than STALE_LOCK_SEC, take it.
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from pathlib import Path
from config import ROOT

LOCK_PATH = ROOT / ".run.lock"
STALE_LOCK_SEC = 30 * 60  # 30 minutes — longer than any realistic run


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _read_lock(p: Path) -> tuple[int, float] | None:
    try:
        text = p.read_text().strip()
        pid_s, ts_s = text.split(":", 1)
        return int(pid_s), float(ts_s)
    except (OSError, ValueError):
        return None


@contextmanager
def acquire_lock():
    """Context manager. Raises RuntimeError if lock is held by a live process
    that started < STALE_LOCK_SEC ago. Otherwise takes the lock and releases
    on exit."""
    if LOCK_PATH.exists():
        existing = _read_lock(LOCK_PATH)
        if existing is not None:
            pid, ts = existing
            age = time.time() - ts
            if age < STALE_LOCK_SEC and _is_pid_alive(pid):
                raise RuntimeError(
                    f"another run is in progress: pid={pid}, age={age:.0f}s "
                    f"(lock at {LOCK_PATH})"
                )
            print(f"[lock] taking stale lock: pid={pid}, age={age:.0f}s")
    LOCK_PATH.write_text(f"{os.getpid()}:{time.time()}")
    try:
        yield
    finally:
        try:
            LOCK_PATH.unlink()
        except OSError:
            pass
