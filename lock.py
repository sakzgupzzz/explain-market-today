"""File-based concurrency lock for main.run().

Belt-and-suspenders alongside the mp3-exists guard. Prevents two simultaneous
local invocations (manual + launchd) from racing on the same minute. Uses a
PID-stamped lock file acquired atomically (O_CREAT|O_EXCL). A lock held by a
LIVE process is never stolen, regardless of age — long Ollama runs
(OLLAMA_TIMEOUT × several calls) legitimately exceed any fixed window. Only a
dead holder's lock is recovered.
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from pathlib import Path
from config import ROOT

LOCK_PATH = ROOT / ".run.lock"
# Informational only (logged with the holder's age). Liveness of the holder
# PID — not age — decides whether a lock is stealable; a live run that takes
# 45 minutes still owns its lock.
STALE_LOCK_SEC = 30 * 60


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
    """Context manager. Acquires the lock atomically via O_CREAT|O_EXCL (no
    check-then-write race). Raises RuntimeError while the holder process is
    ALIVE — regardless of lock age. Steals only a dead holder's lock, then
    retries the atomic open (another process may win that race). Releases on
    exit."""
    fd = -1
    for _ in range(5):  # bounded: dead-holder steal + re-race, not a spin loop
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            existing = _read_lock(LOCK_PATH)
            if existing is not None:
                pid, ts = existing
                age = time.time() - ts
                if _is_pid_alive(pid):
                    raise RuntimeError(
                        f"another run is in progress: pid={pid}, age={age:.0f}s "
                        f"(lock at {LOCK_PATH})"
                    )
                print(f"[lock] taking dead-holder lock: pid={pid}, age={age:.0f}s")
            # Holder is dead (or lock unreadable): remove and retry O_EXCL.
            try:
                LOCK_PATH.unlink()
            except OSError:
                pass
    if fd < 0:
        raise RuntimeError(f"could not acquire lock at {LOCK_PATH}")
    try:
        os.write(fd, f"{os.getpid()}:{time.time()}".encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        try:
            LOCK_PATH.unlink()
        except OSError:
            pass
