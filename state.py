"""Cross-episode memory. Tracks which story clusters have been covered in
recent days so today's render can suppress repeats and offer follow-up framing
on continuing stories.

Persisted as .state.json at the repo root (gitignored)."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from config import STATE_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    """Load state with schema validation. On corruption, rotate the bad file
    to .state.json.broken and return a fresh state. Caller can detect via
    the 'recovered_from_corruption' flag."""
    p = Path(STATE_PATH)
    if not p.exists():
        return {"covered": []}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        broken = p.with_suffix(".json.broken")
        try:
            p.rename(broken)
            print(f"[state] {STATE_PATH} corrupt ({e}); rotated to {broken}")
        except OSError:
            print(f"[state] {STATE_PATH} corrupt ({e}); could not rotate")
        return {"covered": [], "recovered_from_corruption": True}
    if not isinstance(data, dict) or not isinstance(data.get("covered", []), list):
        broken = p.with_suffix(".json.broken")
        try:
            p.rename(broken)
        except OSError:
            pass
        print(f"[state] schema invalid; rotated to {broken}")
        return {"covered": [], "recovered_from_corruption": True}
    # filter out any malformed entries
    data["covered"] = [
        c for c in data.get("covered", [])
        if isinstance(c, dict) and isinstance(c.get("cluster_id"), str)
        and isinstance(c.get("first_covered"), str)
    ]
    return data


def save_state(state: dict) -> None:
    """Atomic write: tmp file + os.replace so a crash mid-write can't leave a
    truncated .state.json behind (load_state would rotate it and forget all
    coverage memory)."""
    p = Path(STATE_PATH)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _last_seen(c: dict) -> str:
    """Most recent coverage timestamp; falls back to first_covered for state
    files written before last_covered existed."""
    return c.get("last_covered") or c.get("first_covered", "")


def covered_within(state: dict, days: int = 3) -> set[str]:
    """Cluster IDs covered within the last `days` days. Keyed on last_covered
    so a continuing story that gets re-covered stays suppressed — keying on
    first_covered let a day-1 story escape the 2-day window on day 3 even
    though it ran again on day 2."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return {
        c["cluster_id"]
        for c in state.get("covered", [])
        if _last_seen(c) >= cutoff
    }


def mark_covered(state: dict, cluster_ids: list[str]) -> dict:
    """Add today's covered clusters and prune anything older than 14 days.
    EVERY hit — including an already-seen ID — refreshes last_covered, so the
    suppression window tracks the latest mention, not the first."""
    by_id = {c["cluster_id"]: c for c in state.get("covered", [])}
    state.setdefault("covered", [])
    now = _now_iso()
    for cid in cluster_ids:
        if cid in by_id:
            by_id[cid]["last_covered"] = now
        else:
            entry = {"cluster_id": cid, "first_covered": now, "last_covered": now}
            state["covered"].append(entry)
            by_id[cid] = entry
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    state["covered"] = [
        c for c in state["covered"] if _last_seen(c) >= cutoff
    ]
    return state


def annotate_clusters(clusters: list[dict], state: dict, suppress_days: int = 2) -> list[dict]:
    """Annotate each cluster with a `seen_recently` flag.
    Caller decides whether to drop or use them for follow-up framing."""
    seen = covered_within(state, suppress_days)
    out = []
    for c in clusters:
        c2 = dict(c)
        c2["seen_recently"] = c["id"] in seen
        out.append(c2)
    return out
