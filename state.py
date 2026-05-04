"""Cross-episode memory. Tracks which story clusters have been covered in
recent days so today's render can suppress repeats and offer follow-up framing
on continuing stories.

Persisted as .state.json at the repo root (gitignored)."""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from config import STATE_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    p = Path(STATE_PATH)
    if not p.exists():
        return {"covered": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"covered": []}


def save_state(state: dict) -> None:
    Path(STATE_PATH).write_text(json.dumps(state, indent=2, sort_keys=True))


def covered_within(state: dict, days: int = 3) -> set[str]:
    """Cluster IDs covered within the last `days` days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return {
        c["cluster_id"]
        for c in state.get("covered", [])
        if c.get("first_covered", "") >= cutoff
    }


def mark_covered(state: dict, cluster_ids: list[str]) -> dict:
    """Add today's covered clusters and prune anything older than 14 days."""
    seen = {c["cluster_id"] for c in state.get("covered", [])}
    state.setdefault("covered", [])
    now = _now_iso()
    for cid in cluster_ids:
        if cid not in seen:
            state["covered"].append({"cluster_id": cid, "first_covered": now})
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    state["covered"] = [
        c for c in state["covered"] if c.get("first_covered", "") >= cutoff
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
