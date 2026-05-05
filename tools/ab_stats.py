"""Aggregate .meta.json sidecars by prompt_variant.

Run: python tools/ab_stats.py
Reports per-variant: episode count, mean turns, mean words, mean duration.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "docs" / "episodes"


def main() -> None:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(EPISODES.glob("*.meta.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        v = m.get("prompt_variant", "?")
        by_variant[v].append(m)

    if not by_variant:
        print("no .meta.json sidecars found in docs/episodes/")
        return

    print(f"{'variant':<8} {'n':>4} {'turns':>10} {'words':>12} {'duration':>12} {'chars':>10}")
    print("-" * 60)
    for v, metas in sorted(by_variant.items()):
        if not metas:
            continue
        turns = [m.get("turns", 0) for m in metas]
        words = [m.get("words", 0) for m in metas]
        durs = [m.get("duration_sec", 0) for m in metas]
        chars = [m.get("char_usage_estimate", 0) for m in metas]
        print(
            f"{v:<8} {len(metas):>4d} "
            f"{statistics.mean(turns):>10.1f} "
            f"{statistics.mean(words):>12.1f} "
            f"{statistics.mean(durs):>10.1f}s "
            f"{statistics.mean(chars):>10.0f}"
        )


if __name__ == "__main__":
    main()
