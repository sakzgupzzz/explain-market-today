"""Local prompt iteration tool — fast feedback loop without burning quota.

Workflow:
    # one-time per day: cache the day's data
    python tools/dev_loop.py --refresh

    # iterate: edit generate_script.py prompt, then
    python tools/dev_loop.py
        # → uses cached market + ranked stories, hits Groq (or local Ollama
        #   if GROQ_API_KEY unset), runs critique + verify + sanitize,
        #   prints final script + lint stats.

    # to run against local Ollama qwen3:14b instead of Groq, unset the env:
    GROQ_API_KEY= python tools/dev_loop.py
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".dev_loop_cache.json"


def refresh_cache() -> None:
    """Pull live data and persist to a cache file. Run once per day."""
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines, flatten
    from cluster import cluster_headlines
    from score import score_clusters
    from interests_loader import load_interests
    from calendar_events import gather as gather_calendar_events

    print("[dev] fetching market…")
    market = fetch_all()
    print("[dev] fetching headlines…")
    h = fetch_headlines()
    flat = flatten(h)
    print(f"[dev] {len(flat)} headlines → clustering…")
    clusters = cluster_headlines(flat)
    interests = load_interests()
    ranked = score_clusters(clusters, market, interests)
    watchlist = (interests.get("watchlist") or {}).get("tickers") or []
    upcoming = gather_calendar_events(watchlist)
    payload = {
        "market": market,
        "ranked": ranked[:30],  # cap so cache stays small
        "interests": interests,
        "upcoming_events": upcoming,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }
    CACHE.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[dev] cached → {CACHE} ({len(ranked)} ranked stories)")


def run_once() -> None:
    if not CACHE.exists():
        print("no cache. run with --refresh first.")
        sys.exit(1)
    payload = json.loads(CACHE.read_text())
    from generate_script import generate, critique_revise
    from verify_facts import verify
    from sanitize import sanitize_script

    market = payload["market"]
    ranked = payload["ranked"]
    interests = payload["interests"]
    upcoming = payload.get("upcoming_events", "")
    date_pretty = datetime.now().strftime("%A, %B %d, %Y")

    print("[dev] generate…")
    script = generate(market, ranked, date_pretty, upcoming_events=upcoming, interests=interests)
    print(f"[dev] critique…")
    script = critique_revise(script, market, ranked)
    print(f"[dev] verify…")
    script = verify(script, market, ranked)
    print(f"[dev] sanitize…")
    script = sanitize_script(script)

    out = ROOT / ".dev_loop_output.txt"
    out.write_text(script)
    print(f"\n[dev] script → {out}")
    # print summary
    import re
    turns = [l for l in script.splitlines() if re.match(r"^[A-Z][A-Z0-9_]{0,15}:\s*\S", l)]
    cleaned = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:|\[[^\]]+\]", "", script, flags=re.M)
    print(f"\n=== {len(turns)} turns · {len(cleaned.split())} words ===")
    print()
    print(script[:2000] + ("\n... (truncated, full at .dev_loop_output.txt)" if len(script) > 2000 else ""))


def main() -> None:
    if "--refresh" in sys.argv:
        refresh_cache()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
    else:
        run_once()


if __name__ == "__main__":
    main()
