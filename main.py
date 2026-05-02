"""Orchestrate: fetch → script → TTS → feed → push."""
from __future__ import annotations
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Load .env before any other module reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import EPISODES_DIR
from fetch_market import fetch_all
from fetch_news import fetch_headlines
from generate_script import generate, critique_revise
from sanitize import sanitize_script
from tts import synth
from publish import build_feed, build_index_html, git_push, write_transcripts, write_chapters


def run(push: bool = True) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    date_pretty = datetime.now().strftime("%A, %B %d, %Y")
    print(f"[{today}] fetching market data…")
    market = fetch_all()
    print(f"[{today}] fetching headlines…")
    headlines_by_cat = fetch_headlines()
    total = sum(len(v) for v in headlines_by_cat.values())
    print(f"[{today}] {total} headlines across {len(headlines_by_cat)} beats")
    print(f"[{today}] generating script with local LLM…")
    script = generate(market, headlines_by_cat, date_pretty)

    print(f"[{today}] critique pass…")
    script = critique_revise(script, market, headlines_by_cat)

    print(f"[{today}] sanitizing…")
    script = sanitize_script(script)

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = EPISODES_DIR / f"{today}.txt"
    mp3_path = EPISODES_DIR / f"{today}.mp3"
    txt_path.write_text(script)
    print(f"[{today}] synthesizing audio…")
    synth_result = synth(script, mp3_path)
    # synth() may return (mp3_path, chunk_timings) for v3 dialogue path
    chunk_timings = synth_result[1] if isinstance(synth_result, tuple) else None

    print(f"[{today}] writing transcripts + chapters…")
    write_transcripts(script, mp3_path, chunk_timings)
    write_chapters(script, mp3_path, chunk_timings)

    print(f"[{today}] building feed…")
    build_feed()
    build_index_html()

    if push:
        print(f"[{today}] pushing…")
        git_push(f"episode {today}")
    print(f"[{today}] done → {mp3_path}")
    return mp3_path


if __name__ == "__main__":
    push = "--no-push" not in sys.argv
    try:
        run(push=push)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
