"""Orchestrate: fetch → script → TTS → feed → push."""
from __future__ import annotations
import sys
import traceback
from datetime import datetime
from pathlib import Path
from config import EPISODES_DIR
from fetch_market import fetch_all
from fetch_news import fetch_headlines
from generate_script import generate
from tts import synth
from publish import build_feed, build_index_html, git_push


def run(push: bool = True) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    date_pretty = datetime.now().strftime("%A, %B %d, %Y")
    print(f"[{today}] fetching market data…")
    market = fetch_all()
    print(f"[{today}] fetching headlines…")
    headlines = fetch_headlines()
    print(f"[{today}] generating script with local LLM…")
    script = generate(market, headlines, date_pretty)

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = EPISODES_DIR / f"{today}.txt"
    mp3_path = EPISODES_DIR / f"{today}.mp3"
    txt_path.write_text(script)
    print(f"[{today}] synthesizing audio…")
    synth(script, mp3_path)

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
