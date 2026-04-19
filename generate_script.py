"""Turn market data + headlines into podcast script via local Ollama."""
from __future__ import annotations
import json
import requests
from datetime import datetime
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, MIN_WORDS, MAX_WORDS


def _fmt_row(r: dict) -> str:
    return f"{r['name']} ({r['symbol']}): {r['close']:.2f} ({r['pct']:+.2f}%)"


def _fmt_section(rows: list[dict]) -> str:
    return "\n".join(_fmt_row(r) for r in rows)


def build_prompt(market: dict, headlines: list[dict], date_str: str) -> str:
    indices = _fmt_section(market["indices"])
    sectors = _fmt_section(market["sectors"])
    macro = _fmt_section(market["macro"])
    gainers = _fmt_section(market["gainers"])
    losers = _fmt_section(market["losers"])
    news = "\n".join(
        f"- [{h['source']}] {h['title']}" + (f" — {h['summary'][:200]}" if h['summary'] else "")
        for h in headlines
    )
    return f"""You are a sharp financial journalist writing a daily US market recap podcast episode.
Date: {date_str}

TODAY'S INDEX CLOSES:
{indices}

SECTOR ETFS:
{sectors}

MACRO:
{macro}

TOP MEGA-CAP GAINERS:
{gainers}

TOP MEGA-CAP LOSERS:
{losers}

NEWS HEADLINES (last 24h):
{news}

Write a single spoken-word podcast script. Requirements:
- Length: however long the news warrants. Quiet day → short ({MIN_WORDS}+ words). Busy day with lots of catalysts → longer (up to {MAX_WORDS} words). Never pad. Never rush past important news.
- Open with the date and a one-sentence headline of the day.
- Cover index moves and WHY — tie specific headlines to the moves. Name the causes.
- Call out which sectors led and lagged and why.
- Mention macro context (rates, dollar, oil) where it explains the tape.
- Cover notable single-stock movers with reasons when the news supports it.
- Conversational tone. No bullet points, no headers, no markdown, no stage directions, no music cues.
- Never invent facts. If the news doesn't explain a move, say the move was unexplained or technical.
- Do not include a title line. Start directly with the spoken content.
- End with a one-line sign-off.
Return ONLY the script text, nothing else."""


def generate(market: dict, headlines: list[dict], date_str: str) -> str:
    prompt = build_prompt(market, headlines, date_str)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_ctx": 8192},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines
    m = fetch_all()
    h = fetch_headlines()
    print(generate(m, h, datetime.now().strftime("%A, %B %d, %Y")))
