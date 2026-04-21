"""Morning-Brew-style daily brief as a 2-host dialogue via local Ollama."""
from __future__ import annotations
import requests
from datetime import datetime
from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, MIN_WORDS, MAX_WORDS, CHARACTERS,
)


def _fmt_row(r: dict) -> str:
    return f"{r['name']} ({r['symbol']}): {r['close']:.2f} ({r['pct']:+.2f}%)"


def _fmt_section(rows: list[dict]) -> str:
    return "\n".join(_fmt_row(r) for r in rows)


def _fmt_headlines(items: list[dict]) -> str:
    if not items:
        return "(nothing notable)"
    return "\n".join(
        f"- [{h['source']}] {h['title']}" + (f" — {h['summary'][:180]}" if h['summary'] else "")
        for h in items
    )


def build_prompt(market: dict, headlines_by_cat: dict[str, list[dict]], date_str: str) -> str:
    indices = _fmt_section(market["indices"])
    sectors = _fmt_section(market["sectors"])
    macro = _fmt_section(market["macro"])
    gainers = _fmt_section(market["gainers"])
    losers = _fmt_section(market["losers"])
    char_lines = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in CHARACTERS.items()
    )
    names = " and ".join(CHARACTERS.keys())
    return f"""You write MARKET TODAY EXPLAINED — a fast, witty daily brief in the spirit of Morning Brew, delivered as a two-host podcast DIALOGUE.
Date: {date_str}

CO-HOSTS:
{char_lines}

==== MARKET DATA ====
INDICES:
{indices}

SECTORS:
{sectors}

MACRO:
{macro}

TOP GAINERS:
{gainers}

TOP LOSERS:
{losers}

==== HEADLINES (last 24h) ====

[MARKETS]
{_fmt_headlines(headlines_by_cat.get("markets", []))}

[BUSINESS]
{_fmt_headlines(headlines_by_cat.get("business", []))}

[TECH]
{_fmt_headlines(headlines_by_cat.get("tech", []))}

[WORLD]
{_fmt_headlines(headlines_by_cat.get("world", []))}

[CULTURE]
{_fmt_headlines(headlines_by_cat.get("culture", []))}

==== WRITE THE EPISODE ====

Hard rules:

1. FORMAT: every spoken line as `NAME: text` on its own line. Only {names} as speakers. No stage directions.
2. STRUCTURE — loose Morning Brew segment flow:
   a. COLD OPEN — JAMIE hits a punchy, specific, curiosity-grabbing one-liner. No "welcome" openers.
   b. MARKETS — what moved and WHY. Index numbers, sector leaders/laggards, macro (rates, dollar, oil), 2-3 notable single-stock movers with actual headline reasons.
   c. BIG STORY OF THE DAY — pick the single juiciest business/tech story from the headlines and dig in for 60-90 seconds of back-and-forth.
   d. QUICK HITS — 3 to 5 rapid-fire short exchanges on other business/tech/world stories, one per beat.
   e. ODD THING — one unusual, fun, or culture story to close (the "Morning Brew" signature odd-note).
   f. SIGN-OFF — two quick banter lines.
3. VOICE: ping-pong pace. Most turns 1-3 sentences. JAMIE drives and reacts with personality. ALEX delivers the "why" and the dry takes. Inject real reactions ("oh that's wild", "wait what"), light humor, skepticism — always tied to real info, never forced.
4. NUMBERS: write as words for smooth TTS. "Up one point two percent." For indices spell each digit: seventy-one twenty-six for 7126. Tickers spelled as letters with spaces: S P Y, N V D A.
5. ACCURACY: never invent facts. If the tape moved without a clear catalyst, ALEX says so. If a story's details aren't in the headlines, stay vague rather than guess.
6. LENGTH: adaptive. Quiet day → around {MIN_WORDS} words. Busy day → up to {MAX_WORDS} words. Never pad. Never skip a great story.
7. OUTPUT: ONLY the `NAME: line` lines. No markdown, no headers, no section labels, no intro/outro commentary. First line must start with `JAMIE:`.
"""


def generate(market: dict, headlines_by_cat: dict[str, list[dict]], date_str: str) -> str:
    prompt = build_prompt(market, headlines_by_cat, date_str)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.75, "num_ctx": 8192},
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
