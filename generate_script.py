"""Turn market data + headlines into 2-host podcast dialogue via local Ollama."""
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
    char_lines = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in CHARACTERS.items()
    )
    names = " and ".join(CHARACTERS.keys())
    return f"""You write a fast-paced, entertaining daily US-market recap podcast as a DIALOGUE between two co-hosts.
Date: {date_str}

CO-HOSTS:
{char_lines}

MARKET DATA — INDICES:
{indices}

SECTORS:
{sectors}

MACRO:
{macro}

TOP GAINERS:
{gainers}

TOP LOSERS:
{losers}

HEADLINES (last 24h):
{news}

Write the episode as a two-person dialogue between {names}. Hard rules:

- Format EVERY line as `NAME: spoken text` on its own line. Example:
  JAMIE: Alright, we are back, and wow, the Dow ripped today.
  ALEX: Yeah, up almost two percent, and I will tell you exactly why.
- Only the two names above, in caps, followed by a colon.
- Short exchanges. Most turns 1-3 sentences. Ping-pong feel, not monologues.
- Open with a HOOK: a punchy, specific, curiosity-grabbing first line from JAMIE. Never "Hello and welcome."
- JAMIE drives and reacts. ALEX delivers the substance and the "why."
- Cover: index moves + why, sector leaders/laggards, macro context (rates, dollar, oil), 2-3 notable single-stock movers with the actual reason from headlines.
- Inject personality: quick reactions ("oh that's wild"), light jokes, skepticism, genuine surprise — tied to real data, never forced.
- Write NUMBERS AS WORDS for the tape ("up one point two percent", "seventy-one twenty-six" for 7126). Spell tickers as letters with spaces ("S P Y", "N V D A").
- Length adapts to news: quiet day → {MIN_WORDS}+ words, wild day → up to {MAX_WORDS} words. Never pad. Never rush past real news.
- No bullet points, headers, markdown, stage directions, music cues, or sound effects. Just `NAME: line` lines.
- Never invent facts. If the tape moved without a clear catalyst, ALEX should say so plainly.
- End with a quick banter sign-off (one line each, max).

Return ONLY the dialogue lines. No intro, no outro, no commentary. First line must start with `JAMIE:`."""


def generate(market: dict, headlines: list[dict], date_str: str) -> str:
    prompt = build_prompt(market, headlines, date_str)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 8192},
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
