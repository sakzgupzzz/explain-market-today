"""Funny multi-host daily market + news roundtable via local Ollama."""
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
    names = ", ".join(CHARACTERS.keys())
    name_list = list(CHARACTERS.keys())
    names_csv = ", ".join(name_list[:-1]) + ", and " + name_list[-1] if len(name_list) > 1 else name_list[0]
    return f"""You write MARKET TODAY EXPLAINED — a fast, funny daily brief delivered as a multi-host podcast roundtable. The show is genuinely FUNNY — the hosts are sharp, wry, make jokes constantly, riff on each other, land dry punchlines, and take the piss out of the news (and each other) while still delivering the actual info.
Date: {date_str}

CAST ({names_csv}):
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

1. FORMAT: every spoken line as `NAME: text` on its own line. Only these names allowed: {names}. No stage directions.
2. STRUCTURE — loose roundtable flow. Hand off between hosts by beat:
   a. COLD OPEN — JAMIE opens with a punchy, specific, funny-on-the-edges one-liner. No "welcome to the show" openers. Drop the audience straight into a joke or a reaction.
   b. MARKETS (ALEX leads, JAMIE reacts) — index moves + WHY, sector leaders/laggards, macro (rates, dollar, oil), 2-3 notable single-stock movers with actual headline reasons. Crack jokes about the moves, the narratives, CEOs, Wall Street's mood.
   c. BIG STORY OF THE DAY (ALEX for markets/business, MAYA for tech, RIO for world/culture) — 60-90 seconds of real back-and-forth. Others chime in with jokes, skepticism, "wait, what", tangents.
   d. QUICK HITS — rapid rotation with jokes packed in: MAYA on a tech story, RIO on a world/culture story, ALEX on another business story. Short exchanges, 2-3 lines each, punchline-first when possible.
   e. ODD THING — KAI closes with one unusual, fun, or quirky story and a joke. Others react with more jokes, disbelief, or a running-gag callback.
   f. SIGN-OFF — quick banter, ideally with a callback to an earlier joke. One line per host, 2-3 hosts.
3. VOICE: ping-pong pace. Most turns 1-3 sentences. Each host speaks in their lane, but ALL hosts make jokes. JAMIE reacts and needles, ALEX does dry deadpan zingers, MAYA delivers fast-talker tech snark, RIO lands warm but cutting observations, KAI throws absurd one-liners. Natural filler allowed — "I mean", "come on", "oh no", "wait" — it reads as real speech.
4. NUMBERS: write as words for smooth TTS. "Up one point two percent." For indices spell each digit: seventy-one twenty-six for 7126. Tickers spelled as letters with spaces: S P Y, N V D A.
5. ACCURACY: never invent facts. If the tape moved without a clear catalyst, ALEX says so. If a story's details aren't in the headlines, stay vague rather than guess.
6. LENGTH: adaptive. Quiet day → around {MIN_WORDS} words. Busy day → up to {MAX_WORDS} words. Never pad. Never skip a great story.
7. EVERY host must speak at least twice. Do not let one host dominate.
8. HUMOR RULES:
   - Drop in jokes, wisecracks, sarcasm, callbacks, running gags throughout — not just the odd-thing segment.
   - Jokes should feel organic to each host's personality, not pasted on.
   - Punch up at institutions, Wall Street tropes, corporate PR spin, hype cycles. Never punch down at individuals' identities, appearance, or protected characteristics.
   - Skip cheesy dad jokes and overused memes. Aim for dry wit, absurdist observations, honest cynicism about spin.
   - A callback joke at the end (referencing something said earlier in the episode) is chef's kiss.
   - If a host makes a joke, another host can react to the joke — that's the point. Build on bits.
9. OUTPUT: ONLY the `NAME: line` lines. No markdown, no headers, no section labels, no intro/outro commentary, no brand names like Morning Brew. First line must start with `JAMIE:`.
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
