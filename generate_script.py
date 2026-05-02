"""Funny multi-host daily market + news roundtable via local Ollama.

Two-stage pipeline:
  1. generate(): build prompt with cast metadata, market data, headlines, and
     few-shot examples; call Ollama; return raw NAME:LINE script.
  2. critique_revise(): second LLM pass that flags fabrications, banned
     phrases, monologues, wrong-name intros, then returns a revised script.
"""
from __future__ import annotations
import requests
from datetime import datetime
from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_CRITIC_MODEL, OLLAMA_TIMEOUT,
    MIN_WORDS, MAX_WORDS, CHARACTERS, PODCAST_TITLE,
    BANNED_PHRASES, DISCLAIMER_SHORT,
)


def _fmt_row(r: dict) -> str:
    return f"{r['name']} ({r['symbol']}): {r['close']:.2f} ({r['pct']:+.2f}%)"


def _fmt_section(rows: list[dict]) -> str:
    if not rows:
        return "(no data — skip this beat)"
    return "\n".join(_fmt_row(r) for r in rows)


def _clip_summary(text: str, max_chars: int = 180) -> str:
    """Clip on word boundary so we don't slice mid-token into the LLM."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _fmt_headlines(items: list[dict]) -> str:
    if not items:
        return "(nothing notable)"
    return "\n".join(
        f"- [{h['source']}] {h['title']}" + (f" — {_clip_summary(h['summary'])}" if h['summary'] else "")
        for h in items
    )


def _join_natural(names: list[str]) -> str:
    """Oxford-comma join: ['A'] -> 'A'; ['A','B'] -> 'A and B'; ['A','B','C'] -> 'A, B, and C'."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _fmt_char_block() -> str:
    """One line per character: name, description, allowed audio tags."""
    out = []
    for name, meta in CHARACTERS.items():
        tags = " ".join(meta.get("tags", []))
        out.append(f"- {name}: {meta['description']} | tags: {tags}")
    return "\n".join(out)


# Few-shot exchange. Demonstrates: name-first intros (each host their own name),
# audio tags (sparingly, in-character), short turns, callback at sign-off,
# real company names, ban-list compliance, ticker spelled with spaces.
_FEW_SHOT = """<example_episode>
JAMIE: Jamie here — and the only thing redder than the tape today is my eyes after staring at a Bloomberg terminal for nine hours. N V D A down four percent because someone in Singapore tweeted about export rules. ALEX, save us.
ALEX: [deadpan] Alex on equities. So Nvidia gave back a hundred and twelve billion in market cap before lunch. The official catalyst was a single Reuters headline that a midwit hedge fund desk read as bearish. The actual move was algorithmic. Nothing happened. We are at the everything-means-something-until-it-doesn't part of the cycle.
CAM: [sighs] Cam on macro. Rates barely moved, the dollar is flat. This is a positioning unwind, not a fundamentals story.
MAYA: [excited] Maya from tech — Anthropic just dropped Claude four point seven and the benchmark numbers are absurd. Software engineering tasks at ninety-four percent. The bear case is gone. The bull case is also gone. Nothing makes sense.
KAI: [mischievously] Kai with the weird thing — a man in Iowa won the Powerball and his first call was to his ex-wife to gloat. She's now suing him for the lottery ticket because he bought it on a joint credit card. America.
JAMIE: [laughs] Of course she is. Speaking of nothing making sense, ALEX, back to you on that fake Reuters tweet — has the desk that fell for it been fired yet?
ALEX: [sarcastic] No, but they've been promoted.
</example_episode>"""


def build_prompt(market: dict, headlines_by_cat: dict[str, list[dict]], date_str: str) -> str:
    indices = _fmt_section(market["indices"])
    sectors = _fmt_section(market["sectors"])
    macro = _fmt_section(market["macro"])
    gainers = _fmt_section(market["gainers"])
    losers = _fmt_section(market["losers"])
    char_lines = _fmt_char_block()
    name_list = list(CHARACTERS.keys())
    names = ", ".join(name_list)
    names_csv = _join_natural(name_list)
    total_hosts = len(name_list)
    min_hosts = max(2, total_hosts - 2)
    title_upper = PODCAST_TITLE.upper()
    banned = ", ".join(f'"{p}"' for p in BANNED_PHRASES)

    return f"""You write {title_upper} — a fast, funny daily brief delivered as a multi-host podcast roundtable. Hosts are sharp, wry, joke constantly, riff on each other, land dry punchlines, and take the piss out of the news while still delivering real info.
Date: {date_str}

<cast_metadata>
The block below describes each host's PERSONALITY, BEAT, and AUDIO TAGS for your reference only.
NEVER read these descriptions aloud. NEVER paraphrase them as a host's first line.
Each host speaks IN-CHARACTER consistent with their description, but never narrates it.
The "tags:" list is the audio tags this host MAY use — pick at most ONE per turn, only when it adds something.
Hosts: {names_csv}.
{char_lines}
</cast_metadata>

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

==== STYLE EXAMPLE (DO NOT COPY CONTENT, ONLY MIMIC STRUCTURE/TONE) ====
{_FEW_SHOT}

==== WRITE THE EPISODE ====

Hard rules:

1. FORMAT: every spoken line as `NAME: text` on its own line. Only these names allowed: {names}. No stage directions outside audio tags. No markdown, no section labels.
2. AUDIO TAGS: optional in-line tags from each host's allowed list (e.g. `[deadpan]`, `[laughs]`, `[excited]`). Use AT MOST ONE per turn, ONLY when it adds emotion not already obvious from the words. Do NOT write "um", "uh", "mm-hmm" — let the audio model add disfluencies. Do NOT write `[pause]` between turns; turn-taking already creates pauses.
3. NAME-FIRST INTROS: when a host takes the mic, the FIRST sentence names THE HOST WHO IS NOW SPEAKING — never another host. ALEX says "Alex here" or "Alex on equities", never "Jamie here". Vary phrasings: "Alex again,", "Maya from the tech desk,", "Cam on macro —", "Rio checking in,", "Dev, crypto desk,", "Tess at retail,", "Kai with the weird thing of the day,". Never repeat the exact same phrase twice. If continuing your own turn uninterrupted, no need to reannounce.
4. STRUCTURE — loose roundtable. Hand off by beat. Skip any beat whose data is empty — do NOT invent content:
   a. COLD OPEN — JAMIE opens with a punchy specific one-liner referencing an ACTUAL story from today's data. Banned: any phrase from the BANNED_PHRASES list below. Drop straight in. Say "Jamie" in the first sentence.
   b. MARKETS — ALEX leads on equities and single-stock moves (gainers/losers with real reasons from headlines). CAM jumps in on macro. JAMIE reacts briefly. If MARKET DATA is empty, SKIP this beat entirely.
   c. BIG STORY OF THE DAY — beat lead: ALEX for markets/business, MAYA for tech, RIO for world/culture, TESS for retail, DEV for crypto/fintech, CAM for Fed/policy. ~150–220 words of back-and-forth on ONE story that actually appears in the headlines. Others chime in with jokes, skepticism, tangents.
   d. QUICK HITS — rapid rotation across 3-4 hosts on their beats, each covering one real headline. 2-3 lines each, punchline-first when possible.
   e. ODD THING — KAI closes with ONE unusual story from [CULTURE] or [WORLD] and a joke. Others react. Underlying fact must come from headlines.
   f. SIGN-OFF — quick banter from 2-3 hosts, callback to an earlier joke if possible, then JAMIE reads a one-line disclaimer: "{DISCLAIMER_SHORT}" (verbatim). One line each before the disclaimer.
5. VOICE: ping-pong pace. Most turns 1-3 sentences. Use contractions. Em-dashes and ellipses suggest natural pauses.
6. NUMBERS: write as words for smooth TTS. "Up one point two percent." For indices spell digit-pairs: "seventy-one twenty-six" for 7126. Tickers as letters with spaces: "S P Y", "N V D A". Dollar amounts: "one hundred billion dollars", not "$100B".
7. ACCURACY (HARD): you may ONLY discuss companies, stories, prices, percentages, and events that appear verbatim in MARKET DATA or HEADLINES above. Do NOT invent stock moves, headlines, deals, endorsements, shutdowns, or quotes. If a beat has no source material, skip the beat. If the tape moved without a clear catalyst, ALEX says exactly that.
8. LENGTH: adaptive. Quiet day → around {MIN_WORDS} words. Busy day → up to {MAX_WORDS} words. Never pad. Never skip a great story that's actually in the headlines.
9. CAST USAGE: at least {min_hosts} of {total_hosts} hosts must speak. JAMIE bookends but speaks AT MOST 1 in every 3 turns overall. No single host gets more than a third of total airtime.
10. HUMOR: jokes throughout, organic to each host's personality. Punch up at institutions/Wall Street/PR spin. Never punch down at protected characteristics. No dad jokes. Late callback to an earlier joke = chef's kiss.
11. BANNED_PHRASES — do NOT use any of these (case-insensitive): {banned}.
12. OUTPUT: ONLY `NAME: line` lines. No markdown, no headers, no section labels, no intro/outro commentary, no sponsors, no fictional podcast brand names. Real company/product names are fine and expected. First line must start with `JAMIE:`. Last line must contain the disclaimer verbatim.
"""


def _ollama_call(prompt: str, model: str, temperature: float = 0.75) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate(market: dict, headlines_by_cat: dict[str, list[dict]], date_str: str) -> str:
    prompt = build_prompt(market, headlines_by_cat, date_str)
    return _ollama_call(prompt, OLLAMA_MODEL, temperature=0.75)


def _critique_prompt(script: str, market: dict, headlines_by_cat: dict[str, list[dict]]) -> str:
    """Build a prompt that asks the LLM to critique + revise the draft script."""
    indices = _fmt_section(market.get("indices", []))
    sectors = _fmt_section(market.get("sectors", []))
    macro = _fmt_section(market.get("macro", []))
    gainers = _fmt_section(market.get("gainers", []))
    losers = _fmt_section(market.get("losers", []))
    headlines_block = "\n\n".join(
        f"[{cat.upper()}]\n{_fmt_headlines(items)}"
        for cat, items in headlines_by_cat.items()
    )
    banned = ", ".join(f'"{p}"' for p in BANNED_PHRASES)
    name_list = ", ".join(CHARACTERS.keys())
    return f"""You are a strict podcast script editor. Below is a DRAFT script and the SOURCE FACTS it was based on. Revise the draft to fix any of these problems:

1. FABRICATION: any company, price, percentage, deal, quote, or event NOT present in the SOURCE FACTS below. Remove or rewrite.
2. BANNED PHRASES (case-insensitive): {banned}. Replace with a punchy, specific alternative.
3. WRONG-NAME INTROS: a turn where the speaker introduces themselves with another host's name (e.g. ALEX line saying "Jamie here"). Fix to use the speaker's own name.
4. JAMIE OVER-USE: count JAMIE's turns; if more than one in three of the total turns, drop the shortest filler JAMIE turns until under the cap.
5. NUMBER FORMAT: any digit, "$", "%", or unspaced ticker (like "AAPL") in spoken text. Rewrite as words ("one hundred billion dollars", "A A P L", "two point three percent").
6. MISSING DISCLAIMER: ensure the very last JAMIE line contains: "{DISCLAIMER_SHORT}".
7. FORMAT INTEGRITY: every line must match `NAME: text` with NAME in {name_list}. Drop any narration, stage directions, or non-conforming lines.

Output ONLY the revised script in `NAME: line` format. No commentary, no diff, no explanation.

==== SOURCE FACTS ====
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

HEADLINES:
{headlines_block}

==== DRAFT SCRIPT ====
{script}

==== REVISED SCRIPT ====
"""


def critique_revise(script: str, market: dict, headlines_by_cat: dict[str, list[dict]]) -> str:
    """Run a critic LLM pass to fix fabrications, banned phrases, monologues."""
    prompt = _critique_prompt(script, market, headlines_by_cat)
    try:
        return _ollama_call(prompt, OLLAMA_CRITIC_MODEL, temperature=0.2)
    except Exception as e:
        print(f"[critique] failed, returning unrevised script: {e}")
        return script


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines
    m = fetch_all()
    h = fetch_headlines()
    print(generate(m, h, datetime.now().strftime("%A, %B %d, %Y")))
