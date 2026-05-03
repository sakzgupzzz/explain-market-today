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
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL, GROQ_CRITIC_MODEL,
    MIN_WORDS, MAX_WORDS, MIN_TURNS, CHARACTERS, PODCAST_TITLE,
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


# Few-shot exchange. Demonstrates: name-first intros (each host uses their own
# name), audio tags (sparingly, in-character), VARIED turn lengths (short
# reactions interleaved with substantive turns), ping-pong rhythm, real
# company names, ban-list compliance, ticker spelled with spaces.
#
# IMPORTANT: every concrete fact in this example is FABRICATED — none of
# these companies, prices, or events exist or happened. The example is here
# to teach STRUCTURE only. The model must use stories from the HEADLINES
# block, NEVER from this example. The placeholder names below are obviously
# fake (FabriCo, Quanto, Thrune Bank) so any leak is easy to catch in review.
_FEW_SHOT = """<example_episode>
JAMIE: Jamie here — and FabriCo just announced a forty-eight billion dollar buyback while their warehouse workers are on strike. ALEX, untangle this for us.
ALEX: [deadpan] Alex on equities. F A B R is up six point two percent on the announcement. The buyback is roughly nine times last year's R and D budget. Make of that what you will.
CAM: Cam on macro — and the dollar index is up one tenth of a percent overnight, which is to say nothing happened.
JAMIE: Right, exactly.
MAYA: [excited] Maya from tech. Quanto Robotics shipped their household model and the Wall Street Journal review is brutal — quote, "less useful than a blender."
KAI: A blender does one thing well, though.
MAYA: That's the point.
RIO: Rio checking in — the strike at FabriCo's Memphis plant is into day six. That's the human side here. Forty-two hundred workers.
TESS: Tess at retail — and the buyback is a tell. Companies announce buybacks in week two of strikes. It's a pressure move.
DEV: [snorts] Dev on crypto — meanwhile Bitcoin did absolutely nothing today. Refreshing.
JAMIE: [laughs] Sure.
ALEX: Final note on F A B R — the CEO sold one point three million dollars of stock yesterday afternoon. That's in the eight K filing.
JAMIE: Wait, what?
ALEX: [sarcastic] You heard me.
KAI: [mischievously] Kai with the weird thing. A man in Vermont broke into Thrune Bank — to deposit money. Said he didn't trust ATMs. They arrested him.
JAMIE: [laughs] That's it. Wrap it. This show is for entertainment and education only — nothing here is investment advice.
</example_episode>

NOTICE: the example above used FAKE companies (FabriCo, Quanto Robotics, Thrune Bank). Your output must use REAL companies and events from the HEADLINES block above. Do NOT mention FabriCo, Quanto, Thrune Bank, the Memphis strike, the Vermont bank-deposit story, or any other story shown in the example. Mimic the rhythm — varied turn lengths, ping-pong reactions, audio tags — not the content."""


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
5. PING-PONG RHYTHM (HARD): the show is a CONVERSATION, not a series of monologues. Hosts INTERRUPT, REACT, RIFF on each other constantly. Vary turn lengths deliberately:
   - ~60% of turns are SHORT reactions (5-15 words): "Wait, what?", "Come on.", "Hold on, that's not right.", "Oh no.", "Yeah but —", "Right, exactly.", "[laughs] Sure.", "That's it?", "No way.", "Of course it is."
   - ~30% are MEDIUM substantive turns (15-40 words): one fact + one reaction.
   - ~10% are LONG explainers (40-80 words) — the BIG STORY beat lead, mostly.
   - After every substantive turn (medium or long), AT LEAST ONE other host MUST react with a short turn before the next substantive turn.
   - No host speaks 2 turns in a row unless the line is genuinely continuous.
   - Episodes that read as a sequence of equal-length monologues will be rejected and regenerated.
6. SPECIFICS (HARD): every substantive turn must contain a CONCRETE FACT from the SOURCE DATA — a company name, a price, a percentage, a dollar amount, a person's name, a place, a date. Vague observations like "the market is wild today" without a specific number are banned. Cite the data, then react to it.
7. VOICE: contractions, em-dashes, ellipses for natural pauses. Hosts cut each other off, finish each other's sentences, push back.
8. NUMBERS: write as words for smooth TTS. "Up one point two percent." For indices spell digit-pairs: "seventy-one twenty-six" for 7126. Tickers as letters with spaces: "S P Y", "N V D A", "C R M". Dollar amounts: "one hundred billion dollars", not "$100B".
9. ACCURACY (HARD): you may ONLY discuss companies, stories, prices, percentages, and events that appear verbatim in MARKET DATA or HEADLINES above. Do NOT invent stock moves, headlines, deals, endorsements, shutdowns, or quotes. If a beat has no source material, skip the beat. If the tape moved without a clear catalyst, ALEX says exactly that.
10. LENGTH: adaptive but DENSE. Quiet day → around {MIN_WORDS} words across AT LEAST {MIN_TURNS} turns. Busy day → up to {MAX_WORDS} words across 40+ turns. Never pad. Never skip a great story that's in the headlines. If you produce fewer than {MIN_TURNS} turns the episode is rejected.
11. CAST USAGE: ALL {total_hosts} hosts SHOULD appear; minimum {min_hosts}. JAMIE bookends but speaks AT MOST 1 in every 4 turns. No single host gets more than 25% of total turns.
12. HUMOR: jokes throughout, organic to each host's personality. Punch up at institutions/Wall Street/PR spin. Never punch down at protected characteristics. No dad jokes. Late callback to an earlier joke = chef's kiss.
13. BANNED_PHRASES — do NOT use any of these (case-insensitive): {banned}.
14. OUTPUT: ONLY `NAME: line` lines. No markdown, no headers, no section labels, no intro/outro commentary, no sponsors, no fictional podcast brand names. Real company/product names are fine and expected. First line must start with `JAMIE:`. Last line must contain the disclaimer verbatim.
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


def _groq_call(prompt: str, model: str, temperature: float = 0.75) -> str:
    """Groq's OpenAI-compatible chat-completions endpoint. Sub-second inference
    for open-weight models on their custom LPU hardware."""
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _llm_call(prompt: str, ollama_model: str, groq_model: str, temperature: float = 0.75) -> str:
    """Dispatch to Groq when GROQ_API_KEY is present, else Ollama. Local dev
    stays on Ollama unless the env var is exported."""
    if GROQ_API_KEY:
        return _groq_call(prompt, groq_model, temperature)
    return _ollama_call(prompt, ollama_model, temperature)


_TURN_LINE_RE = __import__("re").compile(r"^[A-Z][A-Z0-9_]{0,15}:\s*\S")


def _count_turns(script: str) -> int:
    return sum(1 for line in script.splitlines() if _TURN_LINE_RE.match(line))


def generate(
    market: dict,
    headlines_by_cat: dict[str, list[dict]],
    date_str: str,
    max_retries: int = 1,
) -> str:
    """Generate the dialogue. If the result has fewer than MIN_TURNS turns,
    retry once with a stronger 'more turns, more reactions' addendum.
    Sleeps before retry to clear Groq's per-minute TPM window."""
    import time
    prompt = build_prompt(market, headlines_by_cat, date_str)
    script = _llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.75)
    turns = _count_turns(script)
    print(f"[generate] first pass: {turns} turns")

    attempts = 0
    while turns < MIN_TURNS and attempts < max_retries:
        attempts += 1
        if GROQ_API_KEY:
            print("[generate] sleeping 35s to clear Groq TPM window before retry…")
            time.sleep(35)
        addendum = (
            f"\n\nYour previous draft had only {turns} turns. The minimum is "
            f"{MIN_TURNS}. Rewrite the episode with MORE turns — break monologues "
            f"into a long-turn-followed-by-short-reaction pattern, add reactions "
            f"between every substantive turn, and use more hosts. Aim for 30-40 turns."
        )
        retry_prompt = prompt + addendum
        try:
            retried = _llm_call(retry_prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.85)
            new_turns = _count_turns(retried)
            print(f"[generate] retry {attempts}: {new_turns} turns")
            if new_turns > turns:
                script = retried
                turns = new_turns
        except Exception as e:
            print(f"[generate] retry {attempts} failed ({e}); keeping first-pass script")
            break
    return script


CRITIQUE_HEADLINES_PER_BEAT = 12  # cap to keep request well under 32k ctx


def _critique_prompt(script: str, market: dict, headlines_by_cat: dict[str, list[dict]]) -> str:
    """Build a prompt that asks the LLM to critique + revise the draft script."""
    indices = _fmt_section(market.get("indices", []))
    sectors = _fmt_section(market.get("sectors", []))
    macro = _fmt_section(market.get("macro", []))
    gainers = _fmt_section(market.get("gainers", []))
    losers = _fmt_section(market.get("losers", []))
    headlines_block = "\n\n".join(
        f"[{cat.upper()}]\n{_fmt_headlines(items[:CRITIQUE_HEADLINES_PER_BEAT])}"
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
        return _llm_call(prompt, OLLAMA_CRITIC_MODEL, GROQ_CRITIC_MODEL, temperature=0.2)
    except Exception as e:
        print(f"[critique] failed, returning unrevised script: {e}")
        return script


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines
    m = fetch_all()
    h = fetch_headlines()
    print(generate(m, h, datetime.now().strftime("%A, %B %d, %Y")))
