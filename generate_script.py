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


def _fmt_ranked_stories(ranked: list[dict], top_n: int = 15, compact: bool = False) -> str:
    """Render a pre-ranked story list. Each story shows its score, sources,
    title, and clipped summary. Stories are tagged with category for
    LLM-side beat routing.

    `compact=True` drops summaries and trims sources — used by the
    critique + verify passes so their request payloads stay under Groq's
    per-message size cap (40-story summaries blew past 413 Payload Too
    Large)."""
    if not ranked:
        return "(no stories ranked above the floor)"
    out = []
    for c in ranked[:top_n]:
        cats = "/".join(c.get("categories") or [])
        srcs_list = (c.get("sources") or [])[: 1 if compact else 3]
        srcs = ", ".join(srcs_list)
        title = c.get("title") or ""
        summary = c.get("summary") or ""
        score = c.get("score", 0)
        if compact:
            line = f"- [{score:>4.1f} · {cats} · {srcs}] {title[:120]}"
        else:
            line = f"- [score {score:>5.1f} · {cats} · {srcs}] {title}"
            if summary:
                line += f"\n    {_clip_summary(summary, 200)}"
        out.append(line)
    return "\n".join(out)


def _fmt_followups(seen_recently: list[dict]) -> str:
    if not seen_recently:
        return ""
    lines = ["RECENTLY COVERED (mention only with a fresh follow-up; do NOT re-explain):"]
    for c in seen_recently:
        lines.append(f"- {c.get('title','')}")
    return "\n".join(lines)


_TONE_FRAGMENTS = {
    "neutral": "Hosts are professional, lightly witty, accurate-first.",
    "dry": "Hosts are dry, deadpan, sardonic. Punchlines land late and quiet, not loud. Skepticism over enthusiasm.",
    "snarky": "Hosts are sharp, snarky, and openly take the piss out of corporate spin and Wall Street narrative. Punchier, more confrontational than dry.",
}

# Prompt-template versioning. PROMPT_VARIANT env override lets you A/B
# test prompts: tag the meta sidecar with the variant, then aggregate
# .meta.json to compare turn count / word count / banned-phrase rate /
# topic diversity per variant. Default 'A'. Add 'B' / 'C' branches inside
# build_prompt as needed.
import os as _os
PROMPT_VERSION = "v1.4"
PROMPT_VARIANT = _os.environ.get("PROMPT_VARIANT", "A").upper()

_LENGTH_PRESETS = {
    "short": {"min_words": 600, "max_words": 1500, "min_turns": 22},
    "standard": {"min_words": 1000, "max_words": 2700, "min_turns": 30},
    "long": {"min_words": 1400, "max_words": 3500, "min_turns": 38},
}


def _resolve_prefs(interests: dict | None) -> tuple[str, dict]:
    prefs = (interests or {}).get("preferences") or {}
    tone = (prefs.get("tone") or "dry").lower()
    if tone not in _TONE_FRAGMENTS:
        tone = "dry"
    length = (prefs.get("length") or "standard").lower()
    length_preset = _LENGTH_PRESETS.get(length, _LENGTH_PRESETS["standard"])
    return tone, length_preset


def build_prompt(
    market: dict,
    ranked_stories: list[dict],
    date_str: str,
    follow_ups: list[dict] | None = None,
    upcoming_events: str = "",
    interests: dict | None = None,
) -> str:
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
    followups_block = _fmt_followups(follow_ups or [])
    tone, length_preset = _resolve_prefs(interests)
    tone_line = _TONE_FRAGMENTS[tone]
    pref_min_words = length_preset["min_words"]
    pref_max_words = length_preset["max_words"]
    pref_min_turns = length_preset["min_turns"]

    return f"""You write {title_upper} — a fast, daily brief delivered as a multi-host podcast roundtable. {tone_line} Hosts joke, riff, push back on each other.
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

==== TOP STORIES (pre-ranked by importance — pick from these in score order) ====
{_fmt_ranked_stories(ranked_stories)}

{followups_block}

{upcoming_events}

==== STYLE EXAMPLE (DO NOT COPY CONTENT, ONLY MIMIC STRUCTURE/TONE) ====
{_FEW_SHOT}

==== WRITE THE EPISODE ====

Hard rules:

1. FORMAT: every spoken line as `NAME: text` on its own line. Only these names allowed: {names}. No stage directions outside audio tags. No markdown, no section labels.
2. AUDIO TAGS: optional in-line tags from each host's allowed list (e.g. `[deadpan]`, `[laughs]`, `[excited]`). Use AT MOST ONE per turn, ONLY when it adds emotion not already obvious from the words. Do NOT write "um", "uh", "mm-hmm" — let the audio model add disfluencies. Do NOT write `[pause]` between turns; turn-taking already creates pauses.
3. NAME-FIRST INTROS: when a host takes the mic, the FIRST sentence names THE HOST WHO IS NOW SPEAKING — never another host. ALEX says "Alex here" or "Alex on the desk", never "Jamie here". Vary phrasings: "Alex again,", "Maya here,", "Maya from tech,", "Alex on markets —", "Jamie back —". Never repeat the exact same phrase twice. If continuing your own turn uninterrupted, no need to reannounce.
4. STRUCTURE — three-host roundtable. The order below is mandatory. Skip any beat whose data is empty:
   a. COLD OPEN (1 turn) — JAMIE one punchy specific line referencing an ACTUAL story. Banned phrases below.
   b. MARKETS (3-5 turns) — ALEX leads with INDEX moves (S&P, Nasdaq, Dow, VIX) + biggest gainer + biggest loser + macro (rates, dollar). MUST come second, right after cold open. JAMIE/MAYA add ONE reaction each. If MARKET DATA empty, skip beat.
   c. BIG STORY (5-7 turns MAX) — pick ONE story from the top 3 ranked. Beat lead is whichever host's desk it falls under. The other two chime in with jokes, skepticism, push-back. Then move on. Do NOT over-cover one story.
   d. QUICK HITS (8-12 turns total, 1-3 turns per story) — rapid rotation across at least 4 DIFFERENT stories. Each story gets one substantive turn + one reaction. Move on. No story gets more than 3 quick-hits turns. Mix beat ownership.
   e. ODD THING (2-3 turns) — MAYA closes with ONE unusual story from culture/world. JAMIE or ALEX reacts once. Move on quickly.
   f. SIGN-OFF (3-4 turns) — callback to a joke earlier in the episode, then JAMIE reads disclaimer verbatim: "{DISCLAIMER_SHORT}". The callback is REQUIRED — reference something specific said earlier (a host's joke, an absurd company name, etc.).
4a. TOPIC BUDGET (HARD): NO single story gets more than 7 turns total across the entire episode. If you find yourself writing a 5th turn about the same company/event, MOVE ON. Use AT LEAST 6 different stories from the TOP STORIES list — episodes that focus on 1-2 stories get rejected.
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
10. LENGTH: adaptive but DENSE. Quiet day → around {pref_min_words} words across AT LEAST {pref_min_turns} turns. Busy day → up to {pref_max_words} words. HARD MAX: 35 turns. If you produce fewer than {pref_min_turns} turns the episode is rejected. If you produce more than 35 turns consolidate — pack two reactions into one substantive turn rather than spreading thin. Going past 35 turns disables the critique pass and ships a less-polished episode, so stay at or below 35.
11. CAST USAGE: ALL {total_hosts} hosts SHOULD appear; minimum {min_hosts}. JAMIE bookends but speaks AT MOST 1 in every 4 turns. No single host gets more than 25% of total turns.
12. HUMOR: jokes throughout, organic to each host's personality. Punch up at institutions/Wall Street/PR spin. Never punch down at protected characteristics. No dad jokes. Late callback to an earlier joke = chef's kiss.
13. BANNED_PHRASES — do NOT use any of these (case-insensitive): {banned}.
14. OUTPUT: ONLY `NAME: line` lines. No markdown, no headers, no section labels, no intro/outro commentary, no sponsors, no fictional podcast brand names. Real company/product names are fine and expected. First line must start with `JAMIE:`.
15. DISCLAIMER (HARD): EXACTLY ONE turn in the entire episode contains the disclaimer "{DISCLAIMER_SHORT}". It MUST be the very last line of the script. ZERO other turns may contain that line or paraphrases of it. No banter, jokes, or sign-off chatter after the disclaimer — the disclaimer turn ends the episode. If you find yourself writing more than one disclaimer line, delete all but the final one.
16. SELF-REFERENCE: no host ever addresses themselves by name. JAMIE never says "Later, Jamie" or "Jamie out". Hosts pass the mic to OTHER hosts. Sign-off pattern: each host says ONE short closing line in their own voice (callback to an earlier joke), then JAMIE reads the disclaimer once.
17. NO TWO REACTIONS BACK-TO-BACK: a "reaction" is a turn with no concrete fact — just an emotional response ("Right, exactly", "Yeah, sure", "[laughs] Of course it is", "Wait, what?", "Come on"). After ANY reaction turn the next turn MUST be substantive (contain a specific number, name, place, or new fact from the source data). Two reaction turns in a row is banned.
18. NO REPEATED JOKE TEMPLATES: each sentence frame appears AT MOST TWICE per episode. Banned-after-second-use frames include: "what every X needs is Y", "because that's exactly what we need, more X", "right, exactly", "of course it is", "who doesn't love a good X". Use varied syntax for sarcasm — DIFFERENT structure each time.
19. NO HOST SPEAKS TWO TURNS IN A ROW. If two consecutive `NAME:` lines have the same NAME, you must either merge them or insert a turn from a different host between them.
20. NO TEASER TURNS: every story-introducing turn must INCLUDE the substantive content in the same turn. Banned: "Did you hear about X?" / "Speaking of X..." with no fact in the same turn or the next. If you can't deliver the substance, cut the story entirely. Vague-question-followed-by-vague-non-answer pairs are banned.
21. SIGN-OFF CALLBACK (HARD): the final 2-3 turns before the disclaimer MUST contain a SPECIFIC callback — repeat a line another host said, name a company from earlier, riff on a specific joke from the body of the episode. Generic outros ("we'll catch you tomorrow", "have a great day") are banned.
22. KEEP NUMBERS WHEN PRESENT: every fact about a company, index, ticker, or move includes the SPECIFIC number from the source data. "S and P is up" without a percent is banned. "Treasury yield is down" without basis points is banned. Cite the data, then react to it.
23. NO REPEATED TURN-STARTERS: the opening 1-3 words of consecutive turns must vary. Banned starter patterns when used MORE THAN TWICE: "And in other news...", "What about...", "Yeah, and...", "[curious] What...", "Oh, and...", "Speaking of...". After their second use, the third+ instance MUST start with different words.
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


def _groq_call(prompt: str, model: str, temperature: float = 0.75, retries: int = 2) -> str:
    """Groq's OpenAI-compatible chat-completions endpoint with retry on
    transient 5xx + 429 (Groq's rate-limit signal). Backoff is exponential
    with jitter; 429 honors Retry-After header if present."""
    import random, time as _time
    last_err: Exception | None = None
    print(f"[groq] {model} prompt={len(prompt)} chars (~{len(prompt)//4} tokens)")
    for attempt in range(retries):
        try:
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
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", "5"))
                print(f"[groq] 429 rate-limited, waiting {wait}s (attempt {attempt+1}/{retries})")
                _time.sleep(wait + random.random())
                continue
            if resp.status_code == 413:
                # Empirically, Groq returns 413 when free-tier TPM is
                # exhausted (not just for genuinely-oversized payloads).
                # Wait a full minute to clear the per-minute window.
                wait = 62 + random.random() * 3
                print(f"[groq] 413 (likely TPM exhausted), waiting {wait:.0f}s (attempt {attempt+1}/{retries})")
                _time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = (2 ** attempt) + random.random()
                print(f"[groq] {resp.status_code} server error, retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            last_err = e
            wait = (2 ** attempt) + random.random()
            print(f"[groq] request failed ({e}), retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
            _time.sleep(wait)
    raise RuntimeError(f"groq exhausted {retries} retries: {last_err}")


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
    ranked_stories: list[dict],
    date_str: str,
    follow_ups: list[dict] | None = None,
    upcoming_events: str = "",
    interests: dict | None = None,
    max_retries: int = 1,
) -> str:
    """Generate the dialogue from a pre-ranked story list. If the result has
    fewer than the length preset's min_turns, retry once with a stronger
    'more turns' addendum. Sleeps before retry to clear Groq's per-minute
    TPM window."""
    import time
    _, length_preset = _resolve_prefs(interests)
    target_min_turns = length_preset["min_turns"]
    prompt = build_prompt(
        market, ranked_stories, date_str,
        follow_ups=follow_ups, upcoming_events=upcoming_events,
        interests=interests,
    )
    script = _llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.75)
    turns = _count_turns(script)
    print(f"[generate] first pass: {turns} turns")

    attempts = 0
    while turns < target_min_turns and attempts < max_retries:
        attempts += 1
        if GROQ_API_KEY:
            print("[generate] sleeping 35s to clear Groq TPM window before retry…")
            time.sleep(35)
        addendum = (
            f"\n\nYour previous draft had only {turns} turns. The minimum is "
            f"{target_min_turns}. Rewrite the episode with MORE turns — break monologues "
            f"into a long-turn-followed-by-short-reaction pattern, add reactions "
            f"between every substantive turn, and use more hosts. Aim for {target_min_turns + 6}-{target_min_turns + 12} turns."
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


def _critique_prompt(script: str, market: dict, ranked_stories: list[dict]) -> str:
    """Critique prompt — STRUCTURAL fixes only. Tight ruleset to stay under
    Groq's ~7KB request size cap. Source facts deliberately omitted:
    fabrication checks happen in verify_facts."""
    name_list = ", ".join(CHARACTERS.keys())
    return f"""You are a strict podcast script editor. Revise the DRAFT below to fix the listed structural issues ONLY. Surgical cuts only — do not restructure or trim aggressively.

Hosts: {name_list}. Disclaimer: "{DISCLAIMER_SHORT}".

PRESERVE (do NOT remove these):
- Audio tags like [deadpan], [laughs], [excited], [sarcastic], [sighs], [mischievously], [rushed], [curious] — keep them ALL.
- Specific numbers: even if you spell them as words, the SCRIPT must still mention 'A M D up four point six four percent' not just 'A M D up'. Spell digits, never delete them.
- All real company / product / person names from the draft.
- Every distinct story the draft references — do not drop a story to satisfy any rule.
- Banter and jokes that are NOT on the banned-template list.
- Turn count: keep within 80% of the original. If draft has 31 turns, the revised script has at least 25.

Fix:
- Wrong-name intro: ALEX saying "Jamie here" → "Alex here". Same for all hosts.
- Mid-turn vocative: a host addressing themselves like "What's the story, Jamie?" from JAMIE → drop ", Jamie". Same for ALEX and MAYA.
- Same speaker twice in a row → merge into one turn OR insert a different host between.
- JAMIE > 1 in every 4 turns → drop short filler JAMIE turns until at cap.
- Numbers/tickers in spoken text: $5B / 5% / NVDA / AAPL → spelled words ("five billion dollars", "five percent", "N V D A", "A A P L").
- Banned cold-open phrases: "welcome to the show", "let's dive in", "good morning everyone", "buckle up", "well folks", "stay tuned".
- Banned mid-script templates (each may appear AT MOST twice): "right, exactly", "of course it is", "okay, okay let's move on", "because that's exactly what we need", "what every X needs is Y", "who doesn't love a good X", "what we really need is more X". On 3rd+ occurrence rewrite with different syntax.
- Disclaimer dedup: if the disclaimer line appears more than once, delete every occurrence except the final one. The disclaimer must be the very last line. Drop any banter that comes after.
- Disclaimer missing: add as last JAMIE line.
- Two pure-reaction turns ("Right, exactly" + "[laughs] Sure" with no fact between) → drop the weaker or rewrite the second with a substantive fact.
- Format integrity: every line must be `NAME: text` with NAME in [{name_list}]. Drop narration, stage directions, headers.
- Teaser turns: any "Did you hear about X?" / "Speaking of X" turn that's NOT immediately followed by a turn delivering the substantive fact → drop the teaser turn entirely. Don't keep both halves of a vague-question-followed-by-vague-non-answer pair.
- Missing numbers: if a turn mentions a market/index/ticker/move WITHOUT the specific number from the data (e.g. "S and P is up" with no percent), and the original draft had the number, restore the number. Do NOT delete numbers.
- Sign-off callback: if the final 2-3 turns before the disclaimer have NO specific callback (no reference to a host's earlier line, no company named earlier in the episode), add one specific callback to the second-to-last turn before the disclaimer.

Output ONLY the revised script in `NAME: line` format. No commentary, no diff.

==== DRAFT ====
{script}

==== REVISED ====
"""


CRITIQUE_MAX_PROMPT_CHARS = 7800  # Groq free-tier per-message cap is ~7.8-8KB
                                  # (empirical: 6795 worked, 8241 failed)


def critique_revise(script: str, market: dict, ranked_stories: list[dict]) -> str:
    """Run a critic LLM pass to fix fabrications, banned phrases, monologues.
    Pre-sleeps 8 sec when Groq is in use so the per-message-burst limit
    (manifests as 413) has time to clear after the generate call.

    Skips entirely if the resulting prompt would exceed Groq's per-message
    size cap — better to ship the unrevised script (sanitize still runs)
    than burn 2-3 min waiting for retries that always fail at that size."""
    import time as _time
    prompt = _critique_prompt(script, market, ranked_stories)
    if GROQ_API_KEY and len(prompt) > CRITIQUE_MAX_PROMPT_CHARS:
        print(f"[critique] script too large ({len(prompt)} chars > {CRITIQUE_MAX_PROMPT_CHARS} cap); skipping critique pass")
        return script
    if GROQ_API_KEY:
        _time.sleep(8)
    try:
        return _llm_call(prompt, OLLAMA_CRITIC_MODEL, GROQ_CRITIC_MODEL, temperature=0.2)
    except Exception as e:
        print(f"[critique] failed, returning unrevised script: {e}")
        return script


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines, flatten
    from cluster import cluster_headlines
    from score import score_clusters
    from interests_loader import load_interests
    m = fetch_all()
    h = fetch_headlines()
    clusters = cluster_headlines(flatten(h))
    ranked = score_clusters(clusters, m, load_interests())
    print(generate(m, ranked, datetime.now().strftime("%A, %B %d, %Y")))
