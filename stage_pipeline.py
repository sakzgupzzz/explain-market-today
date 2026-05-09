"""Multi-stage script generation — replaces the single-shot generate prompt
with 7 focused stages that each render one beat at a time.

The single-shot approach drifted on long scripts: orphaned pronouns,
unsupported references, generic asides, disjointed quick-hits. Each stage
here gets a small focused prompt + the entire previous-turns context (free
on Haiku's 200k window), so coherence and callbacks become structural
rather than aspirational.

Pipeline:
   plan()              → JSON outline (story IDs per beat)
   render_cold_open()  → 1-2 turns
   render_markets()    → 4-6 turns
   render_big_story()  → 5-7 turns
   render_quick_hits() → 8-12 turns (2-3 per story)
   render_odd_thing()  → 3 turns
   render_sign_off()   → 3 turns + disclaimer
   stitch()            → concatenate

Each render stage shares context: the script-so-far is passed in as a
PREVIOUS TURNS block. Sign-off sees the entire script and is told what
specific thing to call back to (chosen at plan time).
"""
from __future__ import annotations
import json
import re
from typing import Any

from config import (
    CHARACTERS, DISCLAIMER_SHORT, BANNED_PHRASES,
)
from generate_script import (
    _llm_call, _resolve_prefs, _fmt_section, _join_natural, _fmt_char_block,
    OLLAMA_MODEL, GROQ_MODEL,
)

# ─────────── helpers ───────────

def _ranked_index(ranked: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for c in ranked if c.get("id")}


def _fmt_ranked_for_plan(ranked: list[dict], top_n: int = 14) -> str:
    """Compact list with story IDs the planner returns."""
    out = []
    for c in ranked[:top_n]:
        cats = "/".join(c.get("categories") or [])
        srcs = ", ".join((c.get("sources") or [])[:2])
        title = c.get("title") or ""
        out.append(f'  {c["id"]}  [{c.get("score",0):>4.1f}·{cats}·{srcs}] {title[:100]}')
    return "\n".join(out)


def _strip_json(text: str) -> str:
    """Pull a JSON object out of a model response that may have prose
    around it. Returns the substring from the first '{' to the matching
    closing '}'. Returns '' if no JSON found."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _normalize_lines(text: str) -> str:
    """Drop non-NAME: lines and collapse whitespace."""
    name_set = set(CHARACTERS.keys())
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^([A-Z][A-Z0-9_]{0,15}):\s*(.+)$", line)
        if not m:
            continue
        name = m.group(1)
        body = m.group(2).strip()
        if name in name_set and body:
            out.append(f"{name}: {body}")
    return "\n".join(out)


def _prev_turns_block(turns_so_far: list[str], char_limit: int = 6000) -> str:
    """Last N chars of the script so far, formatted for next-stage context."""
    text = "\n".join(turns_so_far)
    if len(text) > char_limit:
        # Keep tail (most recent context most relevant for callbacks)
        text = "…\n" + text[-char_limit:]
    return text or "(no prior turns yet — this is the start of the show)"


# ─────────── stage 1: PLAN ───────────

_PLAN_PROMPT = """You are a senior podcast producer building today's run-of-show. The cast is three hosts: {names_csv}. The show is a fast, daily news roundtable. {tone_line}

Below are the top-ranked news clusters from today, sorted by importance. Each has an ID. Build a beat-by-beat plan in JSON.

Hosts and beats they cover:
{char_lines}

CLUSTERS:
{ranked_block}

MARKET DATA SUMMARY:
{market_summary}
{yesterday_block}
Output ONLY a JSON object with this exact shape (no commentary, no markdown):

{{
  "cold_open": {{
    "story_id": "<one cluster id from above>",
    "hook": "<≤20 word punchy specific opener line, leveraging the actual story; MUST include at least one specific number (a percent move, a dollar amount, an index level) OR a specific company/person name with a fact attached. NO 'what's behind X' patterns.>"
  }},
  "markets": {{
    "lead_host": "ALEX",
    "key_numbers": ["S&P …", "Nasdaq …", "biggest gainer …", "biggest loser …"],
    "macro_note": "<one-line macro framing, e.g. 'rates softer, dollar weaker, gold up'>"
  }},
  "big_story": {{
    "story_id": "<one cluster id>",
    "lead_host": "<JAMIE | ALEX | MAYA — pick whose beat>",
    "story_title": "<the canonical title from the cluster>",
    "angle": "<one sentence on what the show should focus on, the genuinely interesting angle>",
    "depth_turns": 6
  }},
  "quick_hits": [
    {{"story_id": "<id>", "lead_host": "<host>", "angle": "<one-line specific take, NOT generic>", "conviction": "real | hype | noise"}},
    {{"story_id": "<id>", "lead_host": "<host>", "angle": "<…>", "conviction": "real | hype | noise"}},
    {{"story_id": "<id>", "lead_host": "<host>", "angle": "<…>", "conviction": "real | hype | noise"}},
    {{"story_id": "<id>", "lead_host": "<host>", "angle": "<…>", "conviction": "real | hype | noise"}}
  ],
  "odd_thing": {{
    "story_id": "<id of an unusual / human / culture-section story>",
    "joke_angle": "<one-line on what's funny or weird about it>"
  }},
  "yesterday_callback": {{
    "use": <true | false — true ONLY if a story below directly continues a YESTERDAY topic listed above and is genuinely worth referencing>,
    "topic": "<the specific yesterday topic to call back, or empty string>",
    "fresh_take": "<one-line on what's NEW today vs. what we said yesterday — must add something, not just restate>"
  }},
  "sign_off": {{
    "callback_target": "<a SPECIFIC company name, joke, or observation that will appear earlier in the show — picked from cold_open / big_story / quick_hits>"
  }}
}}

Rules:
- Every story_id MUST exist in the CLUSTERS list above. Do not invent IDs.
- Cold open + big story + quick hits + odd thing must be ALL DIFFERENT clusters.
- Quick hits = 4 entries (no more, no less).
- Each quick hit's `conviction` is your editorial call: 'real' = signal worth trading on; 'hype' = narrative-driven, may not stick; 'noise' = filler.
- Pick stories that play to each host's beat (ALEX = markets/business/macro, MAYA = tech/culture/odd, JAMIE = host/connector).
- The callback_target must be SPECIFIC (a company name, a numeric quirk, a host's wisecrack potential), not generic.
- The cold_open.hook MUST contain a number or a proper noun + fact. Reject any hook that's just a question.
- If the data is thin, prefer fewer beats with depth over many beats spread thin (3 quick hits is fine if the 4th would be filler).
"""


def plan(ranked: list[dict], market: dict, interests: dict | None = None,
         yesterday_topics: list[str] | None = None) -> dict | None:
    """Produce the beat-by-beat outline. Returns None on failure."""
    if not ranked:
        return None
    tone, _ = _resolve_prefs(interests)
    from generate_script import _TONE_FRAGMENTS
    tone_line = _TONE_FRAGMENTS.get(tone, _TONE_FRAGMENTS["dry"])
    name_list = list(CHARACTERS.keys())
    names_csv = _join_natural(name_list)
    char_lines = _fmt_char_block()
    market_summary = (
        "INDICES:\n" + _fmt_section(market.get("indices") or []) +
        "\n\nGAINERS:\n" + _fmt_section((market.get("gainers") or [])[:5]) +
        "\n\nLOSERS:\n" + _fmt_section((market.get("losers") or [])[:5])
    )
    ranked_block = _fmt_ranked_for_plan(ranked, top_n=14)
    yesterday_block = ""
    if yesterday_topics:
        yt = "\n".join(f"  - {t[:120]}" for t in yesterday_topics[:3])
        yesterday_block = f"\n\nYESTERDAY'S TOP TOPICS (use only if today's news genuinely continues one of these):\n{yt}\n"
    prompt = _PLAN_PROMPT.format(
        names_csv=names_csv, tone_line=tone_line, char_lines=char_lines,
        ranked_block=ranked_block, market_summary=market_summary,
        yesterday_block=yesterday_block,
    )
    try:
        raw = _llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.3)
    except Exception as e:
        print(f"[plan] LLM call failed: {e}")
        return None
    json_str = _strip_json(raw)
    if not json_str:
        print(f"[plan] no JSON in response (first 300 chars): {raw[:300]}")
        return None
    try:
        outline = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[plan] JSON parse failed: {e}")
        return None
    # Sanity-check story_ids exist in ranked
    idx = _ranked_index(ranked)
    bad = []
    for beat in ("cold_open", "big_story", "odd_thing"):
        sid = (outline.get(beat) or {}).get("story_id")
        if sid and sid not in idx:
            bad.append((beat, sid))
    for i, qh in enumerate(outline.get("quick_hits") or []):
        sid = qh.get("story_id")
        if sid and sid not in idx:
            bad.append((f"quick_hits[{i}]", sid))
    if bad:
        print(f"[plan] hallucinated IDs: {bad} — falling back to top-ranked stories")
        # repair: replace bad IDs with top-N unused IDs in order
        used = set()
        for beat in ("cold_open", "big_story", "odd_thing"):
            sid = (outline.get(beat) or {}).get("story_id")
            if sid in idx:
                used.add(sid)
        for qh in outline.get("quick_hits") or []:
            sid = qh.get("story_id")
            if sid in idx:
                used.add(sid)
        spare_ids = [c["id"] for c in ranked if c["id"] not in used]
        for label, _ in bad:
            if not spare_ids:
                break
            new_id = spare_ids.pop(0)
            if "[" in label:
                # quick_hits[i]
                idx_n = int(re.search(r"\[(\d+)\]", label).group(1))
                outline["quick_hits"][idx_n]["story_id"] = new_id
            else:
                outline.setdefault(label, {})["story_id"] = new_id
    return outline


# ─────────── stage 2-7: render each beat ───────────

_BANNED_BLOCK = (
    "Banned cold-open / mid-script phrases (case-insensitive): "
    + ", ".join(f'"{p}"' for p in BANNED_PHRASES)
)


def _count_name_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if re.match(r"^[A-Z][A-Z0-9_]{0,15}:\s*\S", ln))


def _render_beat(
    name: str, instruction: str, prev_turns: list[str],
    turn_target_low: int, turn_target_high: int,
    extra_context: str = "", interests: dict | None = None,
    is_last: bool = False,
) -> str:
    """Generic beat renderer. Each stage calls this with its own instruction."""
    from generate_script import _TONE_FRAGMENTS
    tone, _ = _resolve_prefs(interests)
    tone_line = _TONE_FRAGMENTS.get(tone, _TONE_FRAGMENTS["dry"])
    cast_csv = _join_natural(list(CHARACTERS.keys()))
    char_lines = _fmt_char_block()
    prev_block = _prev_turns_block(prev_turns)
    prompt = f"""You are writing one beat of a daily podcast script. {tone_line}

CAST: {cast_csv}
{char_lines}

PREVIOUS TURNS (context — do NOT repeat them, build on them):
{prev_block}

{extra_context}

YOUR JOB: write the {name} beat ({turn_target_low}-{turn_target_high} turns).
{instruction}

Hard rules for this beat:
- Output ONLY `NAME: line` lines. No headers, no commentary, no markdown.
- NAME must be one of: {", ".join(CHARACTERS.keys())}.
- Every substantive turn includes a SPECIFIC fact (number, name, place) — vague reactions ('that's wild', 'big deal') without a fact are banned.
- Audio tags allowed sparingly, in-line: [deadpan], [laughs], [excited], [sarcastic], [sighs], [mischievously], [rushed], [curious]. Never write disfluencies ('um', 'uh').
- Use COMPANY NAMES not tickers — "Nvidia" not "NVDA", "Broadcom" not "AVGO". Spaced letters only for indices and ETFs (S&P, Nasdaq, VIX).
- Numbers as words: "one point two percent", "four billion dollars".
- No host speaks two consecutive turns. No host says "Right, exactly" / "Of course it is" / "What every X needs is Y".
- Do not write the disclaimer. {"Stop after the last substantive turn — disclaimer is appended in audio." if not is_last else ""}
- {_BANNED_BLOCK}
"""
    out = _normalize_lines(_llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.7))
    actual = _count_name_lines(out)
    # Single retry when a beat comes back thin. Multistage lost the length
    # feedback loop the single-shot path had; without this, render_sign_off
    # silently dropping the disclaimer (or render_quick_hits returning 2/4
    # stories) ships unnoticed.
    if actual < turn_target_low:
        addendum = (
            f"\n\nYour previous attempt produced only {actual} turns. The minimum is "
            f"{turn_target_low} and target is {turn_target_high}. Generate the beat "
            f"again with MORE turns — break monologues, add reactions, ensure every "
            f"specific fact has its own turn. Output ONLY `NAME: line` lines, no headers."
        )
        retry_prompt = prompt + addendum
        retried = _normalize_lines(_llm_call(retry_prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.85))
        if _count_name_lines(retried) > actual:
            out = retried
            print(f"[stage] {name} retry: {_count_name_lines(retried)} turns (was {actual})")
        else:
            print(f"[stage] {name} retry produced no improvement, keeping {actual} turns")
    return out


def _hook_is_specific(hook: str) -> bool:
    """A hook is 'specific' if it has a number, percent, dollar amount, or
    a clear proper noun (not just 'AI' / 'tech' / 'markets')."""
    if not hook:
        return False
    if re.search(r"\d", hook):
        return True
    if re.search(r"\$|percent|%", hook):
        return True
    # Two consecutive Capitalized Words = likely a proper noun
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", hook):
        return True
    return False


def _top_mover_fallback(market: dict) -> str:
    """Deterministic mini-hook from market data when the planner gave us
    something flat. Used as the substance the cold open must reference."""
    movers = (market.get("gainers") or []) + (market.get("losers") or [])
    if not movers:
        return ""
    top = max(movers, key=lambda m: abs(m.get("pct", 0)))
    name = top.get("name") or top.get("symbol") or ""
    pct = top.get("pct", 0.0)
    direction = "up" if pct >= 0 else "down"
    return f"{name} {direction} {abs(pct):.1f} percent"


def render_cold_open(plan_d: dict, interests: dict | None = None, market: dict | None = None) -> str:
    co = plan_d.get("cold_open") or {}
    hook = co.get("hook", "")
    if not _hook_is_specific(hook) and market:
        fallback = _top_mover_fallback(market)
        if fallback:
            print(f"[stage] cold_open hook generic ('{hook[:40]}…'); injecting top-mover fallback")
            hook = f"{hook} ({fallback})" if hook else fallback
    instruction = (
        f'JAMIE delivers a punchy 1-line cold open. Use this hook as the substance: "{hook}". '
        f"Say the name 'Jamie' in the first sentence. No greeting, no welcome, no 'good morning', "
        f"no 'today on the show'. Drop straight into the news with a specific number/name. "
        f"The first turn MUST contain a specific number, dollar amount, or proper noun + fact."
    )
    return _render_beat("COLD OPEN", instruction, [], 1, 2, interests=interests)


def render_markets(plan_d: dict, prev_turns: list[str], market: dict, interests: dict | None = None) -> str:
    m = plan_d.get("markets") or {}
    keys = m.get("key_numbers") or []
    macro = m.get("macro_note") or ""
    market_block = (
        "INDICES:\n" + _fmt_section(market.get("indices") or []) +
        "\n\nGAINERS:\n" + _fmt_section((market.get("gainers") or [])[:5]) +
        "\n\nLOSERS:\n" + _fmt_section((market.get("losers") or [])[:5]) +
        "\n\nMACRO:\n" + _fmt_section(market.get("macro") or [])
    )
    instruction = (
        f"ALEX leads, JAMIE and MAYA each react ONCE. ALEX cites the actual numbers from "
        f"the MARKET DATA below in turn 1. Key numbers from the plan: {keys}. Macro frame: {macro}. "
        f"4-6 turns total. Every number must trace to the data block."
    )
    return _render_beat(
        "MARKETS", instruction, prev_turns, 4, 6,
        extra_context=f"MARKET DATA:\n{market_block}",
        interests=interests,
    )


def render_big_story(plan_d: dict, prev_turns: list[str], ranked_idx: dict[str, dict], interests: dict | None = None) -> str:
    bs = plan_d.get("big_story") or {}
    sid = bs.get("story_id")
    cluster = ranked_idx.get(sid, {})
    title = bs.get("story_title") or cluster.get("title", "")
    angle = bs.get("angle", "")
    lead = bs.get("lead_host", "JAMIE")
    summary = (cluster.get("summary") or "")[:400]
    sources = ", ".join((cluster.get("sources") or [])[:3])
    instruction = (
        f"{lead} leads on this story. The other two hosts push back, react, add color. "
        f'Story: "{title}". Angle to focus on: "{angle}". '
        f"5-7 turns of real back-and-forth — not just one host monologuing. End the beat "
        f"on a note that lands (a punchline or sharp observation), not on a question."
    )
    src_block = f"SOURCE STORY:\n  Title: {title}\n  Sources: {sources}\n  Summary: {summary or '(no summary)'}"
    return _render_beat("BIG STORY", instruction, prev_turns, 5, 7,
                        extra_context=src_block, interests=interests)


def render_quick_hits(plan_d: dict, prev_turns: list[str], ranked_idx: dict[str, dict], interests: dict | None = None) -> str:
    qhs = plan_d.get("quick_hits") or []
    if not qhs:
        return ""
    bullets = []
    for i, q in enumerate(qhs):
        sid = q.get("story_id")
        cluster = ranked_idx.get(sid, {})
        conviction = (q.get("conviction") or "").lower()
        if conviction not in ("real", "hype", "noise"):
            conviction = "real"
        bullets.append(
            f"  {i+1}. lead={q.get('lead_host','?')} conviction={conviction} "
            f"angle=\"{q.get('angle','')}\" "
            f"story=\"{cluster.get('title','')[:120]}\" "
            f"summary=\"{(cluster.get('summary') or '')[:200]}\""
        )
    instruction = (
        f"Cover EXACTLY these {len(qhs)} stories in order, 2-3 turns per story. "
        f"Each story: lead host states the specific fact, ONE other host reacts with a punchline "
        f"or sharp take based on the conviction tag (real = signal, hype = narrative theater, "
        f"noise = filler). The reaction tone should reflect the conviction: 'real' gets a "
        f"serious follow-on, 'hype' gets skepticism or a memed take, 'noise' gets dismissed in "
        f"one beat. Move on quickly. No story bleeds into another. No generic transitions "
        f"between stories — just go.\n\nSTORIES TO COVER:\n" + "\n".join(bullets)
    )
    target_low = max(6, len(qhs) * 2)
    target_high = len(qhs) * 3
    return _render_beat("QUICK HITS", instruction, prev_turns, target_low, target_high,
                        interests=interests)


def render_odd_thing(plan_d: dict, prev_turns: list[str], ranked_idx: dict[str, dict], interests: dict | None = None) -> str:
    ot = plan_d.get("odd_thing") or {}
    sid = ot.get("story_id")
    cluster = ranked_idx.get(sid, {})
    title = cluster.get("title", "")
    summary = (cluster.get("summary") or "")[:400]
    angle = ot.get("joke_angle", "")
    instruction = (
        f"MAYA opens with this odd / unusual / human-interest story. JAMIE and ALEX each react ONCE. "
        f'Story: "{title}". Joke angle: "{angle}". '
        f"3 turns total. End on the joke."
    )
    src_block = f"ODD STORY:\n  Title: {title}\n  Summary: {summary or '(no summary)'}"
    return _render_beat("ODD THING", instruction, prev_turns, 3, 4,
                        extra_context=src_block, interests=interests)


def render_lookahead(civic: dict | None, prev_turns: list[str], interests: dict | None = None) -> str:
    """Beat: 'On the tape tomorrow' — fixed slot for upcoming macro releases
    + earnings using civic intel. Stable named ident so listeners learn to
    wait for it. Skipped silently if civic data is empty."""
    if not civic:
        return ""
    from civic_intel import lookahead_block
    block = lookahead_block(civic)
    if not block.strip():
        return ""
    instruction = (
        "JAMIE introduces 'On the tape tomorrow', then ALEX names 1-2 macro "
        "releases (CPI, jobs, FOMC) and MAYA names 1-2 earnings to watch. "
        "Each item must reference a SPECIFIC date and SPECIFIC company/release "
        "name from the LOOKAHEAD DATA. 3-4 turns total. Punchy, not a list."
    )
    return _render_beat(
        "LOOKAHEAD", instruction, prev_turns, 3, 4,
        extra_context=f"LOOKAHEAD DATA:\n{block}",
        interests=interests,
    )


def render_sign_off(plan_d: dict, prev_turns: list[str], interests: dict | None = None) -> str:
    so = plan_d.get("sign_off") or {}
    callback = so.get("callback_target", "")
    instruction = (
        f'EXACTLY 3 turns: '
        f'(1) ALEX or MAYA opens with a SPECIFIC callback to "{callback}" — repeat the line, '
        f'name the company, build on the joke. Must reference something concretely said in the '
        f'PREVIOUS TURNS above. '
        f'(2) The other host adds a one-line riff. '
        f'(3) JAMIE: "{DISCLAIMER_SHORT}" (verbatim, exactly this line, nothing else). End.'
    )
    return _render_beat("SIGN OFF", instruction, prev_turns, 3, 3,
                        interests=interests, is_last=True)


# ─────────── orchestrator ───────────

def stitch(*beats: str) -> str:
    parts = [b.strip() for b in beats if b and b.strip()]
    return "\n".join(parts)


# Module-level handoff — main.py reads after generate_multistage returns.
_LAST_OUTLINE: dict | None = None


def generate_multistage(
    market: dict,
    ranked: list[dict],
    interests: dict | None = None,
    civic: dict | None = None,
    yesterday_topics: list[str] | None = None,
) -> str:
    """Run the 8-stage pipeline. Returns the full script as NAME: lines."""
    global _LAST_OUTLINE
    print("[stage] plan…")
    outline = plan(ranked, market, interests, yesterday_topics=yesterday_topics)
    _LAST_OUTLINE = outline
    if not outline:
        raise RuntimeError("plan stage failed; cannot proceed multi-stage")
    def _sid6(beat: str) -> str:
        sid = (outline.get(beat) or {}).get("story_id")
        return sid[:6] if isinstance(sid, str) and sid else "?"
    print(f"[stage] plan: cold_open={_sid6('cold_open')} "
          f"big_story={_sid6('big_story')} "
          f"quick_hits={len(outline.get('quick_hits') or [])} "
          f"odd_thing={_sid6('odd_thing')}")
    ranked_idx = _ranked_index(ranked)

    print("[stage] cold_open…")
    co = render_cold_open(outline, interests, market=market)
    prev = [co] if co else []

    yc = outline.get("yesterday_callback") or {}
    if yc.get("use") and yc.get("topic"):
        cb_instr = (
            f"ALEX or MAYA delivers ONE short turn referencing yesterday's "
            f'topic: "{yc.get("topic","")}". Add the fresh take: "{yc.get("fresh_take","")}". '
            f"Must build on yesterday — do NOT just restate. 1 turn only. End with "
            f"a specific new fact, not a question."
        )
        cb = _render_beat("YESTERDAY CALLBACK", cb_instr, prev, 1, 1, interests=interests)
        if cb:
            prev.append(cb)

    print("[stage] markets…")
    mk = render_markets(outline, prev, market, interests)
    prev.append(mk) if mk else None

    print("[stage] big_story…")
    bs = render_big_story(outline, prev, ranked_idx, interests)
    prev.append(bs) if bs else None

    print("[stage] quick_hits…")
    qh = render_quick_hits(outline, prev, ranked_idx, interests)
    prev.append(qh) if qh else None

    print("[stage] odd_thing…")
    ot = render_odd_thing(outline, prev, ranked_idx, interests)
    prev.append(ot) if ot else None

    if civic:
        print("[stage] lookahead…")
        la = render_lookahead(civic, prev, interests)
        if la:
            prev.append(la)

    print("[stage] sign_off…")
    so = render_sign_off(outline, prev, interests)
    # Belt-and-suspenders: if sign_off produced no disclaimer line, append
    # the canonical one directly. sanitize._dedup_disclaimer is the second
    # line of defense, but a totally empty sign_off would otherwise ship
    # without any closer.
    if so and DISCLAIMER_SHORT.lower() not in so.lower():
        so = so.rstrip() + f"\nJAMIE: {DISCLAIMER_SHORT}"
        print("[stage] sign_off: appended canonical disclaimer (model dropped it)")
    elif not so:
        so = f"JAMIE: {DISCLAIMER_SHORT}"
        print("[stage] sign_off: empty, injected fallback disclaimer-only sign-off")
    prev.append(so)

    script = stitch(*prev)
    return script
