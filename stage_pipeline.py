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
from datetime import datetime
from typing import Any

from config import (
    CHARACTERS, DISCLAIMER_SHORT, BANNED_PHRASES,
)
from generate_script import (
    _llm_call, _llm_json, _resolve_prefs, _fmt_section, _join_natural,
    _fmt_char_block, OLLAMA_MODEL, GROQ_MODEL,
    OLLAMA_CRITIC_MODEL, GROQ_CRITIC_MODEL,
)
from schemas import (
    ALLOWED_TAGS, build_plan_schema, build_turns_schema, story_signature,
    validate_plan, validate_turns,
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


def _prev_turns_block(turns_so_far: list[str], char_limit: int = 20000) -> str:
    """The script so far, fed to the next beat so it doesn't repeat earlier
    content. The old 6000-char tail-clip dropped the cold_open exactly when
    later beats needed to avoid restating it; Haiku's 200k context easily holds
    the whole script, so keep it (clip only as a runaway guard, and keep the
    HEAD — the cold_open/big_story — which is what gets echoed)."""
    text = "\n".join(turns_so_far)
    if len(text) > char_limit:
        text = text[:char_limit] + "\n…"
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
    "hook": "<≤20 word punchy opener that TEASES the big_story below; MUST include at least one specific number (a percent move, a dollar amount, an index level) OR a specific company/person name with a fact attached. NO 'what's behind X' patterns. Do NOT name a story other than big_story.>"
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
    "use": <true | false — true ONLY if a story below directly continues a YESTERDAY topic listed above AND that yesterday topic is NOT the same story you picked for cold_open / big_story / quick_hits / odd_thing today (a callback that repeats today's lead is filler — set use=false)>,
    "topic": "<the specific yesterday topic to call back, or empty string>",
    "fresh_take": "<one-line on what's NEW today vs. what we said yesterday — must add something, not just restate>"
  }},
  "sign_off": {{
    "callback_target": "<a SPECIFIC company name, joke, or observation that will appear earlier in the show — picked from cold_open / big_story / quick_hits>"
  }}
}}

Rules:
- Every story_id MUST exist in the CLUSTERS list above. Do not invent IDs.
- The cold_open teases big_story (it has no story_id of its own — the hook MUST be about the big_story you pick, so the show always pays off what it teases).
- big_story + quick_hits + odd_thing must be ALL DIFFERENT clusters from each other.
- Quick hits = 4 entries (no more, no less).
- Each quick hit's `conviction` is your editorial call: 'real' = signal worth trading on; 'hype' = narrative-driven, may not stick; 'noise' = filler.
- Pick stories that play to each host's beat (ALEX = markets/business/macro, MAYA = tech/culture/odd, JAMIE = host/connector).
- The callback_target must be SPECIFIC (a company name, a numeric quirk, a host's wisecrack potential), not generic.
- The cold_open.hook MUST contain a number or a proper noun + fact. Reject any hook that's just a question.
- If the data is thin, prefer fewer beats with depth over many beats spread thin (3 quick hits is fine if the 4th would be filler).
"""


def _semantic_dup_violations(outline: dict, idx: dict[str, dict]) -> list[str]:
    """Blinded, heuristic-gated critic for SEMANTIC duplicate beats — two
    DIFFERENT story_ids that are really the same story (e.g. two Nvidia
    clusters that became big_story and a quick_hit; see 2026-05-10 Nvidia
    covered twice). Exact-id uniqueness can't catch this.

    Gate (cheap): only pairs whose title signatures already overlap are even
    considered, so the LLM critic usually doesn't run. Blinded: the critic
    sees ONLY the titles, never the planner's rationale (MARCH-style
    information asymmetry — avoids rubber-stamping the planner's choice).
    """
    beats = _beat_titles(outline, idx)  # [(label, title)]
    suspects: list[tuple[str, str, str, str]] = []
    for i in range(len(beats)):
        for j in range(i + 1, len(beats)):
            la, ta = beats[i]
            lb, tb = beats[j]
            if signatures_overlap(story_signature(ta), story_signature(tb), min_shared=2):
                suspects.append((la, ta, lb, tb))
    if not suspects:
        return []
    listing = "\n".join(
        f"{n+1}. A=[{a_l}] \"{a_t}\"  vs  B=[{b_l}] \"{b_t}\""
        for n, (a_l, a_t, b_l, b_t) in enumerate(suspects)
    )
    critic_prompt = (
        "You are a podcast run-of-show editor. For each numbered pair of story "
        "titles below, answer whether the two titles describe THE SAME underlying "
        "news story (same event/company/subject), which would make covering both "
        "a boring repeat. Judge only the titles shown — no other context.\n\n"
        f"{listing}\n\n"
        "Reply with ONLY the numbers that are the same story, comma-separated "
        "(e.g. '1,3'). If none are the same, reply 'none'."
    )
    try:
        verdict = _llm_call(critic_prompt, OLLAMA_CRITIC_MODEL, GROQ_CRITIC_MODEL,
                            temperature=0.1).strip().lower()
    except Exception as e:
        print(f"[plan] semantic-dup critic skipped ({e})")
        return []
    violations: list[str] = []
    for tok in re.findall(r"\d+", verdict):
        n = int(tok) - 1
        if 0 <= n < len(suspects):
            a_l, a_t, b_l, b_t = suspects[n]
            violations.append(
                f"{a_l} and {b_l} are the same underlying story "
                f"(\"{a_t[:50]}\" / \"{b_t[:50]}\"); replace one with a different story."
            )
    return violations


def _beat_titles(outline: dict, idx: dict[str, dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for label, sid in _beat_story_ids_local(outline).items():
        cl = idx.get(sid)
        if cl and cl.get("title"):
            out.append((label, cl["title"]))
    return out


def _beat_story_ids_local(outline: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for beat in ("big_story", "odd_thing"):
        sid = (outline.get(beat) or {}).get("story_id")
        if sid:
            out[beat] = sid
    for i, qh in enumerate(outline.get("quick_hits") or []):
        sid = qh.get("story_id")
        if sid:
            out[f"quick_hits[{i}]"] = sid
    return out


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
    if market.get("is_stale"):
        market_summary += (
            f"\n\nNOTE: Markets are CLOSED today. The figures above are from "
            f"{market.get('as_of', 'the prior session')}, NOT today. The cold_open "
            f"hook MUST NOT pretend a move happened 'today'. If you use a market "
            f"number, attribute it to the prior session — or pick a non-market "
            f"story for the cold open instead."
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

    idx = _ranked_index(ranked)
    story_ids = [c["id"] for c in ranked[:14] if c.get("id")]
    schema = build_plan_schema(story_ids)
    # Cross-day signatures come from the prior episode's plan sidecar (stable
    # topic TEXT), not cluster ids — ids churn day-over-day, signatures don't.
    yesterday_sigs = [story_signature(t) for t in (yesterday_topics or []) if t]

    def _violations(outline: dict) -> list[str]:
        v = validate_plan(outline, idx, yesterday_sigs)
        # Only run the (LLM) semantic-dup critic once the cheap structural
        # checks pass — no point judging titles on a plan we're rejecting anyway.
        if not v:
            v = _semantic_dup_violations(outline, idx)
        return v

    # Structured path: the decoder is bound to the schema (story_id ∈ enum,
    # hosts ∈ cast, quick_hits == 4), and validate_plan re-prompts on the
    # semantic constraints a schema can't express (cross-beat / cross-day
    # uniqueness, hook substance). No post-hoc id-repair / token-swap / teaser
    # re-pointing — those failure classes are now prevented, not patched.
    try:
        outline = _llm_json(
            prompt, schema, schema_name="episode_plan",
            temperature=0.3, extra_violations=_violations,
        )
        return outline
    except Exception as e:
        print(f"[plan] structured path failed ({e}); trying legacy text parse")

    # Legacy fallback: free-text JSON, best-effort. Only reached when the
    # structured path can't satisfy the contract within its retries.
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
    # Minimal repair only: drop any story_id not in the ranked set so render
    # stages don't KeyError. Everything else is left to the (now-blinded) critic.
    for beat in ("big_story", "odd_thing"):
        b = outline.get(beat) or {}
        if b.get("story_id") and b["story_id"] not in idx and story_ids:
            b["story_id"] = story_ids[0]
    for qh in outline.get("quick_hits") or []:
        if qh.get("story_id") and qh["story_id"] not in idx and story_ids:
            qh["story_id"] = story_ids[0]
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

ALREADY SAID (every fact, number, name, analogy, and joke below is SPENT — the listener just heard it; do NOT restate or paraphrase any of it, build PAST it):
{prev_block}

{extra_context}

YOUR JOB: write the {name} beat ({turn_target_low}-{turn_target_high} turns).
{instruction}

Output format: a JSON object {{"turns": [...]}}. Each turn is an object with:
- "speaker": one of {", ".join(CHARACTERS.keys())} (a field, never written into the text).
- "text": the spoken line. NO inline [bracket] tags — put emotion in the tag field. Never write disfluencies ('um', 'uh'). Speak in the FIRST person — a host never names themselves.
- "tag": OPTIONAL, at most one of {", ".join(ALLOWED_TAGS)} (or omit / ""). Use sparingly.

Content rules:
- Every substantive turn includes a SPECIFIC fact (number, name, place) AND advances past the prior turns — a turn that restates an earlier point in new words is banned. If you can't add a new fact, cut the turn.
- Do NOT reuse an analogy, metaphor, or sentence shape that appears in ALREADY SAID. Never use the frame "X is the [famous brand/event] of [category]" or "[X] equivalent of [Y]" or "corporate speak for [Y]" — these are banned cliché shapes. Invent a fresh comparison from THIS story's details, or just say it plainly.
- Not every turn needs a punchline — a clean factual stop is fine; relentless quipping reads as forced. Avoid leaning on the filler words 'actually', 'basically', 'apparently', 'turns out'.
- Use COMPANY NAMES not tickers — "Nvidia" not "NVDA", "Broadcom" not "AVGO". Spaced letters only for indices and ETFs (S&P, Nasdaq, VIX).
- Numbers as words: "one point two percent", "four billion dollars".
- No host speaks two consecutive turns. No host says "Right, exactly" / "Of course it is" / "What every X needs is Y".
- Do not write the disclaimer. {"Stop after the last substantive turn — disclaimer is appended in audio." if not is_last else ""}
- {_BANNED_BLOCK}
"""
    low, high = turn_target_low, turn_target_high
    schema = build_turns_schema(low, high)
    try:
        payload = _llm_json(
            prompt, schema, schema_name="beat_turns",
            temperature=0.7,
            # Pass the prior beats' text so validate_turns can re-prompt on any
            # turn that recycles a phrase already spoken (cross-beat echo) — the
            # cold_open↔big_story restatement that read as "stuck on one story".
            extra_violations=lambda p: validate_turns(p, prev_text=prev_block),
        )
        return _turns_to_text(payload.get("turns") or [])
    except Exception as e:
        print(f"[stage] {name}: structured render failed ({e}); legacy text path")

    # Legacy fallback: free-text lines + normalize.
    out = _normalize_lines(_llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.7))
    actual = _count_name_lines(out)
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


def _turns_to_text(turns: list[dict]) -> str:
    """Render structured turns to the canonical `NAME: [tag] text` lines the
    rest of the pipeline (sanitize, stitch, TTS) consumes. The tag is emitted
    once, as a prefix; any stray inline brackets in text are dropped (the
    validator already re-prompts on them, this is a final guard)."""
    out = []
    name_set = set(CHARACTERS.keys())
    for t in turns:
        spk = (t.get("speaker") or "").strip()
        if spk not in name_set:
            continue
        text = re.sub(r"\[[^\]]+\]", "", t.get("text") or "").strip()
        if not text:
            continue
        tag = (t.get("tag") or "").strip()
        line = f"{spk}: {tag + ' ' if tag else ''}{text}"
        out.append(line)
    return "\n".join(out)


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


# ── deterministic fallback plan (no LLM) ───────────────────────────────────
# When plan() returns None (every structured + legacy LLM attempt failed) the
# old code raised, dropping the whole run to generate()'s single-shot legacy
# path — which lacks the contract guarantees (enum story_ids, validated turns)
# and re-opens the "weird event" failure classes the restructure closed. This
# builds a contract-valid outline from ranked + market WITHOUT an LLM so the
# normal validated render_* stages run instead. It's a constructive producer,
# not post-hoc repair: the outline satisfies build_plan_schema by construction.

def _fallback_plan(ranked: list[dict], market: dict) -> dict | None:
    """Build a schema-valid outline from ranked stories + market, no LLM.
    Returns None only when there isn't a single usable story (then the caller
    re-raises and generate()'s single-shot path is the last resort)."""
    ids = [c["id"] for c in ranked if c.get("id")]
    if not ids:
        return None
    idx = _ranked_index(ranked)
    hosts = list(CHARACTERS.keys())
    h = lambda i: hosts[i % len(hosts)]

    def _nth(n: int) -> str:
        return ids[n] if n < len(ids) else ids[-1]

    big_id = ids[0]
    big_title = (idx.get(big_id) or {}).get("title") or "today's top story"
    # Hook must carry a number or proper noun (validate_plan rule 3). Prefer a
    # real market mover; fall back to the big-story title (a proper noun).
    mover = _top_mover_fallback(market)
    hook = (f"{mover} — and that's not even the headline." if mover else big_title)
    if len(hook) < 8:
        hook = (big_title + " leads today.")
    hook = hook[:200]

    quick = [
        {"story_id": _nth(1 + i), "lead_host": h(1 + i),
         "angle": ((idx.get(_nth(1 + i)) or {}).get("title") or "")[:120] or "today's news",
         "conviction": "real"}
        for i in range(4)
    ]
    indices = market.get("indices") or []
    key_numbers = [
        f"{r.get('name', r.get('symbol', '?'))} {r.get('pct', 0):+.2f} percent"
        for r in indices[:4]
    ] or ["markets mixed"]

    return {
        "cold_open": {"hook": hook},
        "markets": {
            "lead_host": "ALEX" if "ALEX" in hosts else h(0),
            "key_numbers": key_numbers,
            "macro_note": "Quick read on where the tape closed.",
        },
        "big_story": {
            "story_id": big_id, "lead_host": h(0),
            "story_title": big_title,
            "angle": "Lead with the facts and why it matters.",
            "depth_turns": 6,
        },
        "quick_hits": quick,
        "odd_thing": {"story_id": _nth(5),
                      "joke_angle": "The lighter one to close on."},
        "yesterday_callback": {"use": False, "topic": "", "fresh_take": ""},
        "sign_off": {"callback_target": big_title[:60]},
    }


# ── budget-paced beat sizing ──────────────────────────────────────────────
# eleven_budget.compute_dynamic_preset() paces episode length across the
# remaining monthly ElevenLabs char budget and injects the result into
# interests["preferences"]["_dynamic_preset"]. The single-shot generate()
# path consumes it via _resolve_prefs, but the multistage path renders FIXED
# per-beat turn bands — so it ignored the budget entirely and episodes ran
# full-size every day, exhausting the monthly cap early and tripping the 95%
# guard (took down cron 2026-06-01/02). We scale the content-beat turn bands
# by the paced target here so cumulative char usage stays under the cap.
# No preset (restricted key / local run) → scale 1.0, behavior unchanged.
_NOMINAL_FULL_TURNS = 36   # matches eleven_budget.MAX_TURNS_CEIL (a flush episode)
_MIN_BUDGET_SCALE = 0.5    # never collapse an episode below ~half length


def _budget_scale(interests: dict | None) -> float:
    preset = ((interests or {}).get("preferences") or {}).get("_dynamic_preset")
    if not isinstance(preset, dict):
        return 1.0
    target = preset.get("min_turns")
    if not target:
        return 1.0
    return max(_MIN_BUDGET_SCALE, min(1.0, target / _NOMINAL_FULL_TURNS))


def _scale_band(low: int, high: int, scale: float, floor: int = 1) -> tuple[int, int]:
    """Scale a (low, high) turn band by the budget factor, keeping low ≥ floor
    and high ≥ low so a beat never renders zero turns."""
    lo = max(floor, round(low * scale))
    hi = max(lo, round(high * scale))
    return lo, hi


def render_cold_open(plan_d: dict, interests: dict | None = None, market: dict | None = None) -> str:
    co = plan_d.get("cold_open") or {}
    hook = co.get("hook", "")
    if not _hook_is_specific(hook) and market:
        fallback = _top_mover_fallback(market)
        if fallback:
            print(f"[stage] cold_open hook generic ('{hook[:40]}…'); injecting top-mover fallback")
            hook = f"{hook} ({fallback})" if hook else fallback
    instruction = (
        f'JAMIE delivers a punchy 1-line cold open in FIRST PERSON. Use this hook as the '
        f'substance: "{hook}". JAMIE may identify himself with "I\'m Jamie" or "Jamie here" '
        f"— NEVER refer to Jamie in the third person (no \"Jamie's here to tell you\", no "
        f'"Jamie will explain", no "and Jamie breaks down"). No greeting, no welcome, no '
        f'"good morning", no "today on the show". Drop straight into the news with a '
        f"specific number/name. EXACTLY 1 turn, ≤45 words. MUST contain a specific number, "
        f"dollar amount, or proper noun + fact."
    )
    # Third-person self-reference is now prevented structurally: speaker is a
    # schema field and validate_turns re-prompts on any in-text self-mention.
    return _render_beat("COLD OPEN", instruction, [], 1, 1, interests=interests)


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
    is_stale = bool(market.get("is_stale"))
    as_of = market.get("as_of")
    if is_stale and as_of:
        try:
            pretty = datetime.fromisoformat(as_of).strftime("%A's close")
        except Exception:
            pretty = "the last trading session's close"
        stale_clause = (
            f"\n\nIMPORTANT: markets are CLOSED today (weekend or holiday). The numbers "
            f"above are from {pretty} ({as_of}), NOT today. ALEX MUST frame this beat as "
            f'"{pretty}" — never say "today" for these moves. Open with a single sentence '
            f"acknowledging the weekend/closed-market context, then do the recap of what "
            f"moved last session. JAMIE and MAYA still react once each. 3-4 turns total "
            f"(not 4-6 — the recap is shorter when markets are closed)."
        )
    else:
        stale_clause = ""
    instruction = (
        f"ALEX leads, JAMIE and MAYA each react ONCE. ALEX cites the actual numbers from "
        f"the MARKET DATA below in turn 1. Key numbers from the plan: {keys}. Macro frame: {macro}. "
        f"4-6 turns total. Every number must trace to the data block."
        f"{stale_clause}"
    )
    low, high = (3, 4) if is_stale else (4, 6)
    low, high = _scale_band(low, high, _budget_scale(interests), floor=2)
    return _render_beat(
        "MARKETS", instruction, prev_turns, low, high,
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
    hook = (plan_d.get("cold_open") or {}).get("hook", "")
    instruction = (
        f"{lead} leads on this story. The other two hosts push back, react, add color. "
        f'Story: "{title}". Angle to focus on: "{angle}". '
        f'The cold open ALREADY said this, verbatim: "{hook}". You are FORBIDDEN from '
        f"opening with that fact or any paraphrase of it — the listener just heard the "
        f"headline. Your FIRST turn must start mid-analysis: the mechanism, the why, the "
        f"bear case, or a number that is NOT in the hook. "
        f"5-7 turns of real back-and-forth — not one host monologuing. CRITICAL: each turn "
        f"must add a DISTINCT fact or angle no prior turn has stated (a new number, a "
        f"counterpoint, a second-order consequence, what it means for peers, what to watch "
        f"next). Re-explaining the same point in new words is BANNED — at most ONE "
        f"'what changed' turn, then move on. End on a sharp line or a clean stop, not a question."
    )
    src_block = f"SOURCE STORY:\n  Title: {title}\n  Sources: {sources}\n  Summary: {summary or '(no summary)'}"
    low, high = _scale_band(5, 7, _budget_scale(interests), floor=3)
    return _render_beat("BIG STORY", instruction, prev_turns, low, high,
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
    # Scale by budget, but floor at len(qhs)*2 so every planned story keeps a
    # fact turn + a reaction turn even when tight — flooring at len(qhs) (1 turn
    # each) collapsed quick_hits to flat headlines while big_story stayed full,
    # making the lead story dominate. Quick_hits is where variety lives.
    target_low, target_high = _scale_band(
        target_low, target_high, _budget_scale(interests), floor=len(qhs) * 2
    )
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
    low, high = _scale_band(3, 4, _budget_scale(interests), floor=2)
    return _render_beat("ODD THING", instruction, prev_turns, low, high,
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
    low, high = _scale_band(3, 4, _budget_scale(interests), floor=2)
    return _render_beat(
        "LOOKAHEAD", instruction, prev_turns, low, high,
        extra_context=f"LOOKAHEAD DATA:\n{block}",
        interests=interests,
    )


def _word_count_excl_tags(line: str) -> int:
    """Word count for a turn, excluding the speaker tag and audio tags."""
    body = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:\s*", "", line)
    body = re.sub(r"\[[^\]]+\]", "", body)
    return len(body.split())


def _has_ngram_overlap(line: str, prev_text: str, n: int = 5, threshold: int = 1) -> bool:
    """True when `line` shares an n-gram with `prev_text` — used to detect a
    sign-off turn that's actually re-covering an earlier beat instead of
    riffing on it. n=5 catches verbatim chunks while letting normal English
    fragments slide."""
    def _norm(t: str) -> list[str]:
        t = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:\s*", "", t, flags=re.M)
        t = re.sub(r"\[[^\]]+\]", "", t)
        return re.findall(r"[a-z]+", t.lower())
    a = _norm(line)
    b = _norm(prev_text)
    if len(a) < n or len(b) < n:
        return False
    b_grams = {tuple(b[i:i+n]) for i in range(len(b) - n + 1)}
    hits = sum(1 for i in range(len(a) - n + 1) if tuple(a[i:i+n]) in b_grams)
    return hits >= threshold


def render_sign_off(plan_d: dict, prev_turns: list[str], interests: dict | None = None) -> str:
    so = plan_d.get("sign_off") or {}
    callback = so.get("callback_target", "")
    # Last beat is what listeners just heard — that's what the sign-off must
    # NOT re-cover. Pass it explicitly so the planner can't drift into a
    # restated story (see 2026-05-10: Nvidia covered as quick-hit then
    # re-covered as sign-off; 2026-05-09 same pattern with NRG).
    recent_tail = "\n".join(prev_turns[-2:]) if prev_turns else ""
    instruction = (
        f'EXACTLY 3 turns. This is a TAG, not a recap. The earlier beats already covered '
        f'the subject in depth — do NOT introduce new facts about it, do NOT re-explain it. '
        f'Just land a sharp closer.\n'
        f'(1) ALEX or MAYA: a single SHORT callback to "{callback}" — one fresh metaphor or '
        f'forward-looking jab. ≤25 words. No new statistics, no story summary, no "the thing is". '
        f'Do NOT repeat any punchline, phrase, or analogy already used earlier in the show — '
        f'escalate or twist it instead of restating it.\n'
        f'(2) The other host: a one-line riff that builds on turn 1. ≤20 words. Do NOT '
        f'recycle a punchline. Do NOT add a new statistic.\n'
        f'(3) JAMIE: "{DISCLAIMER_SHORT}" (verbatim, exactly this line, nothing else). End.'
    )
    out = _render_beat("SIGN OFF", instruction, prev_turns, 3, 3,
                       interests=interests, is_last=True)
    # Post-check: catch a sign-off that's smuggling in a story re-cover.
    # If turn 1 or 2 is over the word cap OR shares a 5-gram with the last
    # two beats, retry once with a tighter instruction.
    lines = [ln for ln in out.splitlines() if re.match(r"^[A-Z][A-Z0-9_]{0,15}:", ln)]
    needs_retry = False
    if len(lines) >= 2:
        for idx_l, cap in ((0, 25), (1, 20)):
            if _word_count_excl_tags(lines[idx_l]) > cap:
                needs_retry = True
                break
            if recent_tail and _has_ngram_overlap(lines[idx_l], recent_tail):
                needs_retry = True
                break
    if needs_retry:
        print("[stage] sign_off: turn 1/2 over-runs or echoes prior beat; retrying tighter")
        retry_instr = instruction + (
            "\n\nYour previous attempt re-covered the story or ran long. Tighten: turn 1 "
            "MUST be ≤25 words and contain ZERO new facts, statistics, or re-statements; "
            "it's a wink, not a wrap-up. Turn 2 ≤20 words. Then disclaimer."
        )
        retried = _render_beat("SIGN OFF", retry_instr, prev_turns, 3, 3,
                               interests=interests, is_last=True)
        # Only accept the retry if it's at least no worse on the word caps
        retry_lines = [ln for ln in retried.splitlines()
                       if re.match(r"^[A-Z][A-Z0-9_]{0,15}:", ln)]
        if len(retry_lines) >= 2 and all(
            _word_count_excl_tags(retry_lines[i]) <= cap
            for i, cap in ((0, 25), (1, 20))
        ):
            out = retried
    return out


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
    if not outline:
        # DEGRADE, don't abandon the hardened path: build a contract-valid
        # outline deterministically so the validated render_* stages still run,
        # instead of raising into generate()'s un-validated single-shot legacy.
        print("[stage] plan: LLM plan failed; using deterministic fallback plan")
        outline = _fallback_plan(ranked, market)
        if not outline:
            raise RuntimeError("plan stage failed and no rankable stories; cannot proceed")
    _LAST_OUTLINE = outline
    def _sid6(beat: str) -> str:
        sid = (outline.get(beat) or {}).get("story_id")
        return sid[:6] if isinstance(sid, str) and sid else "?"
    print(f"[stage] plan: cold_open={_sid6('cold_open')} "
          f"big_story={_sid6('big_story')} "
          f"quick_hits={len(outline.get('quick_hits') or [])} "
          f"odd_thing={_sid6('odd_thing')}")
    ranked_idx = _ranked_index(ranked)

    scale = _budget_scale(interests)
    if scale < 1.0:
        print(f"[stage] budget pacing: scaling beat turn bands ×{scale:.2f} "
              f"(tight ElevenLabs char budget → tighter episode)")

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
