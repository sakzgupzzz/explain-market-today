"""Third-pass fact verification.

Takes the post-critique script + ranked source stories. Asks the LLM to
strike or rewrite any turn that introduces a named entity, number, or
direct quote not grounded in the source data.

Cheaper than the critique pass (we only ask for verification, not
restructuring), so we use the smaller llama-3.1-8b-instant model.
Fails open: on any error returns the input unchanged.
"""
from __future__ import annotations
import re
from config import (
    GROQ_API_KEY, ANTHROPIC_API_KEY,
    OLLAMA_CRITIC_MODEL, DISCLAIMER_SHORT, CHARACTERS,
)
from generate_script import (
    _llm_call, _fmt_section, _fmt_ranked_stories,
)

# Use 8b for verification — task is structured (filter unverifiable claims),
# doesn't need 70b reasoning. Cheaper, faster, lower TPM pressure.
VERIFY_MODEL = "llama-3.1-8b-instant"


def _verify_prompt(script: str, market: dict, ranked: list[dict], civic: dict | None = None) -> str:
    indices = _fmt_section(market.get("indices", []))
    movers = _fmt_section((market.get("gainers") or [])[:5] + (market.get("losers") or [])[:5])
    stories = _fmt_ranked_stories(ranked, top_n=12, compact=True)
    name_list = ", ".join(CHARACTERS.keys())
    civic_block = ""
    if civic:
        try:
            from civic_intel import format_for_prompt as _civ_block
            cb = _civ_block(civic)
            if cb:
                civic_block = "\n\nCIVIC INTEL (FRED + EDGAR + Congress, public-domain ground truth):\n" + cb
        except Exception:
            civic_block = ""
    return f"""You are a fact-verification editor. Your only job is to ensure every concrete claim in the SCRIPT below appears in the SOURCE FACTS. You may NOT add new content, restructure beats, or improve writing — only neutralize unverifiable claims.

For each turn:
- If it contains a company name, person's name, dollar amount, percentage, place, date, or direct quote, check that the same fact appears in the SOURCE FACTS.
- If verified, leave the turn unchanged.
- If NOT verified, rewrite the turn to drop the unverifiable detail. Do not invent a replacement. Better a vaguer turn than a fabricated one.
- If the entire turn is unverifiable and cannot be rewritten without losing all substance, drop the turn entirely.
- Always preserve audio tags ([deadpan], [laughs], etc.).
- Always preserve the disclaimer line if present: "{DISCLAIMER_SHORT}".

Output ONLY the verified script in `NAME: line` format with NAME in {name_list}. No commentary.

==== SOURCE FACTS ====
INDICES:
{indices}

MOVERS:
{movers}

TOP STORIES:
{stories}{civic_block}

==== SCRIPT TO VERIFY ====
{script}

==== VERIFIED SCRIPT ====
"""


VERIFY_MAX_PROMPT_CHARS = 7000


_FRED_SERIES_KEYWORDS = {
    "cpi", "consumer price index", "ppi", "producer price index",
    "nfp", "non-farm payroll", "nonfarm payroll", "jobs report",
    "unemployment rate", "fomc", "fed funds rate", "federal funds rate",
    "gdp", "retail sales", "ism manufacturing", "ism services",
}


def _flag_unscheduled_macro_claims(script: str, civic: dict | None) -> None:
    """Soft check: if the script claims a macro print 'today' but civic
    says no such release is scheduled today, ping ntfy for review.
    This is a heuristic — false positives are OK because it's a warning."""
    if not civic:
        return
    macro_today = civic.get("macro_today") or []
    if macro_today:
        return  # something IS scheduled — trust the LLM
    text_l = script.lower()
    today_proximity = any(p in text_l for p in ("today", "this morning", "just released", "just out"))
    if not today_proximity:
        return
    hits = [k for k in _FRED_SERIES_KEYWORDS if k in text_l]
    if not hits:
        return
    try:
        from datetime import datetime as _dt
        from notify import notify_warn
        notify_warn(
            _dt.now().strftime("%Y-%m-%d"),
            "verify_facts.fred",
            f"Script references macro release(s) {hits[:3]} as today's, but FRED calendar shows nothing scheduled.",
        )
        print(f"[verify] flagged unscheduled macro claim(s): {hits[:3]}")
    except Exception:
        pass


# Turn lines look like "JAMIE: ..." — only names from the configured cast count.
_TURN_RE = re.compile(
    rf"^(?:{'|'.join(re.escape(n) for n in CHARACTERS)}):", re.M
)


def _count_turns(text: str) -> int:
    return len(_TURN_RE.findall(text))


def _output_sane(inp: str, out: str) -> bool:
    """Sanity-gate the verifier's output before letting it replace a finished
    script. Rejects empty/truncated/preamble-only responses:
      - output at least ~60% the length of the input
      - a similar count of NAME: turns (verifier may legitimately drop a few)
      - the disclaimer line survives if the input had it
    """
    if not out or not out.strip():
        return False
    if len(out) / max(len(inp), 1) < 0.6:
        return False
    in_turns = _count_turns(inp)
    if in_turns and _count_turns(out) < in_turns * 0.6:
        return False
    if DISCLAIMER_SHORT in inp and DISCLAIMER_SHORT not in out:
        return False
    return True


def _split_at_turn_boundary(script: str) -> tuple[str, str] | None:
    """Split the script into two halves at the NAME: turn start closest to the
    midpoint. Returns None if there's no interior turn boundary to split at."""
    starts = [m.start() for m in _TURN_RE.finditer(script) if m.start() > 0]
    if not starts:
        return None
    mid = len(script) // 2
    cut = min(starts, key=lambda i: abs(i - mid))
    return script[:cut], script[cut:]


def _verify_once(script: str, market: dict, ranked: list[dict], civic: dict | None) -> str:
    """One verification pass over `script`, sanity-checked. Fails open: any
    error, oversized prompt, or insane output returns the input unchanged."""
    import time as _time
    prompt = _verify_prompt(script, market, ranked, civic)
    if GROQ_API_KEY and not ANTHROPIC_API_KEY and len(prompt) > VERIFY_MAX_PROMPT_CHARS:
        print(f"[verify] prompt still too large ({len(prompt)} chars > {VERIFY_MAX_PROMPT_CHARS} cap); skipping this chunk")
        try:
            from datetime import datetime as _dt
            from notify import notify_warn
            notify_warn(
                _dt.now().strftime("%Y-%m-%d"),
                "verify_facts",
                f"script chunk too large to verify ({len(prompt)} chars); shipped unverified",
            )
        except Exception:
            pass
        return script
    if GROQ_API_KEY and not ANTHROPIC_API_KEY:
        _time.sleep(8)
    try:
        out = _llm_call(prompt, OLLAMA_CRITIC_MODEL, VERIFY_MODEL, temperature=0.1)
    except Exception as e:
        print(f"[verify] failed, returning unverified script: {e}")
        try:
            from datetime import datetime as _dt
            from notify import notify_warn
            notify_warn(
                _dt.now().strftime("%Y-%m-%d"),
                "verify_facts",
                f"verify pass failed open: {type(e).__name__}: {e}",
            )
        except Exception:
            pass
        return script
    if not _output_sane(script, out):
        print(f"[verify] output failed sanity checks ({len(out or '')} chars vs {len(script)} input); keeping original")
        return script
    return out


def verify(script: str, market: dict, ranked: list[dict], civic: dict | None = None) -> str:
    if not script.strip():
        return script
    _flag_unscheduled_macro_claims(script, civic)
    prompt = _verify_prompt(script, market, ranked, civic)
    if GROQ_API_KEY and not ANTHROPIC_API_KEY and len(prompt) > VERIFY_MAX_PROMPT_CHARS:
        # Longest scripts used to get zero checking. Split at a turn boundary
        # near the midpoint and verify each half through the same sanity gates.
        halves = _split_at_turn_boundary(script)
        if halves:
            print(f"[verify] prompt {len(prompt)} chars > {VERIFY_MAX_PROMPT_CHARS} cap; verifying in two halves")
            first, second = halves
            v1 = _verify_once(first, market, ranked, civic)
            v2 = _verify_once(second, market, ranked, civic)
            return v1.rstrip("\n") + "\n" + v2.lstrip("\n")
    return _verify_once(script, market, ranked, civic)
