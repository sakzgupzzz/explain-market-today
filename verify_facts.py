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
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL, GROQ_CRITIC_MODEL,
    ANTHROPIC_API_KEY,
    OLLAMA_MODEL, OLLAMA_CRITIC_MODEL, BANNED_PHRASES, DISCLAIMER_SHORT, CHARACTERS,
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


def verify(script: str, market: dict, ranked: list[dict], civic: dict | None = None) -> str:
    if not script.strip():
        return script
    _flag_unscheduled_macro_claims(script, civic)
    import time as _time
    prompt = _verify_prompt(script, market, ranked, civic)
    if GROQ_API_KEY and not ANTHROPIC_API_KEY and len(prompt) > VERIFY_MAX_PROMPT_CHARS:
        print(f"[verify] script too large ({len(prompt)} chars > {VERIFY_MAX_PROMPT_CHARS} cap); skipping verify pass")
        return script
    if GROQ_API_KEY and not ANTHROPIC_API_KEY:
        _time.sleep(8)
    try:
        return _llm_call(prompt, OLLAMA_CRITIC_MODEL, VERIFY_MODEL, temperature=0.1)
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
