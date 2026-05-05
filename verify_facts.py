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
    OLLAMA_MODEL, OLLAMA_CRITIC_MODEL, BANNED_PHRASES, DISCLAIMER_SHORT, CHARACTERS,
)
from generate_script import (
    _llm_call, _fmt_section, _fmt_ranked_stories,
)

# Use 8b for verification — task is structured (filter unverifiable claims),
# doesn't need 70b reasoning. Cheaper, faster, lower TPM pressure.
VERIFY_MODEL = "llama-3.1-8b-instant"


def _verify_prompt(script: str, market: dict, ranked: list[dict]) -> str:
    indices = _fmt_section(market.get("indices", []))
    movers = _fmt_section((market.get("gainers") or [])[:5] + (market.get("losers") or [])[:5])
    stories = _fmt_ranked_stories(ranked, top_n=20)
    name_list = ", ".join(CHARACTERS.keys())
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
{stories}

==== SCRIPT TO VERIFY ====
{script}

==== VERIFIED SCRIPT ====
"""


def verify(script: str, market: dict, ranked: list[dict]) -> str:
    if not script.strip():
        return script
    prompt = _verify_prompt(script, market, ranked)
    try:
        return _llm_call(prompt, OLLAMA_CRITIC_MODEL, VERIFY_MODEL, temperature=0.1)
    except Exception as e:
        print(f"[verify] failed, returning unverified script: {e}")
        return script
