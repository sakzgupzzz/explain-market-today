"""Express render: 60-90 second single-narrator headline brief.

Same source data as the show (ranked stories + market summary), tighter prompt,
single voice. The narrator rotates daily through the host cast (see
EXPRESS_NARRATORS + pick_express_narrator) so the express doesn't always sound
like JAMIE. Targets ~200-260 words for ~90 sec audio after speedup.
"""
from __future__ import annotations
import requests
from config import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL,
    DISCLAIMER_SHORT,
)
from generate_script import (
    _fmt_section, _fmt_ranked_stories, _llm_call,
)


_EXPRESS_TARGET_WORDS = 240
_EXPRESS_MAX_STORIES = 8

# Daily narrator rotation. Deterministic by calendar date so a same-day
# re-run (crash retry, --force) picks the same narrator.
EXPRESS_NARRATORS = ["JAMIE", "ALEX", "MAYA"]

# Per-narrator delivery note — the express prompt is otherwise neutral, and
# without this every host reads the brief in the same register.
_NARRATOR_STYLE = {
    "JAMIE": "Warm, confident anchor read. Bright but never breathless.",
    "ALEX": "Dry, measured desk-strategist read. Precise, a hint of deadpan.",
    "MAYA": "Quick, crisp reporter read. Energetic but clipped — no hype.",
}


def pick_express_narrator(date_iso: str) -> str:
    """Rotate through the cast by calendar date (ISO YYYY-MM-DD)."""
    from datetime import date
    return EXPRESS_NARRATORS[date.fromisoformat(date_iso).toordinal() % len(EXPRESS_NARRATORS)]


def build_express_prompt(market: dict, ranked: list[dict], date_str: str,
                         narrator: str = "JAMIE") -> str:
    indices = _fmt_section(market.get("indices", []))
    movers_g = _fmt_section(market.get("gainers", [])[:5])
    movers_l = _fmt_section(market.get("losers", [])[:5])
    stories = _fmt_ranked_stories(ranked, top_n=_EXPRESS_MAX_STORIES)

    return f"""You write a 90-second daily news briefing read by a single host ({narrator}). It is the express version of MARKET TODAY, EXPLAINED — for listeners who want the substance with no banter.

Narrator delivery: {_NARRATOR_STYLE.get(narrator, _NARRATOR_STYLE["JAMIE"])}

Date: {date_str}

==== MARKET DATA ====
INDICES:
{indices}

TOP GAINERS:
{movers_g}

TOP LOSERS:
{movers_l}

==== TOP STORIES (pre-ranked — pick 6-8 of these in priority order) ====
{stories}

==== WRITE THE BRIEFING ====

Format every line as `{narrator}: text` on its own line. Single host, no other names. Aim for {_EXPRESS_TARGET_WORDS} words total across 8-12 turns.

Structure:
1. Cold open: one specific headline, {narrator}'s name, no greeting.
2. Markets: one or two sentences on indices + biggest mover + reason.
3. 4-6 single-sentence story beats — most important first. One concrete fact each (number, name, place).
4. Sign-off line, then disclaimer verbatim: "{DISCLAIMER_SHORT}"

Hard rules:
- Numbers as words: "one hundred billion dollars", "two point three percent", spaced tickers like "N V D A".
- ONLY mention companies and events from the SOURCE DATA above. No fabrication.
- No banter, no jokes, no audio tags, no host introductions beyond the first.
- No banned filler: "welcome", "let's dive in", "buckle up", "stay tuned".
- Each story sentence ≤ 25 words. Front-load the fact. The point first, the color last.
- Do NOT invent quotes. If a story is on the list but you can't say something specific from the title/summary, skip it.

Output ONLY the {narrator}: lines. First line starts with `{narrator}:`. Last line is the disclaimer.
"""


def render_express(market: dict, ranked: list[dict], date_str: str,
                   narrator: str = "JAMIE") -> str:
    """Single LLM call, no critique pass — express is short enough to trust."""
    prompt = build_express_prompt(market, ranked, date_str, narrator=narrator)
    return _llm_call(prompt, OLLAMA_MODEL, GROQ_MODEL, temperature=0.5)


if __name__ == "__main__":
    from datetime import datetime
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines, flatten
    from cluster import cluster_headlines
    from score import score_clusters
    from interests_loader import load_interests
    m = fetch_all()
    h = fetch_headlines()
    clusters = cluster_headlines(flatten(h))
    ranked = score_clusters(clusters, m, load_interests())
    narrator = pick_express_narrator(datetime.now().strftime("%Y-%m-%d"))
    print(render_express(m, ranked, datetime.now().strftime("%A, %B %d, %Y"), narrator=narrator))
