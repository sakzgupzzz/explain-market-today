"""Email digest renderer — markdown, no LLM call.

Renders a daily digest from the same ranked stories the show + express use.
Output: docs/digest/YYYY-MM-DD.md. Useful for inbox subscribers, Substack
mirroring, or just a copy-paste source for newsletters.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from config import DOCS, PODCAST_TITLE, PODCAST_BASE_URL, DISCLAIMER_FULL


DIGEST_DIR = DOCS / "digest"


def _fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"


def render_email_digest(
    market: dict,
    ranked: list[dict],
    date_str: str,
    top_n: int = 10,
) -> str:
    """Return the markdown body. Caller decides where to write it."""
    lines = []
    lines.append(f"# {PODCAST_TITLE} — {date_str}")
    lines.append("")
    audio_url = f"{PODCAST_BASE_URL}/episodes/{date_str}.mp3"
    express_url = f"{PODCAST_BASE_URL}/express/{date_str}.mp3"
    lines.append(f"**Listen:** [Show audio]({audio_url}) · [60-second express]({express_url})")
    lines.append("")

    # Markets snapshot
    indices = market.get("indices") or []
    if indices:
        lines.append("## Markets")
        for r in indices[:5]:
            lines.append(f"- **{r['name']}** ({r['symbol']}): {r['close']:.2f} ({_fmt_pct(r['pct'])})")
        lines.append("")

    movers = (market.get("gainers") or [])[:3] + (market.get("losers") or [])[:3]
    if movers:
        lines.append("### Movers")
        for r in movers:
            lines.append(f"- **{r['symbol']}**: {r['close']:.2f} ({_fmt_pct(r['pct'])})")
        lines.append("")

    if ranked:
        lines.append("## Top stories")
        for c in ranked[:top_n]:
            title = c.get("title", "")
            sources = ", ".join((c.get("sources") or [])[:3])
            link = c.get("link") or ""
            line = f"- **{title}**"
            if link:
                line = f"- [{title}]({link})"
            line += f" — _{sources}_"
            lines.append(line)
            summary = c.get("summary") or ""
            if summary:
                lines.append(f"  {summary[:280]}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_{DISCLAIMER_FULL}_")
    lines.append("")
    return "\n".join(lines)


def write_digest(market: dict, ranked: list[dict], date_str: str) -> Path:
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    body = render_email_digest(market, ranked, date_str)
    out = DIGEST_DIR / f"{date_str}.md"
    out.write_text(body)
    return out


if __name__ == "__main__":
    from fetch_market import fetch_all
    from fetch_news import fetch_headlines, flatten
    from cluster import cluster_headlines
    from score import score_clusters
    from interests_loader import load_interests
    today = datetime.now().strftime("%Y-%m-%d")
    m = fetch_all()
    h = fetch_headlines()
    ranked = score_clusters(cluster_headlines(flatten(h)), m, load_interests())
    print(render_email_digest(m, ranked, today))
