"""Build the Podcasting 2.0 RSS feed in docs/, write per-episode transcripts
and chapter JSON, render index.html, and git push to GitHub Pages."""
from __future__ import annotations
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from feedgen.feed import FeedGenerator
from config import (
    DOCS, EPISODES_DIR, FEED_PATH, PODCAST_TITLE, PODCAST_AUTHOR,
    PODCAST_EMAIL, PODCAST_DESCRIPTION, PODCAST_LANGUAGE, PODCAST_BASE_URL,
    PODCAST_CATEGORY, PODCAST_SUBCATEGORY, PODCAST_GUID, ROOT,
    DISCLAIMER_FULL, CHARACTERS, BEATS, BEAT_TITLES, BANNED_PHRASES,
)
from tts import audio_duration_seconds, parse_dialogue

PODCAST_NS = "https://podcastindex.org/namespace/1.0"


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─── Episode title (SEO) ────────────────────────────────────────────────────

_TICKER_INTRO_PATTERNS = [
    re.compile(r"\b(?:Alex|Cam|Maya|Rio|Tess|Dev|Kai|Jamie)\b\s+(?:here|again|on the|from the|checking in)[,.]?\s*", re.I),
    re.compile(r"^\[[^\]]+\]\s*"),  # leading audio tag
]


def _strip_intro(line: str) -> str:
    """Remove leading audio tag + name-intro phrase to get the substantive sentence."""
    s = line.strip()
    for rx in _TICKER_INTRO_PATTERNS:
        s = rx.sub("", s, count=1)
    return s.strip()


_BANNED_LOWER = [p.lower() for p in BANNED_PHRASES]


def _has_banned(s: str) -> bool:
    low = s.lower()
    return any(p in low for p in _BANNED_LOWER)


def _episode_title(script: str, date_str: str) -> str:
    """SEO format: 'MMM D: <lead substantive sentence, ≤60 chars total>'.
    Skips lines containing banned phrases so legacy 'Welcome to your daily'
    leads don't surface as Apple/Spotify titles."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        prefix = d.strftime("%b %-d")
    except (ValueError, TypeError):
        prefix = date_str
    turns = parse_dialogue(script)
    lead = ""
    candidates: list[str] = []
    if turns:
        for _, text in turns:
            candidates.append(_strip_intro(text))
    else:
        # legacy single-narrator script — split on sentences
        candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script.strip())]

    for candidate in candidates:
        first_sentence = candidate.split(".")[0].strip()
        if len(first_sentence.split()) < 6:
            continue
        if _has_banned(first_sentence):
            continue
        lead = first_sentence
        break
    if not lead:
        lead = "Daily markets and tech news"
    full = f"{prefix}: {lead}"
    if len(full) > 60:
        full = full[:57].rsplit(" ", 1)[0] + "…"
    return full


# ─── Transcripts: SRT + VTT ─────────────────────────────────────────────────

def _format_srt_time(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    return _format_srt_time(seconds).replace(",", ".")


def _line_durations(turns: list[tuple[str, str]], total_sec: float) -> list[float]:
    """Distribute total seconds across turns proportional to word count."""
    word_counts = [max(1, len(t.split())) for _, t in turns]
    total_words = sum(word_counts)
    if total_words == 0:
        return [total_sec / max(1, len(turns))] * len(turns)
    return [total_sec * (wc / total_words) for wc in word_counts]


def _strip_audio_tags(text: str) -> str:
    return re.sub(r"\[[^\]]+\]\s*", "", text).strip()


def _attach_citations(text: str, ranked: list[dict] | None) -> str:
    """If a turn references a story title from `ranked`, append the source
    link in a NOTE comment so VTT viewers (and grep) can find it. SRT-safe
    too — the NOTE syntax is harmless to render in plain SRT players."""
    if not ranked:
        return text
    lower = text.lower()
    for c in ranked[:15]:
        title = c.get("title", "")
        link = c.get("link", "")
        if not title or not link:
            continue
        # cheap match — first 2 substantive words of the title in the turn
        toks = [w for w in re.split(r"\W+", title) if len(w) > 3]
        if len(toks) >= 2 and all(t.lower() in lower for t in toks[:2]):
            return f"{text}\n  ↳ source: {link}"
    return text


def write_transcripts(
    script: str,
    mp3_path: Path,
    chunk_timings: list[dict] | None = None,
    ranked_stories: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Write .srt and .vtt next to the mp3. Returns (srt_path, vtt_path)."""
    turns = parse_dialogue(script)
    if not turns:
        return mp3_path.with_suffix(".srt"), mp3_path.with_suffix(".vtt")
    total = audio_duration_seconds(mp3_path)
    durations = _line_durations(turns, total)

    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    cum = 0.0
    for idx, ((name, text), dur) in enumerate(zip(turns, durations), start=1):
        start, end = cum, cum + dur
        cum = end
        clean = _strip_audio_tags(text)
        cited = _attach_citations(clean, ranked_stories)
        # SRT
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        srt_lines.append(f"{name}: {cited}")
        srt_lines.append("")
        # VTT (with speaker tag)
        vtt_lines.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
        vtt_lines.append(f"<v {name.title()}>{cited}")
        vtt_lines.append("")

    srt_path = mp3_path.with_suffix(".srt")
    vtt_path = mp3_path.with_suffix(".vtt")
    srt_path.write_text("\n".join(srt_lines))
    vtt_path.write_text("\n".join(vtt_lines))
    return srt_path, vtt_path


# ─── Chapters: Podcasting 2.0 JSON Chapters ───────────────────────────────

def _detect_beats(turns: list[tuple[str, str]]) -> dict[str, int]:
    """Return turn-index of the first turn assigned to each beat. Heuristic."""
    n = len(turns)
    if n == 0:
        return {}
    boundaries = {"cold_open": 0}

    market_kw = re.compile(
        r"\b(s ?&? ?p|nasdaq|dow|russell|vix|sector|equities|index|gainers|losers|tape|rates|treasury|dollar|fed|macro)\b",
        re.I,
    )
    odd_kw = re.compile(r"\b(odd|weird|kai)\b", re.I)
    signoff_kw = re.compile(r"\b(disclaimer|sign[- ]off|that's it|that's all|see you tomorrow|until next time)\b", re.I)

    # markets: first turn after cold open mentioning market keywords (typically ALEX or CAM)
    for i in range(1, n):
        if market_kw.search(turns[i][1]):
            boundaries["markets"] = i
            break

    # big_story: ~30-40% mark, first long turn
    big_story_target = max(3, int(n * 0.30))
    boundaries.setdefault("big_story", big_story_target)

    # quick_hits: ~60% mark
    boundaries["quick_hits"] = max(boundaries.get("big_story", 0) + 1, int(n * 0.60))

    # odd_thing: first KAI turn after halfway, or odd-keyword match
    half = n // 2
    for i in range(half, n):
        name, text = turns[i]
        if name == "KAI" or odd_kw.search(text):
            boundaries["odd_thing"] = i
            break
    boundaries.setdefault("odd_thing", max(boundaries.get("quick_hits", 0) + 1, int(n * 0.80)))

    # sign_off: turn containing "disclaimer" or last 2 turns
    for i in range(n - 1, -1, -1):
        if signoff_kw.search(turns[i][1]):
            boundaries["sign_off"] = i
            break
    boundaries.setdefault("sign_off", max(0, n - 2))

    # ensure monotonic order
    last = -1
    for beat in BEATS:
        if beat in boundaries:
            boundaries[beat] = max(boundaries[beat], last + 1)
            last = boundaries[beat]
    return boundaries


def write_chapters(
    script: str,
    mp3_path: Path,
    chunk_timings: list[dict] | None = None,
) -> Path:
    """Write Podcasting 2.0 JSON chapters next to the mp3."""
    turns = parse_dialogue(script)
    if not turns:
        out = mp3_path.with_suffix(".chapters.json")
        out.write_text(json.dumps({"version": "1.2.0", "chapters": []}))
        return out
    total = audio_duration_seconds(mp3_path)
    durations = _line_durations(turns, total)
    cum_starts: list[float] = [0.0]
    for d in durations:
        cum_starts.append(cum_starts[-1] + d)

    boundaries = _detect_beats(turns)
    chapters = []
    for beat in BEATS:
        if beat not in boundaries:
            continue
        idx = boundaries[beat]
        if idx >= len(cum_starts):
            continue
        chapters.append({
            "startTime": round(cum_starts[idx], 2),
            "title": BEAT_TITLES[beat],
        })
    out = mp3_path.with_suffix(".chapters.json")
    out.write_text(json.dumps({"version": "1.2.0", "chapters": chapters}, indent=2))
    return out


# ─── RSS feed (Podcasting 2.0 namespace) ───────────────────────────────────

def _make_episode_guid(date_str: str) -> str:
    """Stable per-episode GUID based on PODCAST_GUID + date."""
    import hashlib
    h = hashlib.sha1(f"{PODCAST_GUID}:{date_str}".encode()).hexdigest()[:32]
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _episode_pub(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def build_feed() -> None:
    """Build feed via feedgen, then post-process with lxml to inject Podcasting 2.0 tags."""
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    fg = FeedGenerator()
    fg.load_extension("podcast")
    fg.title(PODCAST_TITLE)
    fg.author({"name": PODCAST_AUTHOR, "email": PODCAST_EMAIL})
    fg.link(href=PODCAST_BASE_URL, rel="alternate")
    fg.link(href=f"{PODCAST_BASE_URL}/feed.xml", rel="self")
    fg.language(PODCAST_LANGUAGE)
    fg.description(PODCAST_DESCRIPTION + "\n\n" + DISCLAIMER_FULL)
    fg.podcast.itunes_author(PODCAST_AUTHOR)
    fg.podcast.itunes_summary(PODCAST_DESCRIPTION)
    fg.podcast.itunes_category(PODCAST_CATEGORY, PODCAST_SUBCATEGORY)
    fg.podcast.itunes_explicit("no")
    fg.podcast.itunes_owner(PODCAST_AUTHOR, PODCAST_EMAIL)
    fg.podcast.itunes_image(f"{PODCAST_BASE_URL}/cover.jpg")
    fg.image(url=f"{PODCAST_BASE_URL}/cover.jpg", title=PODCAST_TITLE, link=PODCAST_BASE_URL)

    # collect all episodes, oldest first for stable ordering
    eps = sorted(EPISODES_DIR.glob("*.mp3"))
    episode_meta: list[dict] = []
    for mp3 in eps:
        meta_txt = mp3.with_suffix(".txt")
        script = meta_txt.read_text() if meta_txt.exists() else ""
        date_str = mp3.stem  # YYYY-MM-DD
        size = mp3.stat().st_size
        try:
            dur = audio_duration_seconds(mp3)
        except Exception:
            dur = 0.0
        title = _episode_title(script, date_str)
        guid = _make_episode_guid(date_str)
        srt = mp3.with_suffix(".srt")
        vtt = mp3.with_suffix(".vtt")
        chapters = mp3.with_suffix(".chapters.json")

        fe = fg.add_entry()
        fe.id(guid)
        fe.title(title)
        # Description: first 600 chars of script + disclaimer
        desc_head = (script[:500] + ("…" if len(script) > 500 else "")) if script else ""
        fe.description(desc_head + "\n\n" + DISCLAIMER_FULL)
        fe.content(script, type="CDATA")
        fe.enclosure(f"{PODCAST_BASE_URL}/episodes/{mp3.name}", str(size), "audio/mpeg")
        fe.published(_episode_pub(date_str))
        fe.podcast.itunes_duration(_mmss(dur))
        fe.podcast.itunes_author(PODCAST_AUTHOR)

        episode_meta.append({
            "guid": guid,
            "date_str": date_str,
            "mp3_name": mp3.name,
            "has_srt": srt.exists(),
            "has_vtt": vtt.exists(),
            "has_chapters": chapters.exists(),
            "season": int(date_str[:4]) if date_str[:4].isdigit() else 1,
            "episode_number": int(date_str.replace("-", "")) if date_str.replace("-", "").isdigit() else 1,
        })

    fg.rss_file(str(FEED_PATH), pretty=True)

    # Post-process: inject Podcasting 2.0 namespace + per-episode P2.0 tags.
    _inject_podcasting_2_tags(FEED_PATH, episode_meta)


def _inject_podcasting_2_tags(feed_path: Path, episode_meta: list[dict]) -> None:
    """Open the feed, register `podcast` namespace, add channel-level + per-item tags.
    feedgen already declares xmlns:podcast (because we use podcast extension), so we
    only register for output and don't re-set the attribute."""
    ET.register_namespace("podcast", PODCAST_NS)
    ET.register_namespace("itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    tree = ET.parse(feed_path)
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        return

    # channel-level: guid, locked, type=episodic, persons (one per host)
    guid_el = ET.SubElement(channel, f"{{{PODCAST_NS}}}guid")
    guid_el.text = PODCAST_GUID
    locked_el = ET.SubElement(channel, f"{{{PODCAST_NS}}}locked")
    locked_el.set("owner", PODCAST_EMAIL)
    locked_el.text = "yes"
    type_el = ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}type")
    type_el.text = "episodic"
    for name in CHARACTERS.keys():
        person = ET.SubElement(channel, f"{{{PODCAST_NS}}}person")
        person.set("role", "host")
        person.text = name.title()

    # per-item tags
    items = list(channel.findall("item"))
    for item, meta in zip(items, reversed(episode_meta)):
        # feedgen lists items newest-first; episode_meta is oldest-first → reverse
        season = ET.SubElement(item, f"{{{PODCAST_NS}}}season")
        season.text = str(meta["season"])
        episode = ET.SubElement(item, f"{{{PODCAST_NS}}}episode")
        episode.text = str(meta["episode_number"])
        if meta.get("has_srt"):
            t = ET.SubElement(item, f"{{{PODCAST_NS}}}transcript")
            t.set("url", f"{PODCAST_BASE_URL}/episodes/{meta['date_str']}.srt")
            t.set("type", "application/x-subrip")
        if meta.get("has_vtt"):
            t = ET.SubElement(item, f"{{{PODCAST_NS}}}transcript")
            t.set("url", f"{PODCAST_BASE_URL}/episodes/{meta['date_str']}.vtt")
            t.set("type", "text/vtt")
        if meta.get("has_chapters"):
            c = ET.SubElement(item, f"{{{PODCAST_NS}}}chapters")
            c.set("url", f"{PODCAST_BASE_URL}/episodes/{meta['date_str']}.chapters.json")
            c.set("type", "application/json+chapters")

    tree.write(feed_path, encoding="utf-8", xml_declaration=True)


# ─── git push ───────────────────────────────────────────────────────────────

def git_push(commit_msg: str) -> None:
    """Commit + push docs/ to origin. Rebase on remote first."""
    if not (ROOT / ".git").exists():
        print("No git repo yet. Run setup.sh first.")
        return
    subprocess.run(["git", "-C", str(ROOT), "add", "docs"], check=True)
    res = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"])
    if res.returncode == 0:
        print("Nothing to commit.")
        return
    subprocess.run(["git", "-C", str(ROOT), "commit", "-m", commit_msg], check=True)
    for attempt in range(3):
        subprocess.run(
            ["git", "-C", str(ROOT), "pull", "--rebase", "--autostash", "origin", "main"],
            check=True,
        )
        push = subprocess.run(["git", "-C", str(ROOT), "push", "origin", "main"])
        if push.returncode == 0:
            return
        print(f"push attempt {attempt + 1} raced. retrying after rebase…")
    raise RuntimeError("git push failed after 3 rebase attempts")


# ─── index.html ─────────────────────────────────────────────────────────────

_HOST_ROLES = {
    "JAMIE": "host",
    "ALEX": "markets",
    "MAYA": "tech",
    "RIO": "world",
    "KAI": "odd-thing",
    "CAM": "macro",
    "TESS": "retail",
    "DEV": "crypto",
}


def _wire_code(date_str: str, idx: int) -> str:
    """Three-letter newsroom slug derived from date + sequence index."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        m = d.strftime("%b").upper()  # JAN, FEB…
        return f"{m[:1]}{idx:02d}"
    except ValueError:
        return f"X{idx:02d}"


def _short_date(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%b %-d, %Y").upper()
    except ValueError:
        return date_str


def _runtime_compact(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _word_count(script: str) -> int:
    if not script:
        return 0
    # rough — strips audio tags + name labels
    cleaned = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:", "", script, flags=re.M)
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    return len(cleaned.split())


def _aggregate_health() -> dict:
    """Walk meta.json sidecars to compute monthly char usage + recent run health."""
    metas = []
    for p in sorted(EPISODES_DIR.glob("*.meta.json")):
        try:
            metas.append(json.loads(p.read_text()))
        except Exception:
            continue
    # last 30 days
    from datetime import datetime as _dt, timedelta as _td
    cutoff = (_dt.utcnow() - _td(days=30)).isoformat()
    recent = [m for m in metas if (m.get("generated_at") or "") >= cutoff]
    char_total = sum(m.get("char_usage_estimate", 0) for m in recent)
    avg_dur = (sum(m.get("duration_sec", 0) for m in recent) / len(recent)) if recent else 0
    avg_turns = (sum(m.get("turns", 0) for m in recent) / len(recent)) if recent else 0
    return {
        "episodes_30d": len(recent),
        "chars_30d": char_total,
        "avg_duration_sec": avg_dur,
        "avg_turns": avg_turns,
        "total_episodes": len(metas),
    }


def build_index_html() -> None:
    """Render docs/index.html as a financial-newspaper × terminal hybrid.
    No bundlers, no JS. Pure static HTML + inline CSS rendered server-side."""
    DOCS.mkdir(parents=True, exist_ok=True)
    eps = sorted(EPISODES_DIR.glob("*.mp3"), reverse=True)
    health = _aggregate_health()

    transmissions: list[str] = []
    ticker_items: list[str] = []
    total = len(eps)

    for i, mp3 in enumerate(eps):
        seq = total - i  # newest = highest number
        meta_txt = mp3.with_suffix(".txt")
        script = meta_txt.read_text() if meta_txt.exists() else ""
        title = _episode_title(script, mp3.stem)
        # strip the "Mmm D: " prefix added by _episode_title for body display
        body_title = re.sub(r"^[A-Z][a-z]{2}\s\d{1,2}:\s*", "", title)
        try:
            dur = audio_duration_seconds(mp3)
        except Exception:
            dur = 0.0
        runtime = _runtime_compact(dur)
        date_str = mp3.stem
        words = _word_count(script)
        wire = _wire_code(date_str, seq)

        transmissions.append(f"""
        <article class="dispatch">
          <header class="dispatch-head">
            <span class="seq">{seq:03d}</span>
            <span class="wire">WIRE / {wire}</span>
            <span class="date">{_short_date(date_str)}</span>
            <span class="runtime" aria-label="runtime">{runtime}</span>
          </header>
          <h2 class="dispatch-title"><a href="episodes/{mp3.name}">{body_title}</a></h2>
          <div class="dispatch-meta">
            <span>{words} words</span>
            <span class="sep">·</span>
            <span><a class="plain" href="episodes/{date_str}.txt">transcript</a></span>
            <span class="sep">·</span>
            <span><a class="plain" href="episodes/{mp3.name}">download</a></span>
          </div>
          <audio class="dispatch-audio" preload="none" controls>
            <source src="episodes/{mp3.name}" type="audio/mpeg">
          </audio>
        </article>""")

        ticker_items.append(
            f'<span class="tk"><b>{wire}</b> '
            f'<span class="tk-arrow">▲</span> '
            f'{runtime} <span class="tk-sep">·</span> '
            f'{words}w <span class="tk-sep">·</span> '
            f'{_short_date(date_str).split(",")[0]}</span>'
        )

    # ticker repeated 3x for seamless loop
    ticker_html = "".join(ticker_items) * 3

    desk_rows = "".join(
        f'<tr><td class="desk-name">{name}</td><td class="desk-role">{_HOST_ROLES.get(name, "")}</td></tr>'
        for name in CHARACTERS.keys()
    )

    issue_no = f"NO. {total:03d}"
    today_str = datetime.now().strftime("%a %-d %b %Y").upper()
    last_update = datetime.now(timezone.utc).strftime("%H:%MZ")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PODCAST_TITLE}</title>
<meta name="description" content="{PODCAST_DESCRIPTION}">
<link rel="alternate" type="application/rss+xml" title="{PODCAST_TITLE}" href="feed.xml">
<style>
:root {{
  --paper: #f4f1e8;
  --paper-deep: #ece6d3;
  --ink: #14110d;
  --rule: #14110d;
  --muted: #635a4d;
  --accent: #9c2b1b;
  --terminal-bg: #0e1015;
  --terminal-fg: #d8c8a3;
  --terminal-dim: #7a6f56;
  --terminal-accent: #d9482a;
  --serif: "Iowan Old Style", "Source Serif 4", "Charter", "Georgia", "Cambria", serif;
  --mono: ui-monospace, "JetBrains Mono", "SF Mono", "Cascadia Code", Menlo, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", sans-serif;
}}

* {{ box-sizing: border-box; }}

html {{
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-feature-settings: "liga", "kern", "tnum";
}}

body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.55;
}}

a {{ color: var(--ink); text-decoration: underline; text-underline-offset: 2px; text-decoration-thickness: 1px; }}
a:hover {{ color: var(--accent); }}
a.plain {{ text-decoration: none; border-bottom: 1px dotted var(--muted); padding-bottom: 1px; }}
a.plain:hover {{ border-bottom-color: var(--accent); }}

/* ── Masthead (newspaper top) ──────────────────────────────── */
.masthead {{
  border-bottom: 4px double var(--rule);
  padding: 32px 40px 18px;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: end;
  gap: 24px;
}}
.brand h1 {{
  margin: 0;
  font-family: var(--serif);
  font-weight: 900;
  font-size: clamp(28px, 5vw, 52px);
  line-height: 0.98;
  letter-spacing: -0.01em;
}}
.brand .tag {{
  font-family: var(--mono);
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.2em;
  color: var(--muted);
  margin-top: 10px;
}}
.issue {{
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: right;
  line-height: 1.7;
}}
.issue b {{ color: var(--ink); font-weight: 700; }}
.issue .price {{ display: inline-block; padding: 2px 8px; border: 1px solid var(--ink); margin-left: 8px; }}

/* ── Ticker tape ───────────────────────────────────────────── */
.ticker {{
  background: var(--terminal-bg);
  color: var(--terminal-fg);
  font-family: var(--mono);
  font-size: 13px;
  letter-spacing: 0.04em;
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  overflow: hidden;
  white-space: nowrap;
  padding: 9px 0;
  position: relative;
}}
.ticker::before, .ticker::after {{
  content: "";
  position: absolute;
  top: 0; bottom: 0;
  width: 60px;
  z-index: 2;
  pointer-events: none;
}}
.ticker::before {{ left: 0; background: linear-gradient(to right, var(--terminal-bg), transparent); }}
.ticker::after  {{ right: 0; background: linear-gradient(to left,  var(--terminal-bg), transparent); }}
.ticker-track {{
  display: inline-block;
  padding-left: 100%;
  animation: tape 60s linear infinite;
}}
.ticker .tk {{ display: inline-block; padding: 0 28px; }}
.ticker .tk b {{ color: var(--terminal-accent); font-weight: 700; }}
.ticker .tk-arrow {{ color: var(--terminal-accent); }}
.ticker .tk-sep {{ color: var(--terminal-dim); padding: 0 6px; }}
@keyframes tape {{
  0%   {{ transform: translate3d(0,0,0); }}
  100% {{ transform: translate3d(-100%,0,0); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .ticker-track {{ animation: none; padding-left: 0; }}
}}

/* ── Lead / blurb ──────────────────────────────────────────── */
.lead {{
  padding: 28px 40px 8px;
  max-width: 760px;
  font-size: 19px;
  line-height: 1.5;
}}
.lead::first-letter {{
  font-weight: 900;
  font-size: 4.2em;
  float: left;
  line-height: 0.85;
  margin: 6px 10px 0 0;
  color: var(--accent);
  font-family: var(--serif);
}}

/* ── Two-column body: desk + transmissions ─────────────────── */
.body {{
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 0;
  border-top: 1px solid var(--rule);
  margin-top: 18px;
}}
@media (max-width: 820px) {{
  .body {{ grid-template-columns: 1fr; }}
  .desk {{ border-right: none !important; border-bottom: 1px solid var(--rule); }}
}}

.desk {{
  padding: 24px 28px;
  border-right: 1px solid var(--rule);
  background: var(--paper-deep);
  position: sticky;
  top: 0;
  align-self: start;
  max-height: 100vh;
  overflow: auto;
}}
.desk h3 {{
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 12px;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 8px;
}}
.desk table {{ border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px; }}
.desk td {{ padding: 4px 0; vertical-align: baseline; }}
.desk-name {{ color: var(--ink); font-weight: 700; letter-spacing: 0.05em; }}
.desk-role {{ color: var(--muted); text-align: right; text-transform: lowercase; }}

.subscribe {{ margin-top: 20px; }}
.subscribe a {{
  display: block;
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 6px 0;
  border-bottom: 1px dotted var(--muted);
  text-decoration: none;
}}
.subscribe a::before {{ content: "→ "; color: var(--accent); }}

/* ── Transmissions ─────────────────────────────────────────── */
.transmissions {{ padding: 0; }}
.dispatch {{
  padding: 26px 40px;
  border-bottom: 1px solid var(--rule);
}}
.dispatch:nth-child(odd) {{ background: var(--paper); }}
.dispatch:nth-child(even) {{ background: var(--paper-deep); }}
.dispatch:hover {{ background: #f9f7ed; }}

.dispatch-head {{
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  gap: 14px;
  align-items: baseline;
  padding-bottom: 8px;
  border-bottom: 1px dotted var(--muted);
}}
.dispatch-head .seq {{
  background: var(--ink);
  color: var(--paper);
  padding: 3px 8px;
  font-weight: 700;
  letter-spacing: 0.08em;
}}
.dispatch-head .wire {{ color: var(--accent); font-weight: 700; }}
.dispatch-head .runtime {{ color: var(--ink); font-weight: 700; }}

.dispatch-title {{
  font-family: var(--serif);
  font-weight: 800;
  font-size: clamp(20px, 2.2vw, 28px);
  line-height: 1.2;
  margin: 14px 0 8px;
  letter-spacing: -0.01em;
}}
.dispatch-title a {{ text-decoration: none; }}
.dispatch-title a:hover {{ color: var(--accent); text-decoration: underline; text-decoration-thickness: 2px; }}

.dispatch-meta {{
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
  letter-spacing: 0.04em;
  margin-bottom: 14px;
}}
.dispatch-meta .sep {{ padding: 0 8px; color: var(--muted); }}

.dispatch-audio {{
  width: 100%;
  height: 36px;
  filter: grayscale(0.6) contrast(0.95);
}}

/* ── Status bar ────────────────────────────────────────────── */
.statusbar {{
  background: var(--terminal-bg);
  color: var(--terminal-fg);
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  padding: 12px 40px;
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
  border-top: 1px solid var(--rule);
}}
.statusbar .pulse {{
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--terminal-accent);
  margin-right: 8px;
  vertical-align: middle;
  animation: pulse 1.6s ease-in-out infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50%      {{ opacity: 0.25; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .pulse {{ animation: none; }}
}}
.statusbar .seg {{ color: var(--terminal-fg); }}
.statusbar .seg b {{ color: var(--terminal-accent); font-weight: 700; }}
.statusbar .seg .dim {{ color: var(--terminal-dim); }}

/* ── Disclaimer footer ─────────────────────────────────────── */
.disclaimer {{
  padding: 24px 40px 60px;
  font-size: 12px;
  line-height: 1.55;
  color: var(--muted);
  font-family: var(--sans);
  max-width: 820px;
  border-top: 1px solid var(--rule);
}}
.disclaimer h4 {{
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--ink);
  margin: 0 0 8px;
}}

/* responsive padding */
@media (max-width: 600px) {{
  .masthead, .lead, .dispatch, .statusbar, .disclaimer {{ padding-left: 20px; padding-right: 20px; }}
  .masthead {{ grid-template-columns: 1fr; gap: 14px; }}
  .issue {{ text-align: left; }}
}}
</style>
</head>
<body>

<header class="masthead">
  <div class="brand">
    <h1>Market Today, Explained</h1>
    <div class="tag">Daily · Markets · Tech · World · Culture</div>
  </div>
  <div class="issue">
    <b>{issue_no}</b><br>
    {today_str}<br>
    <span class="price">FREE</span>
  </div>
</header>

<div class="ticker" aria-label="recent transmissions">
  <div class="ticker-track">{ticker_html}</div>
</div>

<section class="lead">
  Eight AI-generated hosts riff on US markets, business, tech, world, and one weird thing. Five to ten minutes, every weekday afternoon, mastered to broadcast loudness. Underneath the jokes: a strict grounding pipeline that refuses to invent a story it can't cite.
</section>

<div class="body">
  <aside class="desk">
    <h3>The Desk</h3>
    <table>{desk_rows}</table>
    <div class="subscribe">
      <h3 style="margin-top:24px">Subscribe</h3>
      <a href="feed.xml">RSS feed</a>
      <a href="https://podcasts.apple.com/" rel="nofollow">Apple Podcasts</a>
      <a href="https://open.spotify.com/" rel="nofollow">Spotify</a>
    </div>
  </aside>

  <main class="transmissions">
    {''.join(transmissions)}
  </main>
</div>

<div class="statusbar">
  <span class="seg"><span class="pulse"></span>FEED ACTIVE</span>
  <span class="seg"><span class="dim">EPISODES</span> <b>{total:03d}</b></span>
  <span class="seg"><span class="dim">30D EPISODES</span> <b>{health['episodes_30d']}</b></span>
  <span class="seg"><span class="dim">30D CHARS</span> <b>{health['chars_30d']:,}</b></span>
  <span class="seg"><span class="dim">AVG DUR</span> <b>{int(health['avg_duration_sec']//60):02d}:{int(health['avg_duration_sec']%60):02d}</b></span>
  <span class="seg"><span class="dim">AVG TURNS</span> <b>{int(health['avg_turns'])}</b></span>
  <span class="seg"><span class="dim">LAST UPDATE</span> <b>{last_update}</b></span>
  <span class="seg"><span class="dim">FORMAT</span> <b>MP3 / 44.1KHZ / -16 LUFS</b></span>
  <span class="seg"><span class="dim">SOURCE</span> <b>GROQ + ELEVENLABS V3</b></span>
</div>

<footer class="disclaimer">
  <h4>Disclaimer</h4>
  <p>{DISCLAIMER_FULL}</p>
</footer>

</body>
</html>
"""
    (DOCS / "index.html").write_text(html)


if __name__ == "__main__":
    build_feed()
    build_index_html()
    print(f"Wrote {FEED_PATH}")
