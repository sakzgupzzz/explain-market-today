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


def write_transcripts(
    script: str,
    mp3_path: Path,
    chunk_timings: list[dict] | None = None,
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
        # SRT
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        srt_lines.append(f"{name}: {clean}")
        srt_lines.append("")
        # VTT (with speaker tag)
        vtt_lines.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
        vtt_lines.append(f"<v {name.title()}>{clean}")
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

def build_index_html() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    eps = sorted(EPISODES_DIR.glob("*.mp3"), reverse=True)
    items = []
    for mp3 in eps:
        meta_txt = mp3.with_suffix(".txt")
        script = meta_txt.read_text() if meta_txt.exists() else ""
        title = _episode_title(script, mp3.stem)
        items.append(f'<li><a href="episodes/{mp3.name}">{title}</a></li>')
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{PODCAST_TITLE}</title>
<style>body{{font-family:system-ui;max-width:680px;margin:4em auto;padding:0 1em;line-height:1.5}}</style>
<h1>{PODCAST_TITLE}</h1>
<p>{PODCAST_DESCRIPTION}</p>
<p><a href="feed.xml">RSS feed</a></p>
<h2>Episodes</h2>
<ul>
{chr(10).join(items)}
</ul>
<hr>
<p style="color:#666;font-size:0.85em">{DISCLAIMER_FULL}</p>
"""
    (DOCS / "index.html").write_text(html)


if __name__ == "__main__":
    build_feed()
    build_index_html()
    print(f"Wrote {FEED_PATH}")
