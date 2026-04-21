"""Build RSS feed in docs/ and git push to GitHub Pages."""
from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from feedgen.feed import FeedGenerator
from config import (
    DOCS, EPISODES_DIR, FEED_PATH, PODCAST_TITLE, PODCAST_AUTHOR,
    PODCAST_EMAIL, PODCAST_DESCRIPTION, PODCAST_LANGUAGE, PODCAST_BASE_URL,
    PODCAST_CATEGORY, PODCAST_SUBCATEGORY, ROOT,
)
from tts import audio_duration_seconds


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_feed() -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    fg = FeedGenerator()
    fg.load_extension("podcast")
    fg.title(PODCAST_TITLE)
    fg.author({"name": PODCAST_AUTHOR, "email": PODCAST_EMAIL})
    fg.link(href=PODCAST_BASE_URL, rel="alternate")
    fg.link(href=f"{PODCAST_BASE_URL}/feed.xml", rel="self")
    fg.language(PODCAST_LANGUAGE)
    fg.description(PODCAST_DESCRIPTION)
    fg.podcast.itunes_author(PODCAST_AUTHOR)
    fg.podcast.itunes_summary(PODCAST_DESCRIPTION)
    fg.podcast.itunes_category(PODCAST_CATEGORY, PODCAST_SUBCATEGORY)
    fg.podcast.itunes_explicit("no")
    fg.podcast.itunes_owner(PODCAST_AUTHOR, PODCAST_EMAIL)
    fg.podcast.itunes_image(f"{PODCAST_BASE_URL}/cover.jpg")
    fg.image(url=f"{PODCAST_BASE_URL}/cover.jpg", title=PODCAST_TITLE, link=PODCAST_BASE_URL)

    # collect all episodes, oldest first for stable GUIDs, but feed lists newest first
    eps = sorted(EPISODES_DIR.glob("*.mp3"))
    for mp3 in eps:
        meta = mp3.with_suffix(".txt")
        script = meta.read_text() if meta.exists() else ""
        date_str = mp3.stem  # YYYY-MM-DD
        try:
            pub = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=21, tzinfo=timezone.utc
            )
        except ValueError:
            pub = datetime.now(timezone.utc)
        size = mp3.stat().st_size
        dur = audio_duration_seconds(mp3)

        fe = fg.add_entry()
        fe.id(f"{PODCAST_BASE_URL}/episodes/{mp3.name}")
        fe.title(f"{PODCAST_TITLE} — {date_str}")
        fe.description(script[:600] + ("…" if len(script) > 600 else ""))
        fe.content(script, type="CDATA")
        fe.enclosure(
            f"{PODCAST_BASE_URL}/episodes/{mp3.name}",
            str(size),
            "audio/mpeg",
        )
        fe.published(pub)
        fe.podcast.itunes_duration(_mmss(dur))
        fe.podcast.itunes_author(PODCAST_AUTHOR)

    fg.rss_file(str(FEED_PATH), pretty=True)


def git_push(commit_msg: str) -> None:
    """Commit + push docs/ to origin. Rebase on remote first to survive concurrent pushes.
    Retries once if push races another writer."""
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


def build_index_html() -> None:
    """Minimal landing page with feed link."""
    DOCS.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{PODCAST_TITLE}</title>
<style>body{{font-family:system-ui;max-width:640px;margin:4em auto;padding:0 1em;line-height:1.5}}</style>
<h1>{PODCAST_TITLE}</h1>
<p>{PODCAST_DESCRIPTION}</p>
<p><a href="feed.xml">RSS feed</a></p>
<h2>Episodes</h2>
<ul>
"""
    for mp3 in sorted(EPISODES_DIR.glob("*.mp3"), reverse=True):
        html += f'<li><a href="episodes/{mp3.name}">{mp3.stem}</a></li>\n'
    html += "</ul>\n"
    (DOCS / "index.html").write_text(html)


if __name__ == "__main__":
    build_feed()
    build_index_html()
    print(f"Wrote {FEED_PATH}")
