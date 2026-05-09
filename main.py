"""Orchestrate the daily pipeline.

Stages:
  1. Ingest    — fetch market data + headlines
  2. Rank      — cluster, score, annotate against memory state
  3. Render    — generate show + express scripts in parallel
  4. Sanitize  — deterministic post-process
  5. Synth     — ElevenLabs v3 dialogue + mastering + stings
  6. Sidecars  — transcripts, chapters, episode metadata
  7. Publish   — RSS feed + index regen
  8. Memory    — record covered story IDs
  9. Push      — git commit + push
"""
from __future__ import annotations
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import EPISODES_DIR, ROOT
from fetch_market import fetch_all
from fetch_news import fetch_headlines, flatten
from cluster import cluster_headlines
from score import score_clusters
from state import load_state, save_state, mark_covered, annotate_clusters
from interests_loader import load_interests
from calendar_events import gather as gather_calendar_events
from eleven_budget import compute_dynamic_preset, format_log_line as fmt_budget_log
from generate_script import generate, critique_revise
from verify_facts import verify as verify_facts
from render_express import render_express
from render_email import write_digest
from render_thread import write_thread
from sanitize import sanitize_script
from tts import synth, audio_duration_seconds
from publish import build_feed, build_index_html, git_push, write_transcripts, write_chapters
from cover_art import write_episode_cover
from lock import acquire_lock
from eleven_usage import check_budget
from notify import notify_success, notify_failure

EXPRESS_DIR = Path(EPISODES_DIR).parent / "express"


def _detect_covered_clusters(script: str, ranked: list[dict], top_n: int = 12) -> list[str]:
    """Heuristic: a story is 'covered' if 2+ tokens of its title appear in the
    script. Restricts the search to the top N candidates so we don't false-
    positive-match arbitrary words."""
    script_l = script.lower()
    out: list[str] = []
    for c in ranked[:top_n]:
        toks = [w for w in re.split(r"\W+", c.get("title", "").lower()) if len(w) > 3]
        if not toks:
            continue
        hits = sum(1 for t in toks if t in script_l)
        if hits >= 2:
            out.append(c["id"])
    return out


def _word_count(text: str) -> int:
    cleaned = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:", "", text, flags=re.M)
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    return len(cleaned.split())


def _turn_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^[A-Z][A-Z0-9_]{0,15}:\s*\S", line))


def _write_meta(mp3_path: Path, script: str, char_usage: int | None = None) -> None:
    """Sidecar episode metadata for analytics + cost dashboard."""
    try:
        dur = audio_duration_seconds(mp3_path)
    except Exception:
        dur = 0.0
    try:
        from generate_script import PROMPT_VERSION, PROMPT_VARIANT
    except Exception:
        PROMPT_VERSION, PROMPT_VARIANT = "?", "?"
    meta = {
        "date": mp3_path.stem,
        "mp3": mp3_path.name,
        "size_bytes": mp3_path.stat().st_size if mp3_path.exists() else 0,
        "duration_sec": round(dur, 2),
        "turns": _turn_count(script),
        "words": _word_count(script),
        "char_usage_estimate": char_usage if char_usage is not None else len(script),
        "prompt_version": PROMPT_VERSION,
        "prompt_variant": PROMPT_VARIANT,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    mp3_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def run(push: bool = True, force: bool = False, mode: str = "show") -> Path:
    """mode = 'show' | 'express' | 'both' (default 'both' triggers both renders)."""
    today = datetime.now().strftime("%Y-%m-%d")
    date_pretty = datetime.now().strftime("%A, %B %d, %Y")
    mp3_path = EPISODES_DIR / f"{today}.mp3"

    if mp3_path.exists() and not force and mode != "express":
        print(f"[{today}] show episode already published at {mp3_path} — skipping (use --force to regenerate)")
        return mp3_path

    # ── ElevenLabs char-usage guard ────────────────────────────────────────
    ok, msg = check_budget(threshold=0.95)
    print(msg)
    if not ok:
        notify_failure(today, "budget_check", msg)
        raise RuntimeError(msg)

    print(f"[{today}] fetching market data…")
    market = fetch_all()
    print(f"[{today}] fetching headlines…")
    headlines_by_cat = fetch_headlines()
    flat = flatten(headlines_by_cat)
    print(f"[{today}] {len(flat)} headlines across {len(headlines_by_cat)} beats")

    print(f"[{today}] clustering + ranking…")
    clusters = cluster_headlines(flat)
    interests = load_interests()
    ranked = score_clusters(clusters, market, interests)
    state = load_state()
    annotated = annotate_clusters(ranked, state, suppress_days=2)
    fresh = [c for c in annotated if not c.get("seen_recently")]
    follow_ups = [c for c in annotated if c.get("seen_recently")][:5]
    print(f"[{today}] {len(clusters)} clusters → {len(fresh)} fresh, {len(follow_ups)} follow-ups")

    # ── upcoming-events context (earnings + macro calendar) ────────────────
    watchlist = (interests.get("watchlist") or {}).get("tickers") or []
    upcoming_events = gather_calendar_events(watchlist)
    if upcoming_events:
        print(f"[{today}] injected upcoming-events block ({len(upcoming_events.splitlines())} lines)")

    # ── dynamic length sizing from ElevenLabs char budget ──────────────────
    # Stretches remaining month-budget evenly across remaining weekday runs
    # so episode length scales with available headroom. Falls back to
    # interests.yaml preferences.length when budget data is unavailable.
    budget_preset = compute_dynamic_preset()
    if budget_preset:
        print(fmt_budget_log(budget_preset))
        interests.setdefault("preferences", {})["_dynamic_preset"] = budget_preset

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    # ── show render ────────────────────────────────────────────────────────
    if mode in ("show", "both"):
        if mp3_path.exists() and not force:
            print(f"[{today}] show already published — skipping show render")
        else:
            print(f"[{today}] generating show script…")
            script = generate(
                market, fresh, date_pretty,
                follow_ups=follow_ups, upcoming_events=upcoming_events,
                interests=interests,
            )
            print(f"[{today}] critique pass…")
            script = critique_revise(script, market, fresh)
            print(f"[{today}] fact verification pass…")
            script = verify_facts(script, market, fresh)
            print(f"[{today}] sanitizing…")
            script = sanitize_script(script)
            txt_path = EPISODES_DIR / f"{today}.txt"
            txt_path.write_text(script)
            print(f"[{today}] synthesizing show audio…")
            synth_result = synth(script, mp3_path)
            chunk_timings = synth_result[1] if isinstance(synth_result, tuple) else None
            print(f"[{today}] writing transcripts + chapters…")
            write_transcripts(script, mp3_path, chunk_timings, ranked_stories=fresh)
            write_chapters(script, mp3_path, chunk_timings)
            lead_title = (fresh[0].get("title") if fresh else "Daily roundup")
            cover = write_episode_cover(today, lead_title)
            if cover:
                print(f"[{today}] wrote per-episode cover → {cover.name}")
            _write_meta(mp3_path, script)

    # ── express render ─────────────────────────────────────────────────────
    # Wrapped in try/except so an express failure can't kill the show
    # publish — show is the load-bearing artifact; express is a bonus.
    if mode in ("express", "both"):
        EXPRESS_DIR.mkdir(parents=True, exist_ok=True)
        ex_mp3 = EXPRESS_DIR / f"{today}.mp3"
        ex_txt = EXPRESS_DIR / f"{today}.txt"
        if ex_mp3.exists() and not force:
            print(f"[{today}] express already published — skipping express render")
        else:
            try:
                print(f"[{today}] generating express script…")
                ex_script = render_express(market, fresh, date_pretty)
                ex_script = sanitize_script(ex_script, verbose=False)
                # Guard: don't synth a script that's just the disclaimer.
                # parse_dialogue would return ≤1 turn → synth'd to silence.
                substantive_lines = [
                    l for l in ex_script.splitlines()
                    if re.match(r"^[A-Z][A-Z0-9_]{0,15}:\s*\S", l)
                    and "entertainment and education only" not in l.lower()
                ]
                if len(substantive_lines) < 3:
                    print(f"[{today}] express script too thin ({len(substantive_lines)} substantive turns); skipping express")
                else:
                    ex_txt.write_text(ex_script)
                    print(f"[{today}] synthesizing express audio…")
                    synth(ex_script, ex_mp3)
                    _write_meta(ex_mp3, ex_script)
            except Exception as e:
                import traceback
                print(f"[{today}] express render failed (non-fatal): {type(e).__name__}: {e}")
                traceback.print_exc()

    # ── memory ─────────────────────────────────────────────────────────────
    if mode in ("show", "both"):
        try:
            script_now = (EPISODES_DIR / f"{today}.txt").read_text()
            covered_ids = _detect_covered_clusters(script_now, fresh)
            mark_covered(state, covered_ids)
            save_state(state)
            print(f"[{today}] memory: marked {len(covered_ids)} cluster(s) as covered")
        except Exception as e:
            print(f"[memory] couldn't update state: {e}")

    # ── deterministic renders (no LLM cost) ────────────────────────────────
    try:
        digest_path = write_digest(market, fresh, today)
        thread_path = write_thread(market, fresh, today)
        print(f"[{today}] wrote digest → {digest_path.name}, thread → {thread_path.name}")
    except Exception as e:
        print(f"[render] digest/thread failed: {e}")

    print(f"[{today}] building feed…")
    build_feed()
    build_index_html()

    if push:
        print(f"[{today}] pushing…")
        git_push(f"episode {today}")
    print(f"[{today}] done → {mp3_path}")

    # ── notify ─────────────────────────────────────────────────────────────
    if mode in ("show", "both") and mp3_path.exists():
        try:
            dur = audio_duration_seconds(mp3_path)
            words = _word_count((EPISODES_DIR / f"{today}.txt").read_text())
            turns = _turn_count((EPISODES_DIR / f"{today}.txt").read_text())
            notify_success(today, mode, turns, words, dur)
        except Exception:
            pass
    return mp3_path


if __name__ == "__main__":
    push = "--no-push" not in sys.argv
    force = "--force" in sys.argv
    mode = "both"
    for flag, name in [("--show", "show"), ("--express", "express"), ("--both", "both")]:
        if flag in sys.argv:
            mode = name
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with acquire_lock():
            run(push=push, force=force, mode=mode)
    except Exception as e:
        traceback.print_exc()
        notify_failure(today_str, "main", f"{type(e).__name__}: {e}")
        sys.exit(1)
