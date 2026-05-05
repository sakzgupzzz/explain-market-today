# Markets Explained, Daily

A fully automated, three-host AI podcast on US markets, business, tech, world, and one weird thing. Generated every weekday afternoon and a 90-second express version every weekday morning.

**Stack:** RSS + yfinance → cluster + score → Groq (`llama-3.3-70b`) for dialogue, `llama-3.1-8b-instant` for verification → ElevenLabs v3 Text-to-Dialogue (Sarah / Brian / Jessica voices) → ffmpeg mastering (-16 LUFS, sidechain music bed) → Podcasting 2.0 RSS on GitHub Pages.

**Cost:** $0 LLM (Groq free tier) + $22/mo ElevenLabs Creator+ + $0 hosting (GitHub Pages, Actions).

---

## Pipeline

```
fetch (RSS × 20 sources, yfinance × 60 tickers)
  → cluster + score (interests-aware ranking)
  → memory annotate (skip stories covered in last 2 days)
  → generate (Groq llama-3.3-70b)
  → critique (2nd Groq pass, llama-3.3-70b @ temp 0.2)
  → verify   (3rd Groq pass, llama-3.1-8b-instant)
  → sanitize (regex: banned phrases, ticker spelling, $X / X% normalization)
  → synth (ElevenLabs v3 dialogue API, chunked, mastered, atempo +10%)
  → wrap (intro sting → bed lead-in → JAMIE intro → dialogue → MAYA outro → bed tail → outro sting)
  → sidecars (.srt, .vtt, .chapters.json, ID3 chapters, .meta.json, per-ep cover.jpg)
  → publish (Podcasting 2.0 feed, listen-tracking index page)
  → git push → GitHub Pages
```

Three render formats from the same ranked-story dataset:

| Format | Output | Cost | Use |
|---|---|---|---|
| **Show** | `docs/episodes/YYYY-MM-DD.mp3` | ~5-9k Eleven chars | 3-host roundtable, 5-9 min |
| **Express** | `docs/express/YYYY-MM-DD.mp3` | ~2-3k Eleven chars | Single-narrator 90 sec briefing |
| **Digest** | `docs/digest/YYYY-MM-DD.md` | $0 | Markdown email digest |
| **Thread** | `docs/threads/YYYY-MM-DD.json` | $0 | 5-tweet thread JSON |

---

## Schedule

GitHub Actions cron is best-effort. Three slots stack for resilience, all guarded by an "mp3 already exists" check so they never double-run:

- **12:00 UTC** weekdays — pre-market express briefing
- **21:15 UTC** weekdays — primary show (after US close)
- **22:30 UTC** weekdays — backup show (only fires if 21:15 missed)

Optional fourth trigger via cron-job.org — POST to the workflow_dispatch API.

---

## Setup

```bash
# 1. Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Required secrets — add to GitHub repo Settings → Secrets and variables → Actions
GROQ_API_KEY=gsk_…           # console.groq.com (free tier)
ELEVENLABS_API_KEY=sk_…      # elevenlabs.io (Creator+ tier, 130k chars/mo)

# 3. Optional secrets
NTFY_TOPIC=your-private-topic    # ntfy.sh push notifications
PODCAST_EMAIL=hello@example.com  # neutral inbox shown in RSS

# 4. One-off asset generation (intro sting, music bed, host bookends, etc.)
python make_assets.py

# 5. Manual run
python main.py --both           # generate today's show + express, push
python main.py --show --no-push # show only, local
python main.py --force          # regenerate even if today's mp3 exists
```

`docs/cover.png` (3000×3000) is the base cover art. Per-episode JPGs are auto-generated on top of it with date + lead headline overlay.

---

## Personalization

Edit `interests.yaml` — boosts/dampens story ranking, sets show vibe knobs:

```yaml
watchlist:
  tickers: [AAPL, NVDA, ANTH]
  sectors: [artificial intelligence, semiconductors]
  keywords: [LLM benchmark, FDA approval, Fed policy]
blocked:
  topics: [sports betting, celebrity gossip]
  sources: []
preferences:
  tone: dry         # neutral | dry | snarky
  length: standard  # short | standard | long
```

---

## Modules

```
config.py              — central settings (tickers, RSS feeds, voices, prompts)
fetch_market.py        — yfinance with retry+jitter, returns [] on full failure
fetch_news.py          — parallel RSS fetch + .feed_health.json auto-disable
cluster.py             — Jaccard title-similarity clustering
score.py               — additive importance scoring with keyword cap
state.py               — .state.json memory (with corruption recovery)
calendar_events.py     — earnings (yfinance) + macro (FOMC/CPI/jobs hardcoded)
interests_loader.py    — interests.yaml loader
generate_script.py     — show prompt + critique pass + Groq retry
verify_facts.py        — third-pass fact verification (8b model)
render_express.py      — single-narrator 90-sec render
render_email.py        — markdown digest
render_thread.py       — 5-tweet thread JSON
sanitize.py            — regex post-process (bans, tickers, numbers, JAMIE cap)
tts.py                 — ElevenLabs v3 dialogue, mastering, sidechain bed,
                         host bookends, sting wrap
publish.py             — RSS (P2.0), transcripts, chapters (JSON + ID3),
                         index.html (newsroom layout, listen tracking)
cover_art.py           — Pillow per-episode cover overlay
eleven_usage.py        — char-usage guard (aborts at 95% of monthly cap)
notify.py              — ntfy.sh webhook (success / failure)
lock.py                — file-based concurrency lock
main.py                — orchestrator
make_assets.py         — one-off generation of intro/outro/bed/host bookends
tools/ab_stats.py      — aggregate .meta.json by prompt_variant
tools/audit_audio.py   — local whisper.cpp audit of a published episode
```

55 unit tests (`pytest tests/`).

---

## Distribution submissions

When ready:

1. Validate feed: https://podba.se/validate?url={PODCAST_BASE_URL}/feed.xml
2. Apple Podcasts: https://podcasters.apple.com → submit RSS
3. Spotify: https://podcasters.spotify.com → submit RSS, verify ownership via emailed code
4. YouTube Music: Studio → Content → Podcasts → Connect to RSS feed (free static-image video upload)
5. Amazon Music: https://podcasters.amazon.com/submit-rss
6. Podcast Index: https://podcastindex.org/add (instant, free)

Custom domain (~$30/yr on `.fm`, `.show`, `.news`) ranks better in app search.

---

## Cost dashboard

`docs/index.html` aggregates `.meta.json` sidecars and shows in the page status bar:

- 30-day episode count
- 30-day char usage (track against 130k/mo cap)
- Average duration / turn count
- Last run timestamp
- Format spec
- Source attribution (Groq + ElevenLabs v3)

Listen tracking via `localStorage` shows per-episode listen %, no server.

---

## Disclaimer

Configured in `config.py` (`DISCLAIMER_SHORT` + `DISCLAIMER_FULL`):

> *Markets Explained, Daily is for entertainment and educational purposes only. Nothing in this podcast is investment, financial, legal, or tax advice. Hosts are AI-generated voices, not licensed financial advisors. Always consult a licensed professional before making investment decisions.*

JAMIE reads the short version at the end of every show. Full version appears in feed `<description>` and per-episode show notes.
