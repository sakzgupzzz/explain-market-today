# Market Today, Explained — Personal Daily Podcast

Free, local, fully automated US market recap podcast.

**Stack**: `yfinance` + free RSS → Ollama (`qwen2.5:14b`) → macOS `say` + `ffmpeg` → GitHub Pages RSS → Spotify / Apple Podcasts.

## One-time setup

```bash
cd /Users/sakshamgupta/Documents/coding_projects/explain_market_today_project
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:
1. Create `.venv`, install Python deps.
2. Verify `qwen2.5:14b` is pulled in Ollama.
3. Init git, create public GitHub repo `explain-market-today`, push.
4. Enable GitHub Pages on `main` /docs.

Note the printed **Pages URL**. If it differs from `PODCAST_BASE_URL` in `config.py`, edit it.

Add a cover image at `docs/cover.jpg` (1400x1400 JPG/PNG, required by Apple/Spotify).

## Manual run

```bash
source .venv/bin/activate
python main.py             # fetch, synth, feed, push
python main.py --no-push   # local only
```

Output: `docs/episodes/YYYY-MM-DD.mp3` + `docs/feed.xml`.

## Schedule daily

```bash
cp com.user.marketpodcast.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.marketpodcast.plist
```

Default: weekdays 4:45pm local. Edit `Hour`/`Minute`/`Weekday` in the plist.

Unload: `launchctl unload ~/Library/LaunchAgents/com.user.marketpodcast.plist`.

## Submit feed to podcast platforms

After first real episode is live:

- **Spotify**: <https://podcasters.spotify.com/submit> → paste `<PODCAST_BASE_URL>/feed.xml`.
- **Apple Podcasts**: <https://podcastsconnect.apple.com> → add show → paste RSS URL.

Both poll the feed; new episodes appear automatically within ~hours of each push.

## Tuning

- Episode length: adaptive. Floor/ceiling tuned via `MIN_WORDS` / `MAX_WORDS` in `config.py` (~150 words/min spoken).
- Voice: `TTS_VOICE` / `TTS_RATE` in `config.py`. `say -v '?'` lists voices.
- Tickers / sectors / feeds: `config.py`.
- Model: `OLLAMA_MODEL` — `qwen2.5:14b` (default, solid), `llama3.3:latest` (heavier, slower), `llama3.2:latest` (fast).

## Troubleshooting

- `ollama` must be running (`ollama serve` or the Ollama.app).
- No episode? Check `run.log` and `launchd.err`.
- Apple/Spotify rejecting feed? Ensure `cover.jpg` exists and `PODCAST_BASE_URL` is reachable.
