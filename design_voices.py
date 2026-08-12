"""Generate synthetic host voices via ElevenLabs Voice Design.

Why this exists: the stock premade voices (Sarah/Brian/Jessica) are the single
biggest "AI tell" left in the show — listeners recognize them from every other
AI demo. Voice Design generates brand-new voices from a text prompt: no real
person, no reference audio, fully licensed for use. This replaces the Instant
Voice Clone path (which needs consented reference audio we don't have).

Two phases:

  1. preview — generate candidate voices for each host and write them to
     voice_previews/<NAME>_<i>.mp3 plus a manifest. Listen, pick favorites.

       python design_voices.py preview
       python design_voices.py preview JAMIE          # just one host

  2. save — turn the chosen previews into permanent library voices and print
     the export lines to wire into the pipeline.

       python design_voices.py save JAMIE:1 ALEX:0 MAYA:2

The save step prints `export ELEVEN_VOICE_<NAME>=<voice_id>` lines. Add them to
your env (or .env / CI secrets) and tts.py picks them up via config.py — no code
change. Each preview generation costs a small amount of ElevenLabs credit; saving
a voice consumes a voice slot on your plan.

Requires ELEVENLABS_API_KEY and network.
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import ROOT, ELEVENLABS_API_KEY, ELEVENLABS_OUTPUT_FORMAT

PREVIEW_DIR = ROOT / "voice_previews"
MANIFEST_PATH = PREVIEW_DIR / "manifest.json"

# Voice Design prompts, one per host. These describe TIMBRE / AGE / PACE /
# ACCENT — not what they say. Tuned to keep the three timbres distinct so a
# listener can tell who's talking within a syllable (same goal as the old
# stock-voice cast, minus the recognizability). Edit freely and re-run preview.
VOICE_PROMPTS: dict[str, dict[str, str]] = {
    "JAMIE": {
        "description": (
            "A warm, confident female podcast host in her early thirties with a "
            "neutral American accent. Bright and engaged, conversational pace, "
            "natural smiling energy without sounding over-produced. Clear studio "
            "recording, no background noise."
        ),
        # Sample text drives how the preview sounds — write in the host's voice.
        "text": (
            "Big day on the tape, and I want to know why. Markets ripped higher "
            "into the close, the Fed's back in the headlines, and there's one tech "
            "story nobody saw coming. Let's get into it — Alex, start us off."
        ),
    },
    "ALEX": {
        "description": (
            "A deep, resonant male markets analyst in his forties with a neutral "
            "American accent. Dry, precise, measured delivery with deadpan humor. "
            "Unhurried and authoritative, like a seasoned desk strategist. Clean "
            "studio recording, no background noise."
        ),
        "text": (
            "Equities closed up four percent. The official reason is strong "
            "earnings. The actual reason, as far as I can tell, is vibes. Rates "
            "ticked lower, the dollar gave some back, and nobody on the desk wants "
            "to say the word bubble out loud."
        ),
    },
    # Guest correspondents — occasional appearances, deliberately contrasting
    # with the core cast (JAMIE warm F/30s, ALEX deep M/40s, MAYA quick F/20s).
    "TESS": {
        "description": (
            "A wry British female economics editor in her early fifties with a "
            "polished Received Pronunciation accent. Dry wit, unhurried, quietly "
            "authoritative — decades of covering central banks. Warm but never "
            "breathless. Clean studio recording, no background noise."
        ),
        "text": (
            "The Bank raised rates again, which surprised precisely no one who "
            "reads the minutes and absolutely everyone who trades on them. "
            "Markets threw their customary tantrum, then thought better of it "
            "by lunch. Plus ça change, as we say on the desk."
        ),
    },
    "RUSS": {
        "description": (
            "A gravelly, laconic American male commodities and energy reporter "
            "in his late fifties with a faint Texas drawl. Slow, plainspoken, "
            "no-nonsense delivery with understated humor — sounds like he's "
            "seen five oil cycles and is impressed by none of them. Clean "
            "studio recording, no background noise."
        ),
        "text": (
            "Crude's back over ninety and everybody's acting brand new about "
            "it. Same story every time — inventories draw down, somebody's "
            "pipeline hiccups, and suddenly the whole desk remembers oil "
            "exists. Natural gas, meanwhile, can't catch a bid to save its life."
        ),
    },
    "MAYA": {
        "description": (
            "A bright, fast-talking female tech and culture reporter in her late "
            "twenties with a neutral American accent. Energetic and hype-aware but "
            "skeptical, quick cadence, expressive and playful. Crisp studio "
            "recording, no background noise."
        ),
        "text": (
            "Okay so the launch demo was gorgeous and also completely staged, which "
            "is honestly the most tech thing I've ever seen. They raised another "
            "billion dollars to build a tool that writes emails. The future is here "
            "and it is checking your inbox."
        ),
    },
}


def _client():
    if not ELEVENLABS_API_KEY:
        sys.exit("ELEVENLABS_API_KEY not set — export it and retry.")
    from elevenlabs.client import ElevenLabs
    # Voice Design generation is slow (multiple candidates per call); the SDK's
    # default ~30s read timeout trips a ReadTimeout. Give it generous headroom.
    return ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=300)


def cmd_preview(names: list[str]) -> None:
    """Generate candidate voices for each requested host. Writes mp3s +
    manifest.json mapping <NAME>:<index> → generated_voice_id."""
    client = _client()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    out_fmt = ELEVENLABS_OUTPUT_FORMAT if ELEVENLABS_OUTPUT_FORMAT.startswith("mp3") else "mp3_44100_128"

    for name in names:
        spec = VOICE_PROMPTS[name]
        print(f"\n[{name}] generating previews — {spec['description'][:60]}...")
        resp = client.text_to_voice.create_previews(
            voice_description=spec["description"],
            text=spec["text"],
            output_format=out_fmt,
        )
        entries = []
        for i, prev in enumerate(resp.previews):
            mp3_path = PREVIEW_DIR / f"{name}_{i}.mp3"
            mp3_path.write_bytes(base64.b64decode(prev.audio_base_64))
            entries.append({
                "index": i,
                "generated_voice_id": prev.generated_voice_id,
                "duration_secs": prev.duration_secs,
                "mp3": str(mp3_path.relative_to(ROOT)),
            })
            print(f"  [{i}] {mp3_path.relative_to(ROOT)}  ({prev.duration_secs:.1f}s)")
        manifest[name] = {
            "description": spec["description"],
            "previews": entries,
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {MANIFEST_PATH.relative_to(ROOT)}.")
    print("Listen to the mp3s, then save your picks, e.g.:")
    print("  python design_voices.py save " + " ".join(f"{n}:0" for n in names))


def cmd_save(picks: list[str]) -> None:
    """Turn chosen previews into permanent library voices. picks = ['JAMIE:1', ...]."""
    if not MANIFEST_PATH.exists():
        sys.exit("No manifest — run `python design_voices.py preview` first.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    client = _client()

    exports: list[str] = []
    for pick in picks:
        try:
            name, idx_s = pick.split(":")
            idx = int(idx_s)
        except ValueError:
            sys.exit(f"Bad pick '{pick}' — expected NAME:INDEX, e.g. JAMIE:1")
        if name not in manifest:
            sys.exit(f"No previews for {name} in manifest — run preview {name} first.")
        entry = next((e for e in manifest[name]["previews"] if e["index"] == idx), None)
        if entry is None:
            sys.exit(f"No preview index {idx} for {name}.")
        print(f"[{name}] saving preview {idx} ({entry['generated_voice_id']})...")
        voice = client.text_to_voice.create_voice_from_preview(
            voice_name=f"{name.title()} (Market Today)",
            voice_description=manifest[name]["description"],
            generated_voice_id=entry["generated_voice_id"],
        )
        voice_id = getattr(voice, "voice_id", None) or getattr(voice, "voiceId", None)
        print(f"  → voice_id {voice_id}")
        exports.append(f"export ELEVEN_VOICE_{name}={voice_id}")

    print("\nAdd these to your env (or .env / CI secrets):\n")
    print("\n".join(exports))


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in ("preview", "save"):
        sys.exit(__doc__)
    cmd, rest = args[0], args[1:]
    if cmd == "preview":
        names = [a.upper() for a in rest] or list(VOICE_PROMPTS)
        bad = [n for n in names if n not in VOICE_PROMPTS]
        if bad:
            sys.exit(f"Unknown host(s): {bad}. Known: {list(VOICE_PROMPTS)}")
        cmd_preview(names)
    else:
        if not rest:
            sys.exit("save needs picks, e.g.: python design_voices.py save JAMIE:1 ALEX:0 MAYA:2")
        cmd_save(rest)


if __name__ == "__main__":
    main()
