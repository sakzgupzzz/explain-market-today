"""One-off: generate intro and outro stings via ElevenLabs Sound Effects API.

Run once locally:
    python make_assets.py

Outputs assets/intro.mp3 and assets/outro.mp3, then commits them. After that,
tts.py wraps every episode with these stings — no per-episode regeneration,
no recurring credit cost.

Pricing: ElevenLabs Sound Effects bills at 40 credits/sec when duration is
set. Two 2-second clips = 160 credits total, ~0.16% of the Creator monthly
budget. Run once.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from elevenlabs.client import ElevenLabs

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

key = os.environ.get("ELEVENLABS_API_KEY", "")
if not key:
    raise SystemExit("ELEVENLABS_API_KEY not set (check .env)")

client = ElevenLabs(api_key=key)

PROMPTS = [
    (
        "intro",
        2.0,
        "Quick news bumper sting: a single low synth bass thump followed by a "
        "bright high digital chime, modern financial podcast opener, no voice, "
        "no melody, ends cleanly on the chime.",
    ),
    (
        "outro",
        2.0,
        "Soft podcast closing sting: a warm mellow synth chord that resolves "
        "and fades out, gentle bell tone at the start, no voice, no melody, "
        "feels like an end card.",
    ),
]


def main() -> None:
    for name, dur, prompt in PROMPTS:
        out = ASSETS / f"{name}.mp3"
        print(f"generating {name} ({dur}s) → {out}")
        audio_iter = client.text_to_sound_effects.convert(
            text=prompt,
            duration_seconds=dur,
            prompt_influence=0.5,
        )
        with open(out, "wb") as f:
            for chunk in audio_iter:
                if chunk:
                    f.write(chunk)
        size_kb = out.stat().st_size // 1024
        print(f"  wrote {size_kb} KB")
    print("done — commit assets/ to repo")


if __name__ == "__main__":
    main()
