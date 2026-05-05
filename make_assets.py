"""One-off: generate audio assets used to wrap every episode.

Run once locally:
    python make_assets.py

Outputs:
    assets/intro.mp3       — 2s news bumper sting (Sound Effects API)
    assets/outro.mp3       — 2s closing sting     (Sound Effects API)
    assets/bed.mp3         — 22s ambient music bed (Sound Effects API)
    assets/host_intro.mp3  — JAMIE voice saying the show's tagline (Text to Speech)

Cost: ~880 credits one-time for sound effects + ~50 credits for the host
intro line. <1% of monthly Creator budget. Re-run when you want to change
the show tagline or refresh the music bed vibe.
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
    (
        "bed",
        22.0,
        "Looping ambient music bed for a financial news podcast: very quiet, "
        "minimal, sparse rhodes piano with soft synth pad underneath, slow "
        "tempo around eighty BPM, no drums, no melody, no voice — just texture "
        "that will sit under spoken dialogue without distracting from it. "
        "Should loop seamlessly. Mood: focused, professional, calm.",
    ),
]


HOST_INTRO_TEXT = "Hey, this is Markets Explained, Daily."
HOST_INTRO_VOICE = "EXAVITQu4vr4xnSDxMaL"  # Sarah — same voice as JAMIE in tts.py

HOST_OUTRO_TEXT = (
    "And that's all for today folks! Make sure to stay curious and keep "
    "asking about the Markets Explained, Daily."
)
HOST_OUTRO_VOICE = "cgSgspJ2msm6clMCkdW9"  # Jessica — same voice as MAYA in tts.py


def _gen_sound_effect(name: str, dur: float, prompt: str) -> None:
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
    print(f"  wrote {out.stat().st_size // 1024} KB")


def _gen_voice_clip(filename: str, text: str, voice_id: str) -> None:
    out = ASSETS / filename
    print(f"generating {filename} (\"{text[:60]}{'…' if len(text) > 60 else ''}\") → {out}")
    for model in ("eleven_v3", "eleven_multilingual_v2"):
        try:
            audio_iter = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=model,
                text=text,
                output_format="mp3_44100_128",
            )
            with open(out, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
            print(f"  wrote {out.stat().st_size // 1024} KB (model={model})")
            return
        except Exception as e:
            print(f"  {model} failed: {e}")
    raise RuntimeError(f"both v3 and multilingual_v2 failed for {filename}")


def main() -> None:
    for name, dur, prompt in PROMPTS:
        _gen_sound_effect(name, dur, prompt)
    _gen_voice_clip("host_intro.mp3", HOST_INTRO_TEXT, HOST_INTRO_VOICE)
    _gen_voice_clip("host_outro.mp3", HOST_OUTRO_TEXT, HOST_OUTRO_VOICE)
    print("done — commit assets/ to repo")


if __name__ == "__main__":
    main()
