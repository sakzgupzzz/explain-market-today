"""Text-to-speech. macOS → `say`. Linux → Piper."""
from __future__ import annotations
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from config import TTS_VOICE, TTS_RATE, PIPER_VOICE_PATH


def synth(text: str, out_mp3: Path) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin" and os.environ.get("FORCE_PIPER") != "1":
        _synth_say(text, out_mp3)
    else:
        _synth_piper(text, out_mp3)
    return out_mp3


def _to_mp3(src: Path, out_mp3: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-codec:a", "libmp3lame", "-b:a", "96k", "-ar", "44100", "-ac", "1",
            str(out_mp3),
        ],
        check=True,
    )


def _synth_say(text: str, out_mp3: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = Path(tmp.name)
    try:
        subprocess.run(
            ["say", "-v", TTS_VOICE, "-r", str(TTS_RATE), "-o", str(aiff), text],
            check=True,
        )
        _to_mp3(aiff, out_mp3)
    finally:
        aiff.unlink(missing_ok=True)


def _synth_piper(text: str, out_mp3: Path) -> None:
    if not Path(PIPER_VOICE_PATH).exists():
        raise FileNotFoundError(
            f"Piper voice not found at {PIPER_VOICE_PATH}. "
            "Set PIPER_VOICE_PATH or install a voice .onnx + .onnx.json."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        subprocess.run(
            ["piper", "--model", PIPER_VOICE_PATH, "--output_file", str(wav)],
            input=text, text=True, check=True,
        )
        _to_mp3(wav, out_mp3)
    finally:
        wav.unlink(missing_ok=True)


def audio_duration_seconds(mp3: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)]
    )
    return float(out.strip())


if __name__ == "__main__":
    import sys
    synth(sys.argv[1] if len(sys.argv) > 1 else "Hello world.", Path("test.mp3"))
    print("wrote test.mp3")
