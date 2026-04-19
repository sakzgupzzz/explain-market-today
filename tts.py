"""macOS `say` → AIFF → MP3 via ffmpeg. Zero install beyond macOS + brew ffmpeg."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from config import TTS_VOICE, TTS_RATE


def synth(text: str, out_mp3: Path) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = Path(tmp.name)
    try:
        subprocess.run(
            ["say", "-v", TTS_VOICE, "-r", str(TTS_RATE), "-o", str(aiff), text],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(aiff),
                "-codec:a", "libmp3lame", "-b:a", "96k", "-ar", "44100", "-ac", "1",
                str(out_mp3),
            ],
            check=True,
        )
    finally:
        aiff.unlink(missing_ok=True)
    return out_mp3


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
