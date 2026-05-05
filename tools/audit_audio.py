"""Local-only audio QA — runs whisper.cpp transcription on a published
episode and compares to the script.

Not run on CI (whisper model is ~1.5 GB; would balloon runner time).
Use locally to spot-check episodes for speaker confusion, dropouts, or
any TTS rendering glitch.

Usage:
    python tools/audit_audio.py docs/episodes/2026-05-05.mp3

Requires whisper.cpp installed and model present:
    brew install whisper-cpp                 (macOS)
    # OR build from https://github.com/ggerganov/whisper.cpp
    Download model: ggml-base.en.bin (~150 MB) — sufficient for clean TTS
"""
from __future__ import annotations
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

MODEL_CANDIDATES = [
    Path.home() / "whisper.cpp" / "models" / "ggml-base.en.bin",
    Path.home() / ".local" / "share" / "whisper" / "ggml-base.en.bin",
    Path("/opt/homebrew/share/whisper/ggml-base.en.bin"),
]
WHISPER_BIN_CANDIDATES = ["whisper-cli", "main", "whisper"]


def _find_model() -> Path | None:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def _find_binary() -> str | None:
    for name in WHISPER_BIN_CANDIDATES:
        try:
            subprocess.run([name, "--help"], check=False, capture_output=True, timeout=5)
            return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def transcribe(mp3: Path) -> str:
    binary = _find_binary()
    model = _find_model()
    if not binary or not model:
        print("Install whisper.cpp + download ggml-base.en.bin first.")
        print("  brew install whisper-cpp")
        print("  curl -L -o ~/.local/share/whisper/ggml-base.en.bin \\")
        print("      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
        sys.exit(1)
    out = subprocess.check_output(
        [binary, "-m", str(model), "-f", str(mp3), "-otxt", "-of", str(mp3.with_suffix(""))],
        timeout=600,
    )
    txt_path = mp3.with_suffix(".whisper.txt")
    if txt_path.exists():
        return txt_path.read_text()
    return out.decode("utf-8", errors="ignore")


def normalize(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"^[A-Z][A-Z0-9_]{0,15}:", "", text, flags=re.M)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def compare(script_path: Path, transcript: str) -> dict:
    script = script_path.read_text() if script_path.exists() else ""
    a = normalize(script)
    b = normalize(transcript)
    sim = SequenceMatcher(None, a, b).ratio()
    return {
        "script_words": len(a.split()),
        "transcript_words": len(b.split()),
        "similarity": round(sim, 3),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mp3 = Path(sys.argv[1])
    if not mp3.exists():
        print(f"not found: {mp3}")
        sys.exit(1)
    print(f"transcribing {mp3}…")
    transcript = transcribe(mp3)
    script = mp3.with_suffix(".txt")
    result = compare(script, transcript)
    print(result)
    if result["similarity"] < 0.6:
        print(f"⚠ low similarity ({result['similarity']}) — possible TTS rendering glitch")


if __name__ == "__main__":
    main()
