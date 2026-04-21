"""Text-to-speech. Dialogue-aware multi-speaker synth on Linux (Piper libritts_r).
macOS falls back to `say` with voice rotation per character."""
from __future__ import annotations
import os
import platform
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from config import (
    TTS_VOICE, TTS_RATE, PIPER_VOICE_PATH, CHARACTERS, DEFAULT_CHARACTER,
    INTER_LINE_SILENCE_MS,
)

# Line pattern: NAME: text
LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]{0,15}):\s*(.+)$")

# Mac `say` voice per character (best-effort). User can remap via env.
MAC_CHARACTER_VOICES = {
    "JAMIE": os.environ.get("MAC_VOICE_JAMIE", "Samantha"),
    "ALEX":  os.environ.get("MAC_VOICE_ALEX",  "Tom"),
}


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """Return list of (character_name, line_text). Lines without a NAME: prefix
    get glued to the previous speaker; lines before any speaker go to DEFAULT_CHARACTER."""
    turns: list[tuple[str, str]] = []
    current_name = DEFAULT_CHARACTER
    current_buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            if current_buf:
                turns.append((current_name, " ".join(current_buf).strip()))
                current_buf = []
            name = m.group(1)
            if name not in CHARACTERS:
                # unknown name — treat as default narrator
                name = DEFAULT_CHARACTER
            current_name = name
            current_buf.append(m.group(2).strip())
        else:
            current_buf.append(line)
    if current_buf:
        turns.append((current_name, " ".join(current_buf).strip()))
    return [(n, t) for n, t in turns if t]


def synth(text: str, out_mp3: Path) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    turns = parse_dialogue(text)
    if not turns:
        turns = [(DEFAULT_CHARACTER, text)]
    if platform.system() == "Darwin" and os.environ.get("FORCE_PIPER") != "1":
        _synth_mac_dialogue(turns, out_mp3)
    else:
        _synth_piper_dialogue(turns, out_mp3)
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


def _concat_wavs(wavs: list[Path], out_wav: Path) -> None:
    """ffmpeg concat demuxer. All inputs must share sample rate + channels."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for w in wavs:
            f.write(f"file '{w.resolve()}'\n")
        listfile = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(out_wav)],
            check=True,
        )
    finally:
        listfile.unlink(missing_ok=True)


def _silence_wav(ms: int, sample_rate: int, path: Path) -> None:
    dur = ms / 1000.0
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anullsrc=r={sample_rate}:cl=mono",
         "-t", f"{dur}", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )


def _piper_segment(text: str, speaker_id: int, out_wav: Path) -> None:
    if not Path(PIPER_VOICE_PATH).exists():
        raise FileNotFoundError(
            f"Piper voice not found at {PIPER_VOICE_PATH}."
        )
    subprocess.run(
        ["piper", "--model", PIPER_VOICE_PATH, "--speaker", str(speaker_id),
         "--output_file", str(out_wav)],
        input=text, text=True, check=True,
    )


def _wav_sample_rate(wav: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)]
    )
    return int(out.strip())


def _synth_piper_dialogue(turns: list[tuple[str, str]], out_mp3: Path) -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="tts_"))
    try:
        segment_wavs: list[Path] = []
        for idx, (name, text) in enumerate(turns):
            speaker = CHARACTERS[name]["speaker"]
            seg = tmpdir / f"seg_{idx:04d}.wav"
            _piper_segment(text, speaker, seg)
            segment_wavs.append(seg)
        # insert silence between turns
        sr = _wav_sample_rate(segment_wavs[0]) if segment_wavs else 22050
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, sr, silence)
        interleaved: list[Path] = []
        for i, seg in enumerate(segment_wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(seg)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        _to_mp3(combined, out_mp3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _synth_mac_dialogue(turns: list[tuple[str, str]], out_mp3: Path) -> None:
    """macOS: rotate `say` voices per character. One AIFF per turn, concat via ffmpeg."""
    tmpdir = Path(tempfile.mkdtemp(prefix="tts_mac_"))
    try:
        wavs: list[Path] = []
        for idx, (name, text) in enumerate(turns):
            voice = MAC_CHARACTER_VOICES.get(name, TTS_VOICE)
            aiff = tmpdir / f"seg_{idx:04d}.aiff"
            wav = tmpdir / f"seg_{idx:04d}.wav"
            subprocess.run(
                ["say", "-v", voice, "-r", str(TTS_RATE), "-o", str(aiff), text],
                check=True,
            )
            # normalize to 22050 mono wav for consistent concat
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                 "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            wavs.append(wav)
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, 22050, silence)
        interleaved: list[Path] = []
        for i, w in enumerate(wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(w)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        _to_mp3(combined, out_mp3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def audio_duration_seconds(mp3: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)]
    )
    return float(out.strip())


if __name__ == "__main__":
    import sys
    demo = (
        "JAMIE: Hey, big day on the tape.\n"
        "ALEX: Yeah, tech ripped. I will tell you why.\n"
    )
    synth(sys.argv[1] if len(sys.argv) > 1 else demo, Path("test.mp3"))
    print("wrote test.mp3")
