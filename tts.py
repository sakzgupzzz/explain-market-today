"""Text-to-speech for the daily roundtable.

Backends, in priority order:
  1. ElevenLabs v3 Text-to-Dialogue API — single batched call per chunk,
     native multi-speaker overlap and prosody. Best quality. Paid.
  2. ElevenLabs v2 per-turn text_to_speech (legacy fallback if v3 SDK unavailable).
  3. macOS `say` — local rotation of system voices per character.
  4. Piper libritts_r — local multi-speaker neural TTS for Linux.

After synthesis, every backend pipes the concatenated audio through a
broadcast mastering chain (highpass + compressor + 2-pass loudnorm to -16 LUFS
+ brick-wall limiter) before the final mp3.

Returns (mp3_path, chunk_timings) where chunk_timings is a list of
{"index", "start_sec", "end_sec", "speakers", "first_line"} for chapter generation.
"""
from __future__ import annotations
import json
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
    TTS_BACKEND, ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_OUTPUT_FORMAT,
    ELEVEN_CHARACTER_VOICES,
)

LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]{0,15}):\s*(.+)$")

# ElevenLabs v3 dialogue API limits (from docs: ≤2000 chars total inputs[].text,
# ≤10 unique voice_ids per request).
V3_MAX_CHARS_PER_REQUEST = 1800  # margin under 2000
V3_MAX_VOICES_PER_REQUEST = 10

MAC_CHARACTER_VOICES = {
    "JAMIE": os.environ.get("MAC_VOICE_JAMIE", "Samantha"),
    "ALEX":  os.environ.get("MAC_VOICE_ALEX",  "Daniel"),
    "MAYA":  os.environ.get("MAC_VOICE_MAYA",  "Karen"),
    "RIO":   os.environ.get("MAC_VOICE_RIO",   "Moira"),
    "KAI":   os.environ.get("MAC_VOICE_KAI",   "Eddy (English (US))"),
    "CAM":   os.environ.get("MAC_VOICE_CAM",   "Flo (English (US))"),
    "TESS":  os.environ.get("MAC_VOICE_TESS",  "Karen"),
    "DEV":   os.environ.get("MAC_VOICE_DEV",   "Daniel"),
}


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """Return list of (character_name, line_text). Lines without NAME: prefix
    glue to the previous speaker; lines before any speaker → DEFAULT_CHARACTER."""
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
            current_name = name if name in CHARACTERS else DEFAULT_CHARACTER
            current_buf.append(m.group(2).strip())
        else:
            current_buf.append(line)
    if current_buf:
        turns.append((current_name, " ".join(current_buf).strip()))
    return [(n, t) for n, t in turns if t]


def _resolve_backend() -> str:
    """Pick TTS backend. Explicit TTS_BACKEND wins; otherwise auto-detect."""
    if TTS_BACKEND in ("eleven", "eleven_v3", "eleven_v2", "mac", "piper"):
        return TTS_BACKEND
    if os.environ.get("FORCE_PIPER") == "1":
        return "piper"
    if ELEVENLABS_API_KEY:
        return "eleven"
    if platform.system() == "Darwin":
        return "mac"
    return "piper"


def synth(text: str, out_mp3: Path) -> tuple[Path, list[dict]]:
    """Synthesize the script to out_mp3. Returns (mp3_path, chunk_timings)."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    turns = parse_dialogue(text)
    if not turns:
        turns = [(DEFAULT_CHARACTER, text)]
    backend = _resolve_backend()
    print(f"[tts] backend={backend} turns={len(turns)}")
    if backend in ("eleven", "eleven_v3"):
        return _synth_eleven_v3(turns, out_mp3)
    if backend == "eleven_v2":
        return _synth_eleven_v2(turns, out_mp3)
    if backend == "mac":
        return _synth_mac_dialogue(turns, out_mp3)
    return _synth_piper_dialogue(turns, out_mp3)


# ─── ElevenLabs v3 — Text-to-Dialogue API ──────────────────────────────────

def _import_eleven_dialogue():
    """Load DialogueInput class. Tries multiple SDK versions."""
    try:
        from elevenlabs.types import DialogueInput  # type: ignore
        return DialogueInput
    except ImportError:
        pass
    try:
        from elevenlabs import DialogueInput  # type: ignore
        return DialogueInput
    except ImportError:
        pass
    # Construct a dict-shaped fallback. The SDK's pydantic model accepts dicts.
    return None


def _chunk_turns(turns: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group turns into chunks under V3_MAX_CHARS_PER_REQUEST and ≤10 unique voices."""
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_chars = 0
    current_voices: set[str] = set()
    for name, text in turns:
        text_len = len(text)
        if (current_chars + text_len > V3_MAX_CHARS_PER_REQUEST
                or (name not in current_voices and len(current_voices) >= V3_MAX_VOICES_PER_REQUEST)) and current:
            chunks.append(current)
            current = []
            current_chars = 0
            current_voices = set()
        current.append((name, text))
        current_chars += text_len
        current_voices.add(name)
    if current:
        chunks.append(current)
    return chunks


def _synth_eleven_v3(turns: list[tuple[str, str]], out_mp3: Path) -> tuple[Path, list[dict]]:
    """Batched v3 dialogue API. One call per chunk. Outputs mastered mp3."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    from elevenlabs.client import ElevenLabs
    DialogueInput = _import_eleven_dialogue()
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    chunks = _chunk_turns(turns)
    tmpdir = Path(tempfile.mkdtemp(prefix="tts_eleven_v3_"))
    try:
        chunk_wavs: list[Path] = []
        chunk_timings: list[dict] = []
        cum_sec = 0.0

        for idx, chunk in enumerate(chunks):
            inputs = []
            for name, text in chunk:
                voice_id = ELEVEN_CHARACTER_VOICES.get(
                    name, ELEVEN_CHARACTER_VOICES[DEFAULT_CHARACTER]
                )
                if DialogueInput is not None:
                    inputs.append(DialogueInput(text=text, voice_id=voice_id))
                else:
                    inputs.append({"text": text, "voice_id": voice_id})

            chunk_mp3 = tmpdir / f"chunk_{idx:03d}.mp3"
            try:
                audio_iter = client.text_to_dialogue.convert(
                    inputs=inputs,
                    model_id="eleven_v3",
                    output_format="mp3_44100_128",
                    apply_text_normalization="auto",
                )
            except (AttributeError, Exception) as e:
                # SDK doesn't have text_to_dialogue — fall back to per-turn v2.
                print(f"[tts] v3 dialogue API unavailable ({e}); falling back to v2 per-turn for chunk {idx}")
                _synth_chunk_v2_fallback(chunk, chunk_mp3, client)
            else:
                with open(chunk_mp3, "wb") as f:
                    for piece in audio_iter:
                        if piece:
                            f.write(piece)

            chunk_wav = tmpdir / f"chunk_{idx:03d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(chunk_mp3),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(chunk_wav)],
                check=True,
            )
            chunk_wavs.append(chunk_wav)
            dur = _wav_duration(chunk_wav)
            speakers = sorted({n for n, _ in chunk})
            first_line = chunk[0][1][:80]
            chunk_timings.append({
                "index": idx,
                "start_sec": cum_sec,
                "end_sec": cum_sec + dur,
                "speakers": speakers,
                "first_speaker": chunk[0][0],
                "first_line": first_line,
            })
            cum_sec += dur

        combined = tmpdir / "combined.wav"
        _concat_wavs(chunk_wavs, combined)
        _master_audio(combined, out_mp3)
        return out_mp3, chunk_timings
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _synth_chunk_v2_fallback(chunk: list[tuple[str, str]], out_mp3: Path, client) -> None:
    """If v3 dialogue endpoint fails, synthesize this chunk per-turn via v2."""
    tmpdir = Path(tempfile.mkdtemp(prefix="v2fallback_"))
    try:
        wavs: list[Path] = []
        for i, (name, text) in enumerate(chunk):
            voice_id = ELEVEN_CHARACTER_VOICES.get(
                name, ELEVEN_CHARACTER_VOICES[DEFAULT_CHARACTER]
            )
            seg_mp3 = tmpdir / f"seg_{i:04d}.mp3"
            audio_iter = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL,
                text=text,
                output_format="mp3_44100_128",
            )
            with open(seg_mp3, "wb") as f:
                for piece in audio_iter:
                    if piece:
                        f.write(piece)
            seg_wav = tmpdir / f"seg_{i:04d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(seg_mp3),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(seg_wav)],
                check=True,
            )
            wavs.append(seg_wav)
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, 44100, silence)
        interleaved: list[Path] = []
        for i, w in enumerate(wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(w)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(combined),
             "-codec:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _synth_eleven_v2(turns: list[tuple[str, str]], out_mp3: Path) -> tuple[Path, list[dict]]:
    """Legacy per-turn v2 path. Returns dummy chunk timings (one per turn)."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    tmpdir = Path(tempfile.mkdtemp(prefix="tts_eleven_v2_"))
    try:
        wavs: list[Path] = []
        timings: list[dict] = []
        cum = 0.0
        for idx, (name, text) in enumerate(turns):
            voice_id = ELEVEN_CHARACTER_VOICES.get(
                name, ELEVEN_CHARACTER_VOICES[DEFAULT_CHARACTER]
            )
            seg_mp3 = tmpdir / f"seg_{idx:04d}.mp3"
            audio_iter = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL,
                text=text,
                output_format="mp3_44100_128",
            )
            with open(seg_mp3, "wb") as f:
                for piece in audio_iter:
                    if piece:
                        f.write(piece)
            seg_wav = tmpdir / f"seg_{idx:04d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(seg_mp3),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(seg_wav)],
                check=True,
            )
            wavs.append(seg_wav)
            d = _wav_duration(seg_wav)
            timings.append({
                "index": idx, "start_sec": cum, "end_sec": cum + d,
                "speakers": [name], "first_speaker": name, "first_line": text[:80],
            })
            cum += d
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, 44100, silence)
        interleaved: list[Path] = []
        for i, w in enumerate(wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(w)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        _master_audio(combined, out_mp3)
        return out_mp3, timings
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── macOS `say` and Piper paths (preserved as fallbacks) ─────────────────

def _synth_mac_dialogue(turns: list[tuple[str, str]], out_mp3: Path) -> tuple[Path, list[dict]]:
    tmpdir = Path(tempfile.mkdtemp(prefix="tts_mac_"))
    try:
        wavs: list[Path] = []
        timings: list[dict] = []
        cum = 0.0
        for idx, (name, text) in enumerate(turns):
            voice = MAC_CHARACTER_VOICES.get(name, TTS_VOICE)
            # `say` does not understand audio tags — strip them
            text_clean = re.sub(r"\[[^\]]+\]", "", text).strip()
            aiff = tmpdir / f"seg_{idx:04d}.aiff"
            wav = tmpdir / f"seg_{idx:04d}.wav"
            subprocess.run(
                ["say", "-v", voice, "-r", str(TTS_RATE), "-o", str(aiff), text_clean],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            wavs.append(wav)
            d = _wav_duration(wav)
            timings.append({
                "index": idx, "start_sec": cum, "end_sec": cum + d,
                "speakers": [name], "first_speaker": name, "first_line": text[:80],
            })
            cum += d
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, 44100, silence)
        interleaved: list[Path] = []
        for i, w in enumerate(wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(w)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        _master_audio(combined, out_mp3)
        return out_mp3, timings
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _synth_piper_dialogue(turns: list[tuple[str, str]], out_mp3: Path) -> tuple[Path, list[dict]]:
    if not Path(PIPER_VOICE_PATH).exists():
        raise FileNotFoundError(f"Piper voice not found at {PIPER_VOICE_PATH}.")
    tmpdir = Path(tempfile.mkdtemp(prefix="tts_piper_"))
    try:
        wavs: list[Path] = []
        timings: list[dict] = []
        cum = 0.0
        for idx, (name, text) in enumerate(turns):
            speaker = CHARACTERS[name]["speaker"]
            text_clean = re.sub(r"\[[^\]]+\]", "", text).strip()
            seg = tmpdir / f"seg_{idx:04d}.wav"
            subprocess.run(
                ["piper", "--model", PIPER_VOICE_PATH, "--speaker", str(speaker),
                 "--output_file", str(seg)],
                input=text_clean, text=True, check=True,
            )
            seg_44 = tmpdir / f"seg_{idx:04d}_44.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(seg),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(seg_44)],
                check=True,
            )
            wavs.append(seg_44)
            d = _wav_duration(seg_44)
            timings.append({
                "index": idx, "start_sec": cum, "end_sec": cum + d,
                "speakers": [name], "first_speaker": name, "first_line": text[:80],
            })
            cum += d
        silence = tmpdir / "silence.wav"
        _silence_wav(INTER_LINE_SILENCE_MS, 44100, silence)
        interleaved: list[Path] = []
        for i, w in enumerate(wavs):
            if i > 0:
                interleaved.append(silence)
            interleaved.append(w)
        combined = tmpdir / "combined.wav"
        _concat_wavs(interleaved, combined)
        _master_audio(combined, out_mp3)
        return out_mp3, timings
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── ffmpeg helpers + mastering chain ─────────────────────────────────────

def _wav_duration(wav: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)]
    )
    return float(out.strip())


def _concat_wavs(wavs: list[Path], out_wav: Path) -> None:
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


def _master_audio(in_wav: Path, out_mp3: Path) -> None:
    """Two-pass loudnorm + highpass + compressor + brick-wall limiter to -16 LUFS.
    Falls back to plain encode if loudnorm measurement fails."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(in_wav), "-af",
             "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        match = re.search(r"\{[\s\S]*?\}", proc.stderr)
        if not match:
            raise ValueError("no loudnorm JSON in stderr")
        data = json.loads(match.group(0))
        chain = (
            "highpass=f=80,"
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={data['input_i']}:"
            f"measured_TP={data['input_tp']}:"
            f"measured_LRA={data['input_lra']}:"
            f"measured_thresh={data['input_thresh']}:"
            f"offset={data['target_offset']}:"
            "linear=true,"
            "alimiter=limit=0.95"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
             "-af", chain, "-ar", "44100", "-ac", "1",
             "-codec:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True,
        )
        print(f"[master] applied 2-pass loudnorm: input_i={data['input_i']} → -16 LUFS")
    except Exception as e:
        print(f"[master] loudnorm failed ({e}), falling back to plain encode")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
             "-codec:a", "libmp3lame", "-b:a", "128k",
             "-ar", "44100", "-ac", "1", str(out_mp3)],
            check=True,
        )


def audio_duration_seconds(mp3: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)]
    )
    return float(out.strip())


if __name__ == "__main__":
    import sys
    demo = (
        "JAMIE: Jamie here — big day on the tape.\n"
        "ALEX: [deadpan] Alex on equities. Tech ripped four percent. The reason was vibes.\n"
    )
    out = Path("test.mp3")
    synth(sys.argv[1] if len(sys.argv) > 1 else demo, out)
    print(f"wrote {out}")
