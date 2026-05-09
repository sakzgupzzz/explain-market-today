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
    ROOT, TTS_VOICE, TTS_RATE, PIPER_VOICE_PATH, CHARACTERS, DEFAULT_CHARACTER,
    INTER_LINE_SILENCE_MS, AUDIO_SPEEDUP,
    TTS_BACKEND, ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_OUTPUT_FORMAT,
    ELEVEN_CHARACTER_VOICES,
)

INTRO_STING = ROOT / "assets" / "intro.mp3"
OUTRO_STING = ROOT / "assets" / "outro.mp3"
HOST_INTRO = ROOT / "assets" / "host_intro.mp3"
HOST_OUTRO = ROOT / "assets" / "host_outro.mp3"
HOST_DISCLAIMER = ROOT / "assets" / "host_disclaimer.mp3"
MUSIC_BED = ROOT / "assets" / "bed.mp3"
STING_GAP_MS = 350         # silence between sting and the next element
HOST_INTRO_GAP_MS = 250    # gap between host intro/outro and dialogue
BED_TAIL_SEC = 1.6         # bed continues this long after host outro before fading out
# Bed gain in dB applied before sidechain duck. -12 dB is audible in
# voice pauses/transitions while still ducking under speech via the
# sidechain compressor in _mix_music_bed. Bump to -10 for more presence,
# drop to -16 for just-barely-there. Tunable via BED_GAIN_DB env var.
BED_GAIN_DB = float(os.environ.get("BED_GAIN_DB", "-8"))

LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]{0,15}):\s*(.+)$")

# ElevenLabs v3 dialogue API limits (from docs: ≤2000 chars total inputs[].text,
# ≤10 unique voice_ids per request).
V3_MAX_CHARS_PER_REQUEST = 1800  # margin under 2000
V3_MAX_VOICES_PER_REQUEST = 10

MAC_CHARACTER_VOICES = {
    "JAMIE": os.environ.get("MAC_VOICE_JAMIE", "Samantha"),
    "ALEX":  os.environ.get("MAC_VOICE_ALEX",  "Daniel"),
    "MAYA":  os.environ.get("MAC_VOICE_MAYA",  "Karen"),
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
    """Synthesize the script to out_mp3. Returns (mp3_path, chunk_timings).
    After backend renders the dialogue, wraps with intro/outro stings if
    assets/intro.mp3 and assets/outro.mp3 exist.

    The pre-recorded disclaimer (assets/host_disclaimer.mp3) replaces any
    disclaimer-flavored turn in the dialogue — saves Eleven char budget
    by not re-synthesizing the same legal line every episode."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    turns = parse_dialogue(text)
    if not turns:
        turns = [(DEFAULT_CHARACTER, text)]
    # Drop any disclaimer-flavored turn(s) from synthesis input — we'll
    # append the pre-recorded clip in _add_host_intro_with_bed instead.
    # Guard: if stripping would leave us with 0 turns, keep the disclaimer
    # in the dialogue (better to "double-disclaim" than synth empty input).
    if HOST_DISCLAIMER.exists():
        before = len(turns)
        stripped = [(n, t) for n, t in turns if "entertainment and education only" not in t.lower()]
        if stripped:
            turns = stripped
            if len(turns) < before:
                print(f"[tts] stripped {before - len(turns)} disclaimer turn(s) from synth — using pre-recorded clip")
        else:
            print(f"[tts] script was only disclaimer ({before} turn(s)); keeping in dialogue path to avoid empty synth")
    if not turns:
        # Defensive — shouldn't reach here after the guard above, but if a
        # caller passes an empty script, log and return without erroring.
        print(f"[tts] WARN: empty turn list after parse — skipping synth, writing 1-sec silence to {out_mp3}")
        _silence_wav(1000, 44100, out_mp3.with_suffix(".tmp.wav"))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_mp3.with_suffix(".tmp.wav")),
             "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True,
        )
        out_mp3.with_suffix(".tmp.wav").unlink(missing_ok=True)
        return out_mp3, []
    backend = _resolve_backend()
    print(f"[tts] backend={backend} turns={len(turns)}")
    if backend in ("eleven", "eleven_v3"):
        result = _synth_eleven_v3(turns, out_mp3)
    elif backend == "eleven_v2":
        result = _synth_eleven_v2(turns, out_mp3)
    elif backend == "mac":
        result = _synth_mac_dialogue(turns, out_mp3)
    else:
        result = _synth_piper_dialogue(turns, out_mp3)

    # Build [2s_bed_lead + host_intro + 0.25s_bed_gap + dialogue] with bed
    # sidechain-ducked underneath the whole thing. Falls back to plain
    # bed-under-dialogue mix if host_intro asset is missing.
    _add_host_intro_with_bed(out_mp3)
    # Outer wrap: intro_sting + (bedded segment) + outro_sting. No bed under
    # the chimes themselves.
    _wrap_with_stings(out_mp3)
    return result


def _add_host_intro_with_bed(in_out_mp3: Path) -> None:
    """Wrap dialogue with bed-mixed host bookends:

      [2s bed-only lead-in] → [JAMIE host_intro] → [0.25s] → [dialogue] →
      [0.25s] → [MAYA host_outro] → [BED_TAIL_SEC bed-only tail with fade-out]

    Bed plays continuously underneath the entire span and sidechain-ducks
    under any voice. Single continuous bed loop — no audible discontinuity
    between intro / dialogue / outro. Each host clip is optional; missing
    assets are skipped. If no bed is present, this is a no-op."""
    if not MUSIC_BED.exists():
        return
    if not HOST_INTRO.exists():
        _mix_music_bed(in_out_mp3)
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="hostbookend_"))
    try:
        host_intro_wav = tmpdir / "host_intro.wav"
        dlg_wav = tmpdir / "dlg.wav"
        for src, dst in [(HOST_INTRO, host_intro_wav), (in_out_mp3, dlg_wav)]:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                check=True,
            )
        lead_sil = tmpdir / "lead.wav"
        gap_sil = tmpdir / "gap.wav"
        tail_sil = tmpdir / "tail.wav"
        _silence_wav(2000, 44100, lead_sil)
        _silence_wav(HOST_INTRO_GAP_MS, 44100, gap_sil)
        _silence_wav(int(BED_TAIL_SEC * 1000), 44100, tail_sil)

        parts = [lead_sil, host_intro_wav, gap_sil, dlg_wav]

        if HOST_OUTRO.exists():
            host_outro_wav = tmpdir / "host_outro.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(HOST_OUTRO),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(host_outro_wav)],
                check=True,
            )
            parts.extend([gap_sil, host_outro_wav])
        # Pre-recorded disclaimer: appended right before the bed tail so it
        # rides under the same fading-out music.
        if HOST_DISCLAIMER.exists():
            host_disclaimer_wav = tmpdir / "host_disclaimer.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(HOST_DISCLAIMER),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(host_disclaimer_wav)],
                check=True,
            )
            parts.extend([gap_sil, host_disclaimer_wav])
        parts.append(tail_sil)

        voice_track = tmpdir / "voice.wav"
        _concat_wavs(parts, voice_track)
        voice_dur = _file_duration_seconds(voice_track)
        # bed fades in over 0.5s at the start; fades out over BED_TAIL_SEC
        # at the end so the music gracefully resolves.
        fade_out_start = max(0.0, voice_dur - BED_TAIL_SEC)
        chain = (
            f"[1:a]aloop=loop=-1:size=2147483647,atrim=duration={voice_dur:.3f},"
            f"afade=t=in:d=0.5,"
            f"afade=t=out:st={fade_out_start:.3f}:d={BED_TAIL_SEC},"
            f"volume={BED_GAIN_DB}dB[bedlow];"
            f"[bedlow][0:a]sidechaincompress="
            f"threshold=0.05:ratio=8:attack=20:release=400:makeup=1:level_sc=1.5[ducked];"
            f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
            f"alimiter=limit=0.97[out]"
        )
        out_tmp = tmpdir / "bedded.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(voice_track), "-i", str(MUSIC_BED),
             "-filter_complex", chain, "-map", "[out]",
             "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k",
             str(out_tmp)],
            check=True,
        )
        shutil.copy(out_tmp, in_out_mp3)
        outro_note = " + host_outro" if HOST_OUTRO.exists() else ""
        print(f"[bed+host] 2s lead + host_intro + dialogue{outro_note} + {BED_TAIL_SEC}s tail "
              f"({voice_dur:.1f}s, gain {BED_GAIN_DB}dB)")
    except Exception as e:
        print(f"[bed+host] failed ({e}); falling back to bed-under-dialogue only")
        _mix_music_bed(in_out_mp3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mix_music_bed(in_out_mp3: Path) -> None:
    """Mix MUSIC_BED under in_out_mp3 with sidechain ducking. Edits in place
    via a temp file. No-op if assets/bed.mp3 missing."""
    if not MUSIC_BED.exists():
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="bed_"))
    try:
        # complex filtergraph:
        # [0] = dialogue mp3 (in_out_mp3)
        # [1] = bed.mp3 looped, trimmed to dialogue duration, attenuated
        # ducked = bed sidechain-compressed by dialogue (drops bed when voice plays)
        # mixed  = dialogue + ducked bed, summed
        dlg_dur = _file_duration_seconds(in_out_mp3)
        chain = (
            f"[1:a]aloop=loop=-1:size=2147483647,atrim=duration={dlg_dur:.3f},"
            f"volume={BED_GAIN_DB}dB[bedlow];"
            f"[bedlow][0:a]sidechaincompress="
            f"threshold=0.05:ratio=8:attack=20:release=400:makeup=1:level_sc=1.5[ducked];"
            f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
            f"alimiter=limit=0.97[out]"
        )
        out_tmp = tmpdir / "bedded.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(in_out_mp3), "-i", str(MUSIC_BED),
             "-filter_complex", chain, "-map", "[out]",
             "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k",
             str(out_tmp)],
            check=True,
        )
        shutil.copy(out_tmp, in_out_mp3)
        print(f"[bed] mixed bed under dialogue ({dlg_dur:.1f}s, gain {BED_GAIN_DB}dB)")
    except Exception as e:
        print(f"[bed] mix failed ({e}); shipping dialogue without bed")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _file_duration_seconds(p: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)]
    )
    return float(out.strip())


def _wrap_with_stings(in_out_mp3: Path) -> None:
    """Wrap content with intro + outro stings (no bed under the stings).
    Order: intro_sting → gap → content → gap → outro_sting.
    Host intro and bed are already baked into in_out_mp3 by
    _add_host_intro_with_bed before this runs."""
    if not (INTRO_STING.exists() and OUTRO_STING.exists()):
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="stings_"))
    try:
        intro_wav = tmpdir / "intro.wav"
        content_wav = tmpdir / "content.wav"
        outro_wav = tmpdir / "outro.wav"
        gap_wav = tmpdir / "gap.wav"
        for src, dst in [(INTRO_STING, intro_wav), (in_out_mp3, content_wav), (OUTRO_STING, outro_wav)]:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                check=True,
            )
        _silence_wav(STING_GAP_MS, 44100, gap_wav)
        combined = tmpdir / "combined.wav"
        _concat_wavs([intro_wav, gap_wav, content_wav, gap_wav, outro_wav], combined)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(combined),
             "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "1",
             str(in_out_mp3)],
            check=True,
        )
        print(f"[stings] wrapped intro_sting + content + outro_sting")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
            except AttributeError as e:
                # SDK doesn't have text_to_dialogue — fall back to per-turn v2.
                # Narrowly only AttributeError: auth/quota errors must propagate
                # so we don't double-bill by re-synthing the same chunk on v2.
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
    """Two-pass loudnorm + highpass + compressor + brick-wall limiter to -16 LUFS,
    then atempo speedup. Falls back to plain encode if loudnorm fails."""
    speedup = max(0.5, min(2.0, AUDIO_SPEEDUP))
    speedup_filter = f",atempo={speedup}" if abs(speedup - 1.0) > 0.01 else ""
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
            f"{speedup_filter}"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
             "-af", chain, "-ar", "44100", "-ac", "1",
             "-codec:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            check=True,
        )
        print(f"[master] applied 2-pass loudnorm + atempo={speedup}: input_i={data['input_i']} → -16 LUFS")
    except Exception as e:
        print(f"[master] loudnorm failed ({e}), falling back to plain encode + atempo")
        chain = f"atempo={speedup}" if speedup_filter else "anull"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
             "-af", chain,
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
