"""TTS chunking + billing-resilience tests against a fake ElevenLabs client.

Covers the money-shaped failure modes:
  - oversized single turn split under the v3 request limit (was API 400)
  - transient 5xx retried instead of failing the whole episode
  - completed (billed) chunks cached and reused on re-run — no double-billing
"""
from __future__ import annotations
import subprocess
import sys
import types

import pytest

import tts
from tts import _chunk_turns, _split_long_turn, V3_MAX_CHARS_PER_REQUEST


def _sentence_block(n_chars: int) -> str:
    s = "The market did a thing today and everyone had opinions about it. "
    out = ""
    while len(out) < n_chars:
        out += s
    return out.strip()


# ─── chunking: oversized turn splitting ─────────────────────────────────────


def test_chunk_turns_splits_single_oversized_turn():
    text = _sentence_block(3000)
    chunks = _chunk_turns([("ALEX", text)])
    assert len(chunks) >= 2
    for chunk in chunks:
        assert sum(len(t) for _, t in chunk) <= V3_MAX_CHARS_PER_REQUEST
        assert all(n == "ALEX" for n, _ in chunk)
    # No words lost across the split
    rejoined = " ".join(t for chunk in chunks for _, t in chunk)
    assert rejoined.split() == text.split()


def test_split_long_turn_prefers_sentence_boundaries():
    text = _sentence_block(3000)
    pieces = _split_long_turn(text, V3_MAX_CHARS_PER_REQUEST)
    assert all(len(p) <= V3_MAX_CHARS_PER_REQUEST for p in pieces)
    # Every piece ends on a sentence boundary (source is all full sentences)
    assert all(p.rstrip().endswith(".") for p in pieces)


def test_split_long_turn_handles_pathological_unbroken_sentence():
    text = "word " * 800  # 4000 chars, no sentence punctuation
    pieces = _split_long_turn(text.strip(), V3_MAX_CHARS_PER_REQUEST)
    assert all(len(p) <= V3_MAX_CHARS_PER_REQUEST for p in pieces)
    assert " ".join(pieces).split() == text.split()


# ─── fake ElevenLabs client ─────────────────────────────────────────────────


class _FakeApiError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"fake api error {status_code}")


class _FakeDialogue:
    """Stands in for client.text_to_dialogue. fail_plan maps 1-based call
    numbers to exceptions raised (once) on that call."""

    def __init__(self, mp3_bytes: bytes, fail_plan: dict | None = None):
        self.mp3_bytes = mp3_bytes
        self.fail_plan = dict(fail_plan or {})
        self.calls: list[str] = []  # request text prefixes, for assertions

    def convert(self, inputs=None, **kwargs):
        text = " ".join(
            (i["text"] if isinstance(i, dict) else i.text) for i in (inputs or [])
        )
        self.calls.append(text[:60])
        n = len(self.calls)
        if n in self.fail_plan:
            raise self.fail_plan.pop(n)
        return iter([self.mp3_bytes])


def _install_fake_eleven(monkeypatch, dialogue: _FakeDialogue) -> None:
    client = types.SimpleNamespace(text_to_dialogue=dialogue)
    mod_client = types.ModuleType("elevenlabs.client")
    mod_client.ElevenLabs = lambda api_key=None: client
    mod_root = types.ModuleType("elevenlabs")
    mod_root.client = mod_client
    monkeypatch.setitem(sys.modules, "elevenlabs", mod_root)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", mod_client)
    # Bare types module: DialogueInput import fails → dict-shaped inputs path
    monkeypatch.setitem(sys.modules, "elevenlabs.types", types.ModuleType("elevenlabs.types"))


@pytest.fixture(scope="module")
def tiny_mp3_bytes(tmp_path_factory) -> bytes:
    p = tmp_path_factory.mktemp("audio") / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=0.3",
         "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k",
         "-f", "mp3", str(p)],
        check=True,
    )
    return p.read_bytes()


@pytest.fixture
def eleven_env(monkeypatch, tmp_path):
    """Point the chunk cache at a throwaway ROOT and enable the fake key."""
    monkeypatch.setattr(tts, "ROOT", tmp_path)
    monkeypatch.setattr(tts, "ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)  # no real backoff waits
    return tmp_path


# ─── retry + chunk-cache behavior ───────────────────────────────────────────


def test_transient_5xx_is_retried_and_run_completes(eleven_env, monkeypatch, tiny_mp3_bytes):
    dlg = _FakeDialogue(tiny_mp3_bytes, fail_plan={1: _FakeApiError(502)})
    _install_fake_eleven(monkeypatch, dlg)
    out = eleven_env / "2026-07-30.mp3.tmp"
    mp3, timings = tts._synth_eleven_v3(
        [("JAMIE", "Markets were green today and nobody could tell you why.")], out
    )
    assert out.exists() and out.stat().st_size > 0
    assert len(dlg.calls) == 2  # failed once on 502, retried once, succeeded
    assert len(timings) == 1 and timings[0]["end_sec"] > 0


def test_non_retryable_error_propagates_immediately(eleven_env, monkeypatch, tiny_mp3_bytes):
    dlg = _FakeDialogue(tiny_mp3_bytes, fail_plan={1: _FakeApiError(401)})
    _install_fake_eleven(monkeypatch, dlg)
    out = eleven_env / "2026-07-30.mp3.tmp"
    with pytest.raises(_FakeApiError):
        tts._synth_eleven_v3([("JAMIE", "This call is doomed.")], out)
    assert len(dlg.calls) == 1  # no retries on auth-shaped errors


def test_failed_run_caches_billed_chunks_and_rerun_skips_them(eleven_env, monkeypatch, tiny_mp3_bytes):
    # Two ~1200-char turns → two chunks. Chunk 2 dies with a non-retryable
    # error on the first run; chunk 1 was already billed.
    turns = [("JAMIE", _sentence_block(1200)), ("ALEX", _sentence_block(1200))]
    dlg = _FakeDialogue(tiny_mp3_bytes, fail_plan={2: _FakeApiError(401)})
    _install_fake_eleven(monkeypatch, dlg)
    out = eleven_env / "2026-07-30.mp3.tmp"

    with pytest.raises(_FakeApiError):
        tts._synth_eleven_v3(turns, out)
    cache = tts._chunk_cache_dir(out)
    assert len(list(cache.glob("*.mp3"))) == 1  # billed chunk survived the crash
    assert not list(cache.glob("*.part"))       # no truncated cache entries

    # Re-run: chunk 1 comes from cache; only chunk 2 hits the API again.
    mp3, timings = tts._synth_eleven_v3(turns, out)
    assert out.exists() and out.stat().st_size > 0
    assert len(timings) == 2
    assert len(dlg.calls) == 3  # 2 calls first run + 1 on re-run, NOT 4
    # The re-run's only API call was for the second chunk's text.
    assert dlg.calls[2] == dlg.calls[1]
    assert len(list(cache.glob("*.mp3"))) == 2


def test_retryable_error_classifier():
    assert tts._is_retryable_tts_error(_FakeApiError(502))
    assert tts._is_retryable_tts_error(_FakeApiError(429))
    assert not tts._is_retryable_tts_error(_FakeApiError(401))
    assert not tts._is_retryable_tts_error(_FakeApiError(400))
    assert tts._is_retryable_tts_error(ConnectionError("reset"))
    assert not tts._is_retryable_tts_error(ValueError("nope"))
