"""Tests for the confirmed review-finding fixes (tasks 2, 7, 13, 14):

  - main._extract_yesterday_topics excludes today's own plan sidecar
  - lock: atomic acquisition, live holder never stolen, dead holder recovered
  - _llm_json/_llm_call: explicit gen/critic role, hard-vs-soft violation
    handling on exhausted retries; generate() retry loop deleted
  - Ollama num_ctx raised for the long multistage prompts
  - state: atomic save, last_covered refresh on re-coverage
  - validate_turns whitelists first-person self-identification
  - legacy plan repair assigns unused ids instead of story_ids[0] 3×
  - config env-with-default reads treat "" as unset
  - interests_loader logs YAML errors instead of swallowing them
"""
from __future__ import annotations
import inspect
import json
import os
import time
from datetime import datetime, timezone, timedelta

import pytest

import generate_script as g
import schemas
import stage_pipeline as sp
import state as state_mod


# ─── task 2: yesterday-topics must exclude today's own plan ─────────────────


def _write_plan(dirpath, date_str, hook):
    (dirpath / f"{date_str}.plan.json").write_text(json.dumps({
        "cold_open": {"hook": hook},
        "big_story": {"story_title": f"{hook} big story"},
        "quick_hits": [{"angle": f"{hook} quick hit"}],
    }))


def test_extract_yesterday_topics_skips_todays_plan(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "EPISODES_DIR", tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    _write_plan(tmp_path, yesterday, "yesterday hook")
    _write_plan(tmp_path, today, "today hook")
    topics = main._extract_yesterday_topics(today)
    assert topics and all("today hook" not in t for t in topics)
    assert any("yesterday hook" in t for t in topics)


def test_extract_yesterday_topics_empty_when_only_todays_plan(tmp_path, monkeypatch):
    # Mid-run retry after the sidecar is written: today's plan must NOT come
    # back as "yesterday" (it made the duplicate guard match itself).
    import main
    monkeypatch.setattr(main, "EPISODES_DIR", tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")
    _write_plan(tmp_path, today, "today hook")
    assert main._extract_yesterday_topics(today) == []


# ─── task 7: lock semantics ─────────────────────────────────────────────────


def _reload_lock(tmp_path, monkeypatch):
    monkeypatch.setattr("config.ROOT", tmp_path)
    import importlib
    import lock
    importlib.reload(lock)
    return lock


def test_lock_live_holder_never_stolen_even_when_old(tmp_path, monkeypatch):
    lock = _reload_lock(tmp_path, monkeypatch)
    # Live pid (our own), timestamp far past STALE_LOCK_SEC: previously stolen,
    # which let run A's finally-unlink delete run B's lock mid-flight.
    lock.LOCK_PATH.write_text(f"{os.getpid()}:{time.time() - 10 * lock.STALE_LOCK_SEC}")
    with pytest.raises(RuntimeError, match="another run is in progress"):
        with lock.acquire_lock():
            pass
    # The foreign lock must survive the failed acquisition attempt.
    assert lock.LOCK_PATH.read_text().startswith(f"{os.getpid()}:")


def test_lock_dead_holder_recovered(tmp_path, monkeypatch):
    lock = _reload_lock(tmp_path, monkeypatch)
    lock.LOCK_PATH.write_text(f"99999999:{time.time()}")  # dead pid, fresh ts
    with lock.acquire_lock():
        assert lock.LOCK_PATH.read_text().startswith(f"{os.getpid()}:")
    assert not lock.LOCK_PATH.exists()


def test_lock_acquisition_is_excl(tmp_path, monkeypatch):
    lock = _reload_lock(tmp_path, monkeypatch)
    with lock.acquire_lock():
        # A second acquire in the same process sees a live holder → raises.
        with pytest.raises(RuntimeError):
            with lock.acquire_lock():
                pass


# ─── task 13: role-aware dispatch ───────────────────────────────────────────


_VALID_TURNS = {"turns": [
    {"speaker": "JAMIE", "text": "fact one", "tag": ""},
    {"speaker": "ALEX", "text": "fact two", "tag": ""},
]}


def test_llm_json_critic_role_uses_groq_critic_model(monkeypatch):
    monkeypatch.setattr(g, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(g, "GROQ_API_KEY", "k")
    monkeypatch.setattr(g, "GROQ_MODEL", "gen-model")
    monkeypatch.setattr(g, "GROQ_CRITIC_MODEL", "critic-model")
    seen = []

    def fake(prompt, schema, schema_name, model, temperature):
        seen.append(model)
        return dict(_VALID_TURNS)

    monkeypatch.setattr(g, "_groq_json", fake)
    schema = schemas.build_turns_schema(2, 4)
    g._llm_json("p", schema, role="critic")
    g._llm_json("p", schema)  # default role = gen
    assert seen == ["critic-model", "gen-model"]


def test_llm_call_critic_role_uses_anthropic_critic_model(monkeypatch):
    monkeypatch.setattr(g, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(g, "ANTHROPIC_MODEL", "gen-model")
    monkeypatch.setattr(g, "ANTHROPIC_CRITIC_MODEL", "critic-model")
    seen = []
    monkeypatch.setattr(g, "_anthropic_call",
                        lambda prompt, model, temperature: seen.append(model) or "ok")
    # Warm critic call: temperature no longer decides the model — role does.
    g._llm_call("p", "om", "gm", temperature=0.7, role="critic")
    g._llm_call("p", "om", "gm", temperature=0.1)  # cold gen call stays gen
    assert seen == ["critic-model", "gen-model"]


# ─── task 13: hard vs soft violations on exhausted retries ──────────────────


def test_llm_json_accepts_soft_residual_violation(monkeypatch):
    monkeypatch.setattr(g, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(g, "GROQ_API_KEY", "")
    monkeypatch.setattr(g, "_ollama_json",
                        lambda *a, **k: dict(_VALID_TURNS))
    out = g._llm_json("p", schemas.build_turns_schema(2, 4),
                      extra_violations=lambda o: ["soft: phrase echo"],
                      max_attempts=2)
    assert out["turns"]  # shipped despite the residual soft violation


def test_llm_json_raises_on_persistent_hard_violation(monkeypatch):
    monkeypatch.setattr(g, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(g, "GROQ_API_KEY", "")
    monkeypatch.setattr(g, "_ollama_json",
                        lambda *a, **k: dict(_VALID_TURNS))
    with pytest.raises(g.HardViolationError):
        g._llm_json("p", schemas.build_turns_schema(2, 4),
                    extra_violations=lambda o: [
                        schemas.HARD_PREFIX + "big_story and quick_hits[0] duplicate"],
                    max_attempts=2)


def test_plan_dup_violations_are_hard_tagged():
    outline = {
        "cold_open": {"hook": "Nvidia up ten percent today"},
        "big_story": {"story_id": "c0"},
        "odd_thing": {"story_id": "c0"},
        "quick_hits": [{"story_id": x} for x in ("c1", "c2", "c3", "c4")],
        "yesterday_callback": {"use": False, "topic": "", "fresh_take": ""},
    }
    idx = {f"c{i}": {"title": f"Story {i}"} for i in range(5)}
    v = schemas.validate_plan(outline, idx)
    assert v and all(x.startswith(schemas.HARD_PREFIX) for x in v)


def test_plan_falls_back_to_none_on_hard_violation(monkeypatch):
    # plan() must NOT drop to the unvalidated legacy text parse when the hard
    # contract can't be met — None → deterministic _fallback_plan upstream.
    def raising(*a, **k):
        raise g.HardViolationError("dup persists")
    monkeypatch.setattr(sp, "_llm_json", raising)
    called = {"legacy": False}
    monkeypatch.setattr(sp, "_llm_call",
                        lambda *a, **k: called.__setitem__("legacy", True) or "{}")
    ranked = [{"id": f"c{i}", "title": f"Story {i}", "sources": []} for i in range(6)]
    assert sp.plan(ranked, {"indices": []}) is None
    assert called["legacy"] is False


# ─── task 13: dead retry loop deleted, config constants gone ────────────────


def test_generate_has_no_max_retries_param():
    assert "max_retries" not in inspect.signature(g.generate).parameters


def test_length_constants_removed_from_config():
    import config
    for name in ("MIN_WORDS", "MAX_WORDS", "MIN_TURNS"):
        assert not hasattr(config, name)


# ─── task 13: Ollama context window fits the prev-turns block ───────────────


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


def test_ollama_calls_use_32k_ctx(monkeypatch):
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append(json)
        return _FakeResp({"response": "JAMIE: hi" if "format" not in json else "{}"})

    monkeypatch.setattr(g.requests, "post", fake_post)
    g._ollama_call("p", "m")
    g._ollama_json("p", {"type": "object"}, "m", 0.4)
    assert all(j["options"]["num_ctx"] == 32768 for j in captured)


# ─── task 14: state — atomic save + last_covered refresh ────────────────────


def test_save_state_atomic(tmp_path, monkeypatch):
    target = tmp_path / ".state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", target)
    state_mod.save_state({"covered": []})
    assert json.loads(target.read_text()) == {"covered": []}
    assert not target.with_suffix(".json.tmp").exists()  # tmp replaced, not left


def test_mark_covered_refreshes_last_covered_on_recoverage():
    old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    state = {"covered": [{"cluster_id": "a", "first_covered": old}]}
    # Day 3: outside the 2-day window when keyed on first coverage…
    assert state_mod.covered_within(state, days=2) == set()
    # …but re-covering it today must re-arm suppression, not append a dup.
    state_mod.mark_covered(state, ["a"])
    assert state_mod.covered_within(state, days=2) == {"a"}
    assert len(state["covered"]) == 1
    assert state["covered"][0]["first_covered"] == old  # first coverage kept


def test_covered_within_falls_back_to_first_covered():
    # Old state files (pre-last_covered) keep working.
    fresh = datetime.now(timezone.utc).isoformat()
    state = {"covered": [{"cluster_id": "b", "first_covered": fresh}]}
    assert state_mod.covered_within(state, days=2) == {"b"}


def test_prune_keeps_recovered_continuing_story():
    ancient = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    state = {"covered": [{"cluster_id": "c", "first_covered": ancient}]}
    state_mod.mark_covered(state, ["c"])  # still in the news → refreshed
    assert any(e["cluster_id"] == "c" for e in state["covered"])


# ─── task 14: self-identification whitelist in validate_turns ───────────────


def test_validate_turns_allows_first_person_self_identification():
    v = schemas.validate_turns({"turns": [
        {"speaker": "JAMIE", "text": "I'm Jamie — Nvidia dropped nine percent overnight."},
        {"speaker": "ALEX", "text": "Alex here with the tape: S&P up half a percent."},
    ]})
    assert not any("themselves" in x for x in v)


def test_validate_turns_still_flags_true_third_person():
    v = schemas.validate_turns({"turns": [
        {"speaker": "JAMIE", "text": "Parker collapsed and Jamie has the details"},
    ]})
    assert any("themselves" in x for x in v)


# ─── task 14: legacy plan repair uses unused ids, drops when exhausted ──────


def test_repair_invalid_story_ids_assigns_distinct_unused():
    idx = {f"c{i}": {"title": f"Story {i}"} for i in range(6)}
    story_ids = [f"c{i}" for i in range(6)]
    outline = {
        "big_story": {"story_id": "BAD1"},
        "odd_thing": {"story_id": "c5"},
        "quick_hits": [{"story_id": "BAD2"}, {"story_id": "c1"},
                       {"story_id": "BAD3"}, {"story_id": "c2"}],
    }
    out = sp._repair_invalid_story_ids(outline, idx, story_ids)
    ids = [out["big_story"]["story_id"], out["odd_thing"]["story_id"]] + \
          [q["story_id"] for q in out["quick_hits"]]
    assert all(i in idx for i in ids)
    assert len(set(ids)) == len(ids)  # no beat shares a story anymore


def test_repair_invalid_story_ids_drops_when_exhausted():
    idx = {"c0": {"title": "Story 0"}}
    outline = {
        "big_story": {"story_id": "c0"},
        "quick_hits": [{"story_id": "BAD1"}, {"story_id": "BAD2"}],
    }
    out = sp._repair_invalid_story_ids(outline, idx, ["c0"])
    # c0 is taken by big_story; no unused ids remain → invalid beats dropped.
    assert out["big_story"]["story_id"] == "c0"
    assert out["quick_hits"] == []


# ─── task 14: config treats empty env strings as unset ──────────────────────


def test_config_empty_env_voice_id_falls_back(monkeypatch):
    import importlib
    import config
    monkeypatch.setenv("ELEVEN_VOICE_JAMIE", "")   # unset GitHub secret
    monkeypatch.setenv("GROQ_MODEL", "")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "")
    try:
        importlib.reload(config)
        assert config.ELEVEN_CHARACTER_VOICES["JAMIE"] == "EXAVITQu4vr4xnSDxMaL"
        assert config.GROQ_MODEL == "llama-3.3-70b-versatile"
        assert config.OLLAMA_TIMEOUT == 1800
    finally:
        monkeypatch.delenv("ELEVEN_VOICE_JAMIE")
        monkeypatch.delenv("GROQ_MODEL")
        monkeypatch.delenv("OLLAMA_TIMEOUT")
        importlib.reload(config)


# ─── task 14: interests_loader logs the parse error ─────────────────────────


def test_interests_loader_prints_error_on_bad_yaml(tmp_path, monkeypatch, capsys):
    import interests_loader
    (tmp_path / "interests.yaml").write_text("tone: [unclosed\n  broken: {")
    monkeypatch.setattr(interests_loader, "ROOT", tmp_path)
    assert interests_loader.load_interests() == {}
    out = capsys.readouterr().out
    assert "[interests]" in out and "using defaults" in out
