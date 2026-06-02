"""Tests for the contract pipeline (schemas.py + generate_script structured
output). These lock in the structural guarantees that replaced the old
regex-repair layer — each test maps to a real "weird event" from the
2026-05-09/05-10 transcripts.
"""
import generate_script as g
import schemas
import stage_pipeline as sp


# ─────────── plan schema: hallucinated ids impossible ───────────

def test_plan_schema_constrains_story_ids_to_enum():
    schema = schemas.build_plan_schema(["c0", "c1", "c2"])
    bad = {
        "cold_open": {"hook": "x" * 10},
        "markets": {"lead_host": "ALEX", "key_numbers": ["a"], "macro_note": "m"},
        "big_story": {"story_id": "NOPE", "lead_host": "ALEX",
                      "story_title": "t", "angle": "a", "depth_turns": 6},
        "quick_hits": [{"story_id": "c1", "lead_host": "MAYA",
                        "angle": "a", "conviction": "real"}] * 4,
        "odd_thing": {"story_id": "c2", "joke_angle": "j"},
        "yesterday_callback": {"use": False, "topic": "", "fresh_take": ""},
        "sign_off": {"callback_target": "x"},
    }
    errs = g._schema_errors(bad, schema)
    assert any("NOPE" in e for e in errs)


def test_cold_open_has_no_story_id_field():
    # Orphan-teaser fix: cold_open can't name its own story, so it can't orphan one.
    schema = schemas.build_plan_schema(["c0"])
    assert "story_id" not in schema["properties"]["cold_open"]["properties"]


# ─────────── plan validator: dup beats + cross-day ───────────

def test_validate_plan_flags_duplicate_story_across_beats():
    outline = {
        "cold_open": {"hook": "Nvidia up ten percent today"},
        "big_story": {"story_id": "c0"},
        "odd_thing": {"story_id": "c0"},  # same as big_story
        "quick_hits": [{"story_id": x} for x in ("c1", "c2", "c3", "c4")],
        "yesterday_callback": {"use": False, "topic": "", "fresh_take": ""},
    }
    idx = {f"c{i}": {"title": f"Story {i}"} for i in range(5)}
    v = schemas.validate_plan(outline, idx)
    assert any("same" in x or "different story" in x for x in v)


def test_validate_plan_flags_cross_day_repeat():
    outline = {
        "cold_open": {"hook": "Parker bankruptcy fintech collapse"},
        "big_story": {"story_id": "c0"},
        "odd_thing": {"story_id": "c1"},
        "quick_hits": [{"story_id": x} for x in ("c2", "c3", "c4", "c5")],
        "yesterday_callback": {"use": False, "topic": "", "fresh_take": ""},
    }
    idx = {"c0": {"title": "Parker fintech startup bankruptcy filing"}}
    idx.update({f"c{i}": {"title": f"Other story {i}"} for i in range(1, 6)})
    yesterday = [schemas.story_signature("Parker fintech startup files bankruptcy")]
    v = schemas.validate_plan(outline, idx, yesterday)
    assert any("last few days" in x for x in v)


# ─────────── turns validator: third-person + consecutive + inline tag ───────────

def test_validate_turns_flags_self_third_person():
    # 2026-05-09: "…and Jamie's here to tell you" inside a JAMIE line.
    v = schemas.validate_turns({"turns": [
        {"speaker": "JAMIE", "text": "Parker collapsed and Jamie has the details"},
    ]})
    assert any("themselves" in x for x in v)


def test_validate_turns_flags_consecutive_speaker():
    v = schemas.validate_turns({"turns": [
        {"speaker": "ALEX", "text": "one"},
        {"speaker": "ALEX", "text": "two"},
    ]})
    assert any("twice in a row" in x for x in v)


def test_validate_turns_flags_inline_tag():
    v = schemas.validate_turns({"turns": [
        {"speaker": "MAYA", "text": "[rushed] markets ripping", "tag": ""},
    ]})
    assert any("inline" in x for x in v)


# ─────────── turns → text rendering ───────────

def test_turns_to_text_single_tag_and_drops_bad_speaker():
    turns = [
        {"speaker": "JAMIE", "text": "Nvidia up", "tag": "[excited]"},
        {"speaker": "BADGUY", "text": "ignored"},
        {"speaker": "ALEX", "text": "[deadpan] yields fell", "tag": ""},
    ]
    out = sp._turns_to_text(turns)
    lines = out.splitlines()
    assert lines == ["JAMIE: [excited] Nvidia up", "ALEX: yields fell"]
    # no stacked tags anywhere
    import re
    assert all(len(re.findall(r"\[[^\]]+\]", ln)) <= 1 for ln in lines)


# ─────────── structured re-prompt loop ───────────

def test_llm_json_reprompts_until_valid(monkeypatch):
    monkeypatch.setattr(g, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(g, "GROQ_API_KEY", "")
    state = {"n": 0}

    def fake(prompt, schema, model, temperature):
        state["n"] += 1
        if state["n"] == 1:
            return {"turns": [{"speaker": "ZZZ", "text": "bad"}]}  # invalid enum + too few
        return {"turns": [
            {"speaker": "JAMIE", "text": "fact one", "tag": "[curious]"},
            {"speaker": "ALEX", "text": "fact two", "tag": ""},
        ]}

    monkeypatch.setattr(g, "_ollama_json", fake)
    out = g._llm_json("p", schemas.build_turns_schema(2, 4),
                      extra_violations=schemas.validate_turns)
    assert state["n"] == 2
    assert len(out["turns"]) == 2


# ─────────── budget pacing: multistage scales beats to char budget ───────────

def _preset(min_turns):
    return {"preferences": {"_dynamic_preset": {"min_turns": min_turns}}}


def test_budget_scale_full_when_no_preset():
    # No preset (restricted key / local run) → no scaling, behavior unchanged.
    assert sp._budget_scale(None) == 1.0
    assert sp._budget_scale({}) == 1.0
    assert sp._budget_scale({"preferences": {}}) == 1.0


def test_budget_scale_full_when_flush():
    # Flush budget → paced min_turns at/above nominal → clamp to 1.0.
    assert sp._budget_scale(_preset(sp._NOMINAL_FULL_TURNS)) == 1.0
    assert sp._budget_scale(_preset(40)) == 1.0


def test_budget_scale_tightens_when_low():
    # Tight budget → smaller paced min_turns → scale shrinks, floored at 0.5.
    half = sp._budget_scale(_preset(sp._NOMINAL_FULL_TURNS // 2))
    assert 0.45 < half <= 0.55
    assert sp._budget_scale(_preset(1)) == sp._MIN_BUDGET_SCALE  # floor holds


def test_scale_band_floors_and_orders():
    # Bands never render zero turns and high stays ≥ low.
    assert sp._scale_band(4, 6, 1.0) == (4, 6)
    assert sp._scale_band(4, 6, 0.5, floor=2) == (2, 3)
    lo, hi = sp._scale_band(3, 4, sp._MIN_BUDGET_SCALE, floor=2)
    assert lo >= 2 and hi >= lo


def test_quick_hits_band_floor_covers_every_story():
    # Even at the tightest scale, the floor keeps ≥1 turn per planned story.
    n = 4
    lo, hi = sp._scale_band(max(6, n * 2), n * 3, sp._MIN_BUDGET_SCALE, floor=n)
    assert lo >= n and hi >= lo
