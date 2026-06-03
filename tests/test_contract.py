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


# ─────────── deterministic fallback plan (plan-stage failure ≠ crash) ───────────

def _fake_ranked(n=6):
    return [
        {"id": f"c{i}", "title": f"Story {i} about Acme Corp earnings beat",
         "summary": f"Summary {i}", "sources": ["reuters.com"]}
        for i in range(n)
    ]


def _fake_market():
    return {"indices": [{"name": "S&P 500", "symbol": "^GSPC", "close": 5200.0, "pct": 1.2},
                        {"name": "Nasdaq", "symbol": "^IXIC", "close": 16300.0, "pct": -0.4}],
            "gainers": [{"name": "Nvidia", "symbol": "NVDA", "close": 120.0, "pct": 5.1}],
            "losers": [{"name": "Tesla", "symbol": "TSLA", "close": 170.0, "pct": -3.2}],
            "macro": []}


def test_fallback_plan_is_schema_valid():
    ranked = _fake_ranked(6)
    out = sp._fallback_plan(ranked, _fake_market())
    schema = schemas.build_plan_schema([c["id"] for c in ranked])
    assert g._schema_errors(out, schema) == []


def test_fallback_plan_passes_validate_plan():
    ranked = _fake_ranked(6)
    out = sp._fallback_plan(ranked, _fake_market())
    assert schemas.validate_plan(out, sp._ranked_index(ranked)) == []


def test_fallback_plan_distinct_beats_when_enough_stories():
    ranked = _fake_ranked(6)
    out = sp._fallback_plan(ranked, _fake_market())
    ids = [out["big_story"]["story_id"]] + \
          [q["story_id"] for q in out["quick_hits"]] + \
          [out["odd_thing"]["story_id"]]
    assert len(set(ids)) == len(ids)  # all 6 beats cover a different story


def test_fallback_plan_none_when_no_stories():
    assert sp._fallback_plan([], {}) is None
    assert sp._fallback_plan([{"title": "no id"}], {}) is None


# ─────────── anti-repetition contract (content novelty) ───────────

def test_ngram_overlap_detects_shared_phrase():
    a = "Victoria's Secret jumped forty percent yesterday after a big earnings beat"
    b = "So Victoria's Secret jumped forty percent yesterday — turns out the punching bag"
    assert schemas.ngram_overlap(b, a, n=6)  # shares 6+ word run
    assert schemas.ngram_overlap("totally unrelated sentence about gold and oil prices",
                                 a, n=6) is None


def test_validate_turns_flags_cross_beat_echo():
    prev = "JAMIE: Victoria's Secret jumped forty percent yesterday after a huge earnings beat."
    payload = {"turns": [
        {"speaker": "ALEX", "text": "Victoria's Secret jumped forty percent yesterday after a huge earnings beat, and here's why."},
        {"speaker": "MAYA", "text": "The cybersecurity names are a totally separate signal from retail."},
    ]}
    v = schemas.validate_turns(payload, prev_text=prev)
    assert any("repeats a phrase already spoken" in x for x in v)


def test_validate_turns_flags_intra_beat_repeat():
    payload = {"turns": [
        {"speaker": "ALEX", "text": "They killed the angel sizing and started selling to real customers."},
        {"speaker": "MAYA", "text": "Right, they killed the angel sizing and started selling to real customers indeed."},
    ]}
    v = schemas.validate_turns(payload)  # no prev_text → intra-beat only
    assert any("repeats a phrase already spoken" in x for x in v)


def test_validate_turns_clean_passes():
    payload = {"turns": [
        {"speaker": "ALEX", "text": "The S&P closed up a third of a percent on thin breadth."},
        {"speaker": "JAMIE", "text": "Meanwhile small caps quietly outran the megacaps for once."},
    ]}
    assert schemas.validate_turns(payload, prev_text="JAMIE: Totally different cold open about oil.") == []


def test_validate_plan_flags_callback_repeating_big_story():
    ranked = [{"id": "vsx", "title": "Victoria Secret shares surge forty percent turnaround"}]
    ranked += [{"id": f"c{i}", "title": f"Distinct unrelated headline number {i} about gold"}
               for i in range(1, 6)]
    outline = {
        "cold_open": {"hook": "Victoria Secret turnaround is the story today"},
        "big_story": {"story_id": "vsx"},
        "quick_hits": [{"story_id": ranked[i]["id"]} for i in (1, 2, 3, 4)],
        "odd_thing": {"story_id": ranked[5]["id"]},
        "yesterday_callback": {"use": True, "topic": "Victoria Secret turnaround surge continues"},
    }
    v = schemas.validate_plan(outline, sp._ranked_index(ranked))
    assert any("repeats today's big_story" in x for x in v)


def test_semantic_dup_critic_importable_no_nameerror():
    # Regression: stage_pipeline must import signatures_overlap, else
    # _semantic_dup_violations NameErrors and silently kills the structured
    # plan path (every run fell back to the unvalidated legacy parser).
    # Distinct titles → no suspect pairs → returns [] without any LLM call.
    ranked = [
        {"id": "a", "title": "Federal Reserve holds interest rates steady again"},
        {"id": "b", "title": "Nvidia unveils a brand new datacenter accelerator"},
        {"id": "c", "title": "Oil prices tumble on surprise inventory build"},
        {"id": "d", "title": "Retail sales disappoint across department stores"},
        {"id": "e", "title": "Bitcoin rallies past a fresh all-time record high"},
        {"id": "f", "title": "Airline strike grounds thousands of summer flights"},
    ]
    outline = {
        "big_story": {"story_id": "a"},
        "quick_hits": [{"story_id": x} for x in ("b", "c", "d", "e")],
        "odd_thing": {"story_id": "f"},
    }
    assert sp._semantic_dup_violations(outline, sp._ranked_index(ranked)) == []


def test_multistage_degrades_on_plan_failure(monkeypatch):
    # plan() returning None must NOT raise — it must fall back and still ship.
    ranked = _fake_ranked(6)
    monkeypatch.setattr(sp, "plan", lambda *a, **k: None)
    monkeypatch.setattr(sp, "_render_beat",
                        lambda name, *a, **k: f"JAMIE: {name} line.")
    monkeypatch.setattr(sp, "render_cold_open",
                        lambda *a, **k: "JAMIE: cold open line.")
    script = sp.generate_multistage(_fake_market(), ranked)
    assert script.strip()
    assert sp._LAST_OUTLINE is not None
