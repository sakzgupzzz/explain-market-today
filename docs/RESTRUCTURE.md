# Pipeline restructure — contract generation + voice (2026-05)

## Why

The 2026-05-09/05-10 transcripts had recurring "weird events":

| Failure | Example |
|---|---|
| Dead teaser | Cold open named CVS; show never covered it (big story = Makary). |
| Duplicate segment | Nvidia covered twice (05-10); NRG twice (05-09). |
| Stale market data | S&P +0.84 / Nasdaq +1.71 / AMD +11.44 identical two days running. |
| Speaker self-reference | "…and Jamie's here to tell you" *inside* a JAMIE line. |
| Stacked emotion tags | `[rushed] [laughs]`, `[deadpan] … [deadpan]`. |
| Cross-day bleed | Parker bankruptcy ran as cold open on consecutive days. |

Root cause: every failure had spawned a **post-hoc regex/token "repair"** patch in
`stage_pipeline.py` (~270 lines of in-flight diff). Symptom-patching, not a
structure fix — and repair can't fix *semantic* duplication (two valid clusters
that are the same story).

## Research (three deep-research passes, adversarially verified)

**Architecture** (PodAgent ACL'25, PodBench Jan'26, DOME, open-notebooklm, MARCH,
DeepMind ICLR'24, Snorkel): planning improves *structural adherence* specifically;
schema-constrained decoding eliminates malformation at the decode level; a
memory/coverage contract is the structured anti-dup/anti-orphan mechanism. Critic
loops are **not** a free win — unconditional self-critique *corrodes* correct
output (Snorkel 98→57%). A critic must be **blinded** (MARCH info-asymmetry) and
**confidence-gated**.

**Voice — ElevenLabs** (vendor docs, one peer-reviewed F0 study): expressiveness
is v3's whole advance over v2 and is driven by the **script** (audio tags +
punctuation), not voice settings. Stock voices read "generic AI"; IVC/designed
voices recommended for v3. Flat F0 = the #1 robotic tell.

**Voice — free/local** (arXiv 2508.04179 InterSpeech'25, neutral Human-Fooling-Rate
benchmark): no open model matches ElevenLabs yet. ElevenLabs 69.85% / PlayHT 71.49%
≈ human 70.68%; best open (F5-TTS) only 50.26% — a ~20pt gap. Best *free/local*
for this stack: **Kokoro-82M** (Apache-2.0, MLX-native Apple Silicon, 54 voices)
and **Chatterbox** (MIT, zero-shot cloning + emotion). Flow-matching models
(F5/E2) collapse toward over-smoothed mean prosody → flatter delivery.

## What changed

Constraints moved from post-hoc repair → into the generation contract.

- **`schemas.py` (new)** — single source of truth. `build_plan_schema` (story_id
  enum, host enum, quick_hits==4), `build_turns_schema` (speaker enum, one
  optional `tag` from a closed set). `validate_plan` / `validate_turns` =
  pure-function semantic checks. `story_signature` = stable cross-day identity
  (token set, not cluster id).
- **`generate_script._llm_json`** — structured-output dispatch
  (Anthropic tool-use / Groq json_schema / Ollama format), with a local schema +
  semantic re-prompt loop (`_schema_errors`). No new dependency.
- **`stage_pipeline.plan`** — structured path; cold_open has **no** story_id
  (teases big_story by construction → orphan teaser impossible). Re-prompts on
  violation instead of mutating. **Deleted**: id-repair, cold_open==big_story
  re-pointing, token-overlap cross-day swap, yesterday_callback drop heuristic.
- **`stage_pipeline._render_beat`** — emits structured turns →
  `_turns_to_text` (`NAME: [tag] text`). **Deleted**: `_strip_jamie_third_person`
  + the Jamie regex. Speaker is a field; third-person self-reference + stacked
  tags are structurally impossible.
- **`_semantic_dup_violations`** — blinded, heuristic-gated critic at plan time:
  only title pairs whose signatures overlap reach an LLM that sees *titles only*
  (no planner rationale) and judges same-story. Folds into the re-prompt loop.
- **Stale market data** — kept the in-flight `is_stale`/`as_of` work; it's a
  first-class planner + markets-beat input.
- **Voice** — `_synth_eleven_v3` now passes `ModelSettings(stability=…)`
  (`ELEVEN_V3_STABILITY`, default 0.5 Natural). v2-fallback model bumped to
  `eleven_turbo_v2_5`. The single inline `[tag]` per line is exactly v3's format.

Legacy single-shot path + text fallbacks retained behind existing flags for
rollback. Coverage: `tests/test_contract.py`.

## Open / deferred

- **Voices are still stock** (Sarah/Brian/Jessica) — biggest remaining "AI tell".
  Action: create Instant Voice Clones (needs reference audio) or pick
  less-common library IDs; set `ELEVEN_VOICE_{JAMIE,ALEX,MAYA}`.
- **Free/local migration** (Kokoro / Chatterbox) — viable but ~20pt realism gap;
  evaluate only if leaving paid ElevenLabs.
- **Local Ollama** constrained-decode fidelity for the plan schema is untested;
  Anthropic/Groq are the reliable structured paths.
- Research is mostly vendor + adjacent-domain — treat the first week of episodes
  as an A/B eval against the failures above.
