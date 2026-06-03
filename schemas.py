"""Generation contracts for the script pipeline.

This module is the single source of truth for what a valid plan and a valid
rendered beat look like. It replaces the old approach — generate free text,
then repair it with regex/token heuristics — with two complementary defences:

  1. SCHEMAS  (build_plan_schema / build_turns_schema): JSON schemas handed to
     the model's structured-output mode. The decoder physically cannot emit a
     speaker outside the cast, a story_id outside the ranked set, or a stacked
     emotion tag. Whole classes of the old "weird events" become unspeakable.

  2. VALIDATORS (validate_plan / validate_turns): pure functions that catch the
     SEMANTIC constraints a JSON schema cannot express — cross-beat / cross-day
     topic uniqueness, teaser→payoff coverage. A violation re-prompts the model
     with the specific failure (see stage_pipeline) rather than silently
     mutating the output.

Design notes:
- cold_open has NO story_id. It teases big_story by construction, so an
  "orphan teaser" (a cold open naming a topic the show never covers) is
  structurally impossible — there is no separate cold-open story to orphan.
- Story identity across days is matched on a SIGNATURE (substantive token set),
  not the cluster id. Cluster ids churn day-over-day as the lead source shifts;
  signatures are stable. See state.py.
"""
from __future__ import annotations
import re

from config import CHARACTERS

# ─────────── tags ───────────
# The union of every audio tag any host is allowed to use. The turns schema
# constrains the per-turn `tag` field to this enum (plus the empty string),
# which makes a stacked "[rushed] [laughs]" impossible: a turn carries at most
# one tag, chosen from a closed set.
ALLOWED_TAGS: list[str] = sorted({
    t for c in CHARACTERS.values() for t in c.get("tags", [])
})

SPEAKERS: list[str] = list(CHARACTERS.keys())

CONVICTIONS = ["real", "hype", "noise"]

# ─────────── story signatures (cross-day identity) ───────────
# Lowercased ≥5-letter tokens minus common news filler. Two stories with a
# large token overlap are "the same story" for cross-day suppression.
_SIGNATURE_STOP = {
    "about", "after", "again", "against", "their", "there", "these", "those",
    "today", "yesterday", "would", "could", "should", "where", "which",
    "while", "every", "still", "really", "stock", "stocks", "company",
    "companies", "market", "markets", "earnings", "report", "reports",
    "billion", "million", "trillion", "percent", "shares", "share",
    "actually", "basically", "first", "quarter", "first-quarter",
}


def story_signature(text: str) -> set[str]:
    """Substantive token set used to match the same story across days."""
    if not text:
        return set()
    return {
        t for t in re.findall(r"[A-Za-z]{5,}", text.lower())
        if t not in _SIGNATURE_STOP
    }


def signatures_overlap(a: set[str], b: set[str], min_shared: int = 3) -> bool:
    """True when two signatures share enough substance to be the same story."""
    return len(a & b) >= min_shared


# ─────────── n-gram overlap (content-novelty checks) ───────────

def _norm_words(text: str) -> list[str]:
    """Lowercased word tokens, audio tags + punctuation stripped."""
    text = re.sub(r"\[[^\]]+\]", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def ngram_overlap(text: str, prior: str, n: int = 6) -> str | None:
    """Return the first n-word phrase `text` shares with `prior`, else None.
    Used to flag a turn that recycles a phrase already spoken (cross-beat) or
    restates an earlier turn (intra-beat). n=6 catches verbatim/near-verbatim
    chunks while letting ordinary English collocations pass."""
    a, b = _norm_words(text), _norm_words(prior)
    if len(a) < n or len(b) < n:
        return None
    b_grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    for i in range(len(a) - n + 1):
        g = tuple(a[i:i + n])
        if g in b_grams:
            return " ".join(g)
    return None


# ─────────── plan schema ───────────

def build_plan_schema(story_ids: list[str]) -> dict:
    """JSON schema for the episode outline.

    `story_ids` is the closed set of cluster ids the planner may reference.
    Constraining each story_id to this enum eliminates the hallucinated-id
    failure class at the decode level — no downstream id-repair needed.

    Note: cross-beat uniqueness (big_story ≠ quick_hits ≠ odd_thing) and
    cross-day suppression are NOT expressible here; validate_plan() handles
    them and the caller re-prompts on violation.
    """
    story_id = {"type": "string", "enum": story_ids}
    host = {"type": "string", "enum": SPEAKERS}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "cold_open", "markets", "big_story", "quick_hits",
            "odd_thing", "yesterday_callback", "sign_off",
        ],
        "properties": {
            "cold_open": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hook"],
                "properties": {
                    # No story_id: the cold open teases big_story by construction.
                    "hook": {"type": "string", "minLength": 8, "maxLength": 200},
                },
            },
            "markets": {
                "type": "object",
                "additionalProperties": False,
                "required": ["lead_host", "key_numbers", "macro_note"],
                "properties": {
                    "lead_host": host,
                    "key_numbers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1, "maxItems": 6,
                    },
                    "macro_note": {"type": "string"},
                },
            },
            "big_story": {
                "type": "object",
                "additionalProperties": False,
                "required": ["story_id", "lead_host", "story_title", "angle", "depth_turns"],
                "properties": {
                    "story_id": story_id,
                    "lead_host": host,
                    "story_title": {"type": "string"},
                    "angle": {"type": "string"},
                    "depth_turns": {"type": "integer", "minimum": 4, "maximum": 8},
                },
            },
            "quick_hits": {
                "type": "array",
                "minItems": 4, "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["story_id", "lead_host", "angle", "conviction"],
                    "properties": {
                        "story_id": story_id,
                        "lead_host": host,
                        "angle": {"type": "string"},
                        "conviction": {"type": "string", "enum": CONVICTIONS},
                    },
                },
            },
            "odd_thing": {
                "type": "object",
                "additionalProperties": False,
                "required": ["story_id", "joke_angle"],
                "properties": {
                    "story_id": story_id,
                    "joke_angle": {"type": "string"},
                },
            },
            "yesterday_callback": {
                "type": "object",
                "additionalProperties": False,
                "required": ["use", "topic", "fresh_take"],
                "properties": {
                    "use": {"type": "boolean"},
                    "topic": {"type": "string"},
                    "fresh_take": {"type": "string"},
                },
            },
            "sign_off": {
                "type": "object",
                "additionalProperties": False,
                "required": ["callback_target"],
                "properties": {
                    "callback_target": {"type": "string"},
                },
            },
        },
    }


# ─────────── turns schema ───────────

def build_turns_schema(min_turns: int, max_turns: int,
                       speakers: list[str] | None = None) -> dict:
    """JSON schema for a rendered beat: an array of dialogue turns.

    Each turn is {speaker, text, tag?}. `speaker` is a closed enum, so a turn
    can never be attributed outside the cast and a host can never appear "as
    text" (the third-person-self-reference failure: JAMIE is a field value, not
    prose the model writes about). `tag` is at most one value from ALLOWED_TAGS,
    so stacked tags are impossible. Tags belong in this field, never inline in
    `text`.
    """
    spk = speakers or SPEAKERS
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["turns"],
        "properties": {
            "turns": {
                "type": "array",
                "minItems": min_turns,
                "maxItems": max_turns,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["speaker", "text"],
                    "properties": {
                        "speaker": {"type": "string", "enum": spk},
                        "text": {"type": "string", "minLength": 1},
                        "tag": {"type": "string", "enum": [""] + ALLOWED_TAGS},
                    },
                },
            },
        },
    }


# ─────────── plan validator (semantic, post-decode) ───────────

def _beat_story_ids(outline: dict) -> dict[str, str]:
    """Map of beat-label -> story_id for the beats that reference one."""
    out: dict[str, str] = {}
    bs = (outline.get("big_story") or {}).get("story_id")
    if bs:
        out["big_story"] = bs
    ot = (outline.get("odd_thing") or {}).get("story_id")
    if ot:
        out["odd_thing"] = ot
    for i, qh in enumerate(outline.get("quick_hits") or []):
        sid = qh.get("story_id")
        if sid:
            out[f"quick_hits[{i}]"] = sid
    return out


def validate_plan(outline: dict, ranked_idx: dict[str, dict],
                  yesterday_sigs: list[set[str]] | None = None) -> list[str]:
    """Return a list of human-readable constraint violations. Empty == valid.

    These are the checks JSON-schema cannot express. The caller feeds each
    violation back to the planner as a targeted re-prompt.
    """
    violations: list[str] = []
    beats = _beat_story_ids(outline)

    # 1. Cross-beat uniqueness: every beat covers a DIFFERENT story.
    seen: dict[str, str] = {}
    for label, sid in beats.items():
        if sid in seen:
            violations.append(
                f"story_id {sid} is used by both {seen[sid]} and {label}; "
                f"every beat must cover a different story."
            )
        else:
            seen[sid] = label

    # 2. Cross-day suppression: no beat may re-cover a story we ran in the
    #    last few days unless the planner explicitly flagged it as a
    #    yesterday_callback with a fresh take.
    if yesterday_sigs:
        cb = outline.get("yesterday_callback") or {}
        callback_topic = cb.get("topic", "") if cb.get("use") else ""
        callback_sig = story_signature(callback_topic)
        for label, sid in beats.items():
            cluster = ranked_idx.get(sid)
            if not cluster:
                continue
            sig = story_signature(cluster.get("title", ""))
            for ysig in yesterday_sigs:
                if signatures_overlap(sig, ysig):
                    # Allowed only if THIS beat is the declared callback.
                    if callback_sig and signatures_overlap(sig, callback_sig):
                        break
                    violations.append(
                        f"{label} (story {sid}) repeats a story already covered "
                        f"in the last few days; pick a fresh story or declare it "
                        f"as yesterday_callback with a genuinely new angle."
                    )
                    break

    # 2b. The yesterday_callback must not re-cover today's OWN lead. The plan
    #     prompt says set use=false when the callback repeats the big story, but
    #     nothing enforced it — so VS shipped as cold_open + callback + big_story
    #     (three hits on one story in the first 90s). Flag the overlap.
    cb = outline.get("yesterday_callback") or {}
    if cb.get("use") and cb.get("topic"):
        cb_sig = story_signature(cb["topic"])
        bs_cluster = ranked_idx.get((outline.get("big_story") or {}).get("story_id"))
        bs_sig = story_signature(bs_cluster.get("title", "")) if bs_cluster else set()
        hook_sig = story_signature((outline.get("cold_open") or {}).get("hook", ""))
        if (bs_sig and signatures_overlap(cb_sig, bs_sig, min_shared=2)) or \
           (hook_sig and signatures_overlap(cb_sig, hook_sig, min_shared=2)):
            violations.append(
                "yesterday_callback.topic repeats today's big_story / cold_open; "
                "set use=false or point the callback at a DIFFERENT continuing "
                "thread — don't cover the lead story three times."
            )

    # 3. Cold-open hook substance: must carry a number or a proper noun + fact,
    #    not be a bare question.
    hook = (outline.get("cold_open") or {}).get("hook", "")
    if hook:
        has_number = bool(re.search(r"\d", hook))
        has_proper_noun = bool(re.search(r"\b[A-Z][a-zA-Z]{2,}", hook))
        if hook.rstrip().endswith("?") and not (has_number or has_proper_noun):
            violations.append(
                "cold_open.hook is a bare question with no number or named "
                "entity; rewrite it to lead with a specific fact."
            )

    return violations


# ─────────── turns validator (semantic, post-decode) ───────────

def validate_turns(payload: dict, prev_text: str = "") -> list[str]:
    """Semantic checks on a rendered beat that the schema can't express:
    no host speaks twice in a row, no host refers to THEMSELVES in the third
    person, no inline tags, and — when `prev_text` (the turns from earlier
    beats) is supplied — no recycling of a phrase already spoken in an earlier
    beat or restating an earlier turn within THIS beat. Content novelty was the
    one constraint the contract never enforced; these checks make repetition a
    re-prompt instead of a shipped defect."""
    violations: list[str] = []
    turns = payload.get("turns") or []
    prev_speaker = None
    seen_so_far = prev_text or ""
    for i, turn in enumerate(turns):
        spk = turn.get("speaker", "")
        text = turn.get("text", "") or ""
        if spk and spk == prev_speaker:
            violations.append(
                f"turns[{i}]: {spk} speaks twice in a row; merge with the "
                f"previous turn or put a different host between them."
            )
        prev_speaker = spk
        # Self third-person: the speaker's own name appears in their own line.
        if spk and re.search(rf"\b{re.escape(spk.title())}\b", text, re.IGNORECASE):
            violations.append(
                f"turns[{i}]: {spk} refers to themselves ('{spk.title()}') in "
                f"their own line; speak in the first person."
            )
        # Inline audio tags belong in the `tag` field, not the text.
        if re.search(r"\[[^\]]+\]", text):
            violations.append(
                f"turns[{i}]: remove the inline [bracket] tag from text; put a "
                f"single emotion in the `tag` field instead."
            )
        # Content novelty: don't recycle a 6-word phrase from an earlier beat
        # or an earlier turn in this beat. seen_so_far accumulates as we go so a
        # later turn is checked against everything before it.
        if seen_so_far:
            dup = ngram_overlap(text, seen_so_far, n=6)
            if dup:
                violations.append(
                    f"turns[{i}]: repeats a phrase already spoken (\"{dup}…\"); "
                    f"the cold open / earlier turns already covered this — say "
                    f"something NEW (a different fact, number, or angle) or cut it."
                )
        seen_so_far = (seen_so_far + "\n" + text) if seen_so_far else text
    return violations
