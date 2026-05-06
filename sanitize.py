"""Post-process LLM-generated dialogue scripts before TTS.

Deterministic guardrails for things the prompt can't reliably enforce on a 14B
local model: banned openers, wrong-name intros, parenthesized tickers,
JAMIE airtime cap. Idempotent — running twice yields the same output.
"""
from __future__ import annotations
import re
from collections import Counter
from config import CHARACTERS, DEFAULT_CHARACTER, DISCLAIMER_SHORT, BANNED_PHRASES

LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]{0,15}):\s*(.+)$")

# Cold-open phrases we want stripped from JAMIE's first line. Order matters —
# longer phrases first so they match before their substrings.
_BANNED_OPENERS = [
    re.compile(r"^\s*welcome\s+to\s+(?:the\s+show|your\s+daily[^.,!?]*|[^.,!?]*market\s+recap)[.,!?]?\s*", re.I),
    re.compile(r"^\s*good\s+(?:morning|afternoon|evening)(?:\s+everyone|,?\s+folks)?[.,!?]?\s*", re.I),
    re.compile(r"^\s*hey\s+everyone[.,!?]?\s*", re.I),
    re.compile(r"^\s*hello\s+(?:everyone|folks|listeners)[.,!?]?\s*", re.I),
    re.compile(r"^\s*well,?\s+folks[.,!?]?\s*", re.I),
    re.compile(r"^\s*folks,?\s+", re.I),
    re.compile(r"^\s*as\s+always[.,!?]?\s*", re.I),
    re.compile(r"^\s*it'?s\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:morning|afternoon|evening)[.,!?]?\s*", re.I),
    re.compile(r"^\s*ready\s+for\s+(?:some\s+)?(?:laughs|insights|action|news)\??\s*", re.I),
    re.compile(r"^\s*let'?s\s+dive\s+in[.,!?]?\s*", re.I),
]

# Map of all host first-names (lowercase) → canonical NAME.
_NAME_TO_CANONICAL = {n.lower(): n for n in CHARACTERS.keys()}

# Patterns that indicate a self-introduction phrase containing a name.
# Captures the name in group 1.
_SELF_INTRO_RE = re.compile(
    r"\b(" + "|".join(re.escape(n.title()) for n in CHARACTERS.keys()) + r")\b"
    r"(?=\s+(?:here|again|checking\s+in|back|on\s+the|with|from\s+the|at\s+the|, |—))",
    re.I,
)

# Parenthesized ticker e.g. (AAPL), (MSFT), (V).
_PAREN_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")

# Standalone ticker — 2-5 caps as a whole word, not preceded by `[` (audio tag)
# or another cap (avoids splitting acronyms inside larger words). Common
# false-positives are filtered against an explicit allowlist below.
_STANDALONE_TICKER_RE = re.compile(r"(?<![A-Z])(?<![\[\(])\b([A-Z]{2,5})\b(?![A-Z])")

# Acronyms / words that look like tickers but aren't — never spell these out.
_TICKER_FALSE_POSITIVES = {
    "CEO", "CFO", "COO", "CTO", "CIO", "IPO", "ETF", "API", "AI", "GDP",
    "PR", "OK", "USA", "US", "UK", "EU", "UN", "FDA", "FBI", "CIA", "NSA",
    "SEC", "FTC", "DOJ", "EPA", "IRS", "FAA", "CDC", "NIH",
    "CES", "CEO", "ESG", "VPN", "SaaS", "B2B", "B2C", "FYI", "TBD", "TLDR",
    "USD", "EUR", "GBP", "JPY", "CNY", "GMT", "UTC", "EST", "EDT", "PST",
    "GPU", "CPU", "RAM", "SSD", "HDD", "USB", "URL", "HTTP", "HTTPS",
    "JSON", "XML", "HTML", "CSS", "SQL", "iOS", "macOS", "BBQ", "DIY",
    "NEW", "OLD", "BIG", "TOP", "PER", "PRO", "AND", "FOR", "BUT", "NOT",
    "ALL", "ANY", "ONE", "TWO", "OUR", "WAS", "ARE", "WHO", "HOW", "WHY",
}

# Dollar amount patterns: $5B, $5 billion, $5.2M, $100, etc.
_DOLLAR_AMOUNT_RE = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T|K)?\b",
    re.I,
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# Catches "1.11 percent", "5.2 percent" etc. — digit + word "percent" without
# the % symbol. Critique sometimes leaves these because it focuses on % and
# misses the spelled-out variant.
_PERCENT_WORD_RE = re.compile(r"(\d+(?:\.\d+)?)\s+percent\b", re.I)
_DOLLAR_WORD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+(billion|million|trillion|thousand)\s+dollars?\b",
    re.I,
)


def _spell_number(n: str) -> str:
    """Spell '5.2' as 'five point two'. Keeps it simple — defers to TTS for
    big numbers like '4200', which ElevenLabs handles fine."""
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    if "." not in n:
        return n  # let TTS handle whole numbers
    whole, _, frac = n.partition(".")
    return f"{whole} point {' '.join(digits.get(d, d) for d in frac)}"


def _normalize_dollars(text: str) -> tuple[str, int]:
    """$5.2B → 'five point two billion dollars', $100 → '100 dollars'."""
    fixes = 0
    SUFFIX = {
        "b": "billion", "billion": "billion",
        "m": "million", "million": "million",
        "t": "trillion", "trillion": "trillion",
        "k": "thousand",
    }
    def repl(m: re.Match) -> str:
        nonlocal fixes
        fixes += 1
        num, suffix = m.group(1), (m.group(2) or "").lower()
        spelled = _spell_number(num)
        if suffix:
            return f"{spelled} {SUFFIX.get(suffix, suffix)} dollars"
        return f"{spelled} dollars"
    return _DOLLAR_AMOUNT_RE.sub(repl, text), fixes


def _normalize_percents(text: str) -> tuple[str, int]:
    """5.2% → 'five point two percent'. Also catches '1.11 percent' (digit
    followed by spelled 'percent') so we get both variants the LLM can emit."""
    fixes = 0
    def repl_pct(m: re.Match) -> str:
        nonlocal fixes
        fixes += 1
        return f"{_spell_number(m.group(1))} percent"
    text, _ = _PERCENT_RE.subn(repl_pct, text)
    # Now handle "digit + space + 'percent'" — e.g. "1.11 percent"
    def repl_word(m: re.Match) -> str:
        nonlocal fixes
        n = m.group(1)
        # only normalize fractional numbers; whole numbers like "5 percent"
        # are fine for TTS to read.
        if "." in n:
            fixes += 1
            return f"{_spell_number(n)} percent"
        return m.group(0)
    text = _PERCENT_WORD_RE.sub(repl_word, text)
    return text, fixes


def _normalize_dollars_word(text: str) -> tuple[str, int]:
    """'5.2 billion dollars' → 'five point two billion dollars'. Like the
    percent variant — catches the digit+spelled-suffix form when LLM
    didn't fully spell out a fractional amount."""
    fixes = 0
    def repl(m: re.Match) -> str:
        nonlocal fixes
        n, suffix = m.group(1), m.group(2).lower()
        if "." in n:
            fixes += 1
            return f"{_spell_number(n)} {suffix} dollars"
        return m.group(0)
    return _DOLLAR_WORD_RE.sub(repl, text), fixes


def _space_standalone_tickers(text: str) -> tuple[str, int]:
    """Standalone NVDA / AAPL / MSFT etc → 'N V D A'. Skips known acronyms
    AND host names (JAMIE / ALEX / MAYA / etc.) so we don't mangle dialogue."""
    fixes = 0
    char_names = {n.upper() for n in CHARACTERS.keys()}
    def repl(m: re.Match) -> str:
        nonlocal fixes
        tk = m.group(1)
        if tk in _TICKER_FALSE_POSITIVES or tk in char_names:
            return tk
        # also skip if it's all the same letter (e.g. "II", "III")
        if len(set(tk)) == 1:
            return tk
        fixes += 1
        return " ".join(tk)
    return _STANDALONE_TICKER_RE.sub(repl, text), fixes


def _strip_banned_openers(text: str) -> tuple[str, list[str]]:
    """Iteratively strip banned cold-open phrases from the start of text."""
    removed: list[str] = []
    while True:
        before = text
        for rx in _BANNED_OPENERS:
            m = rx.match(text)
            if m:
                removed.append(m.group(0).strip())
                text = text[m.end():]
                break
        if text == before:
            break
    return text.lstrip(), removed


def _fix_wrong_name_intros(speaker: str, text: str) -> tuple[str, int]:
    """If the speaker's line contains a self-intro using ANOTHER host's name,
    rewrite that name to the actual speaker's title-case form."""
    fixes = 0
    speaker_pretty = speaker.title()

    def repl(m: re.Match) -> str:
        nonlocal fixes
        found = m.group(1).lower()
        canonical = _NAME_TO_CANONICAL.get(found)
        if canonical and canonical != speaker:
            fixes += 1
            return speaker_pretty
        return m.group(0)

    return _SELF_INTRO_RE.sub(repl, text), fixes


def _space_tickers(text: str) -> tuple[str, int]:
    """Convert (AAPL) → A A P L for cleaner TTS pronunciation."""
    fixes = 0

    def repl(m: re.Match) -> str:
        nonlocal fixes
        ticker = m.group(1)
        # skip very common all-caps non-tickers
        if ticker in {"CEO", "CFO", "COO", "CTO", "IPO", "ETF", "API", "AI", "GDP", "PR", "OK"}:
            return m.group(0)
        fixes += 1
        return " ".join(ticker)

    return _PAREN_TICKER_RE.sub(repl, text), fixes


# --- post-process passes for newer rules ---


_BANNED_RE_LIST = [re.compile(rf"[^.!?]*\b{re.escape(p)}\b[^.!?]*[.!?]?", re.I) for p in BANNED_PHRASES]


def _scrub_banned_sentences(text: str) -> tuple[str, int]:
    """Strip any sentence containing a banned phrase. If the turn is empty
    after scrubbing, callers drop the turn."""
    fixes = 0
    for rx in _BANNED_RE_LIST:
        new = rx.sub("", text)
        if new != text:
            fixes += text.lower().count("") - new.lower().count("")  # crude
            text = new
    return re.sub(r"\s+", " ", text).strip(), fixes


def _drop_repeated_openers(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """If a 1-3 word turn-opener appears in 3+ turns, rewrite the 3rd+
    occurrence to drop the opener (keep the substance). Detection on
    the first 5 word-tokens, normalized."""
    counts: dict[str, int] = {}
    drops = 0
    out: list[tuple[str, str]] = []
    for name, text in turns:
        # strip leading audio tag for detection
        body = re.sub(r"^\[[^\]]+\]\s*", "", text)
        first_words = re.split(r"\s+", body, maxsplit=4)
        if len(first_words) < 2:
            out.append((name, text))
            continue
        key = " ".join(w.lower().strip(",.;:!?-—") for w in first_words[:3] if w)
        # only consider meaningful openers (not just one common word)
        if len(key) < 8:
            out.append((name, text))
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 2:
            # rewrite turn: drop the opening clause up to first comma or em-dash
            stripped = re.sub(r"^\[[^\]]+\]\s*", "", text)
            # drop everything up to and including first comma/em-dash + space
            new_body = re.sub(r"^[^,.—\-]{1,40}[,—\-]\s*", "", stripped, count=1)
            if not new_body or new_body == stripped:
                # nothing reasonable to strip — just drop the turn entirely
                drops += 1
                continue
            # preserve audio tag prefix if it existed
            tag_match = re.match(r"^(\[[^\]]+\]\s*)", text)
            new_text = (tag_match.group(1) if tag_match else "") + new_body
            new_text = new_text[0].upper() + new_text[1:] if new_text else new_text
            out.append((name, new_text))
            drops += 1
        else:
            out.append((name, text))
    return out, drops


def _scrub_banned_in_turns(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """Per-turn: scrub sentences containing banned phrases. Only drop a turn
    if the scrub MADE it too short (i.e. the original had a banned phrase
    AND removing that phrase left <3 words). Naturally-short clean turns
    like 'Wow.' or 'Apple beat earnings.' are preserved unchanged."""
    fixes = 0
    out: list[tuple[str, str]] = []
    for name, text in turns:
        new, _ = _scrub_banned_sentences(text)
        if new == text:
            # No banned phrase found — keep the turn untouched, even if short
            out.append((name, text))
            continue
        # Scrub did fire. Drop the turn only if scrubbing made it useless.
        fixes += 1
        if len(new.split()) >= 3:
            out.append((name, new))
        # else: drop entirely
    return out, fixes


def _disclaimer_signature(text: str) -> bool:
    """Detect any turn that's a disclaimer (verbatim or paraphrase). Looks for
    the canonical short fragment 'entertainment and education only'."""
    return "entertainment and education only" in text.lower() or "investment advice" in text.lower()


def _dedup_disclaimer(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """Keep AT MOST ONE disclaimer turn, and only at the very end. Drop any
    earlier disclaimer turns and any non-disclaimer turns that follow the
    last disclaimer (so the disclaimer is genuinely the closing line)."""
    if not turns:
        return turns, 0
    # Find indices of all disclaimer-flavored turns
    discl_idxs = [i for i, (_, t) in enumerate(turns) if _disclaimer_signature(t)]
    if not discl_idxs:
        return turns, 0
    keep_idx = discl_idxs[-1]
    # Replace the kept turn with the canonical short disclaimer line under JAMIE
    speaker = turns[keep_idx][0] if turns[keep_idx][0] in CHARACTERS else DEFAULT_CHARACTER
    canonical_turn = (speaker, DISCLAIMER_SHORT)
    # Strip any disclaimer-flavored turns that came BEFORE keep_idx,
    # and drop everything AFTER keep_idx so the disclaimer is the last line.
    out: list[tuple[str, str]] = []
    drops = 0
    for i, t in enumerate(turns):
        if i in discl_idxs and i != keep_idx:
            drops += 1
            continue
        if i > keep_idx:
            drops += 1
            continue
        if i == keep_idx:
            out.append(canonical_turn)
            continue
        out.append(t)
    return out, drops


def _drop_self_reference(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """If a host's turn includes their own name in a sign-off pattern OR a
    mid-turn vocative (e.g. JAMIE saying 'What's the story, Jamie?'), strip
    that fragment. Conservative — only matches the patterns where the
    self-name is being USED as direct address, never narrative ('I told
    Jamie earlier' is preserved)."""
    fixes = 0
    out: list[tuple[str, str]] = []
    for name, text in turns:
        title = name.title()
        # 1) Sign-off forms: "Later, Jamie —", "See ya, Jamie", "Thanks, Jamie"
        new = re.sub(
            rf"\b(later|out|see ya|catch you|thanks),?\s+{re.escape(title)}\b[,.\s—-]*",
            "",
            text,
            flags=re.I,
        )
        new = re.sub(rf"\b{re.escape(title)},\s+(out|signing off)\b[.,!]*", "", new, flags=re.I)
        # 2) Mid-turn vocative — ", Jamie?" / ", Jamie." / ", Jamie!" / ", Jamie —"
        # at the end of a clause/sentence where speaker is JAMIE.
        new = re.sub(
            rf",\s+{re.escape(title)}(?=[?!.\s—\-]|$)",
            "",
            new,
            flags=re.I,
        )
        # 3) Trailing exclamation like "Sure, Jamie!" — handled by #2 because
        # the comma+name pattern matches before the punctuation.
        if new != text:
            fixes += 1
        out.append((name, new.strip() or text))  # never produce empty turn
    return out, fixes


def _collapse_same_speaker_streaks(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """Two consecutive turns by the same speaker → merge into one."""
    if not turns:
        return turns, 0
    out: list[tuple[str, str]] = [turns[0]]
    merges = 0
    for name, text in turns[1:]:
        if out and out[-1][0] == name:
            out[-1] = (name, (out[-1][1] + " " + text).strip())
            merges += 1
        else:
            out.append((name, text))
    return out, merges


def _enforce_jamie_cap(turns: list[tuple[str, str]], cap_ratio: float = 1 / 3) -> tuple[list[tuple[str, str]], int]:
    """Drop JAMIE turns shorter than 8 words until JAMIE airtime ≤ cap_ratio.
    Preserves JAMIE's substantive turns (cold open, big reactions)."""
    if not turns:
        return turns, 0
    counts = Counter(name for name, _ in turns)
    total = sum(counts.values())
    target = max(1, int(total * cap_ratio))
    if counts.get(DEFAULT_CHARACTER, 0) <= target:
        return turns, 0

    jamie_indices_with_len = [
        (i, len(text.split())) for i, (name, text) in enumerate(turns)
        if name == DEFAULT_CHARACTER
    ]
    # Drop shortest JAMIE turns first, but never the very first turn (cold open).
    droppable = sorted(
        [(i, wc) for i, wc in jamie_indices_with_len if i != 0 and wc < 8],
        key=lambda p: p[1],
    )
    drop_set: set[int] = set()
    excess = counts[DEFAULT_CHARACTER] - target
    for i, _ in droppable:
        if excess <= 0:
            break
        drop_set.add(i)
        excess -= 1

    return [t for i, t in enumerate(turns) if i not in drop_set], len(drop_set)


def _parse(text: str) -> list[tuple[str, str]]:
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


def _format(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{n}: {t}" for n, t in turns)


def sanitize_script(text: str, verbose: bool = True) -> str:
    """Apply all post-process guardrails to a generated dialogue script."""
    turns = _parse(text)
    if not turns:
        return text

    stats = {
        "opener_strips": 0, "name_fixes": 0, "ticker_fixes": 0,
        "standalone_ticker_fixes": 0, "dollar_fixes": 0, "percent_fixes": 0,
        "jamie_drops": 0,
    }

    # 1) strip banned openers from first turn only
    first_name, first_text = turns[0]
    cleaned, removed = _strip_banned_openers(first_text)
    if removed:
        stats["opener_strips"] = len(removed)
        if len(cleaned.split()) < 3 and len(turns) > 1:
            turns = turns[1:]
        else:
            turns[0] = (first_name, cleaned)

    # 2) per-turn text fixes
    for i, (name, text_) in enumerate(turns):
        text_, n_fixes = _fix_wrong_name_intros(name, text_)
        stats["name_fixes"] += n_fixes
        text_, t_fixes = _space_tickers(text_)
        stats["ticker_fixes"] += t_fixes
        text_, st_fixes = _space_standalone_tickers(text_)
        stats["standalone_ticker_fixes"] += st_fixes
        text_, d_fixes = _normalize_dollars(text_)
        stats["dollar_fixes"] += d_fixes
        text_, dw_fixes = _normalize_dollars_word(text_)
        stats["dollar_fixes"] += dw_fixes
        text_, p_fixes = _normalize_percents(text_)
        stats["percent_fixes"] += p_fixes
        turns[i] = (name, text_)

    # 2.5) scrub mid-script banned phrases (drops sentences containing them;
    # drops the whole turn if it becomes too short to be useful)
    turns, banned_drops = _scrub_banned_in_turns(turns)
    stats["banned_scrubs"] = banned_drops

    # 2.6) drop / rewrite repeated turn-openers (3rd+ "And in other news…")
    turns, opener_repeats = _drop_repeated_openers(turns)
    stats["opener_repeats"] = opener_repeats

    # 3) cap JAMIE airtime
    turns, dropped = _enforce_jamie_cap(turns)
    stats["jamie_drops"] = dropped

    # 4) collapse same-speaker streaks (two consecutive NAME: lines → merge)
    turns, merges = _collapse_same_speaker_streaks(turns)
    stats["streak_merges"] = merges

    # 5) drop self-reference patterns ("Later, Jamie" from JAMIE)
    turns, self_ref_fixes = _drop_self_reference(turns)
    stats["self_ref_fixes"] = self_ref_fixes

    # 6) dedup disclaimer (keep only the final one; drop anything after)
    turns, discl_drops = _dedup_disclaimer(turns)
    stats["disclaimer_drops"] = discl_drops

    # ensure the script ENDS with the disclaimer; if model never produced
    # one, append a JAMIE turn with the canonical short version.
    if turns and not _disclaimer_signature(turns[-1][1]):
        turns.append((DEFAULT_CHARACTER, DISCLAIMER_SHORT))
        stats["disclaimer_appended"] = 1

    if verbose:
        print(
            f"[sanitize] openers={stats['opener_strips']} "
            f"name_fixes={stats['name_fixes']} "
            f"tickers_paren={stats['ticker_fixes']} "
            f"tickers_standalone={stats['standalone_ticker_fixes']} "
            f"dollars={stats['dollar_fixes']} "
            f"percents={stats['percent_fixes']} "
            f"banned_scrubs={stats.get('banned_scrubs',0)} "
            f"opener_repeats={stats.get('opener_repeats',0)} "
            f"jamie_drops={stats['jamie_drops']} "
            f"streak_merges={stats.get('streak_merges',0)} "
            f"self_ref={stats.get('self_ref_fixes',0)} "
            f"disclaimer_drops={stats.get('disclaimer_drops',0)}"
            + (f" disclaimer_appended=1" if stats.get('disclaimer_appended') else "")
        )
    return _format(turns)


if __name__ == "__main__":
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else open(sys.argv[1]).read()
    print(sanitize_script(src))
