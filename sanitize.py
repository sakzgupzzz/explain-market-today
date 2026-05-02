"""Post-process LLM-generated dialogue scripts before TTS.

Deterministic guardrails for things the prompt can't reliably enforce on a 14B
local model: banned openers, wrong-name intros, parenthesized tickers,
JAMIE airtime cap. Idempotent — running twice yields the same output.
"""
from __future__ import annotations
import re
from collections import Counter
from config import CHARACTERS, DEFAULT_CHARACTER

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

    stats = {"opener_strips": 0, "name_fixes": 0, "ticker_fixes": 0, "jamie_drops": 0}

    # 1) strip banned openers from first turn only
    first_name, first_text = turns[0]
    cleaned, removed = _strip_banned_openers(first_text)
    if removed:
        stats["opener_strips"] = len(removed)
        # If stripping left the line empty or trivially short, drop the turn.
        if len(cleaned.split()) < 3 and len(turns) > 1:
            turns = turns[1:]
        else:
            turns[0] = (first_name, cleaned)

    # 2) fix wrong-name intros + 3) space tickers, per turn
    for i, (name, text_) in enumerate(turns):
        text_, n_fixes = _fix_wrong_name_intros(name, text_)
        stats["name_fixes"] += n_fixes
        text_, t_fixes = _space_tickers(text_)
        stats["ticker_fixes"] += t_fixes
        turns[i] = (name, text_)

    # 4) cap JAMIE airtime
    turns, dropped = _enforce_jamie_cap(turns)
    stats["jamie_drops"] = dropped

    if verbose:
        print(
            f"[sanitize] openers={stats['opener_strips']} "
            f"name_fixes={stats['name_fixes']} "
            f"ticker_fixes={stats['ticker_fixes']} "
            f"jamie_drops={stats['jamie_drops']}"
        )
    return _format(turns)


if __name__ == "__main__":
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else open(sys.argv[1]).read()
    print(sanitize_script(src))
