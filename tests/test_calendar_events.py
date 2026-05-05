"""Calendar events module."""
from __future__ import annotations
from datetime import date, timedelta
from unittest.mock import patch

import calendar_events


def test_macro_calendar_contains_known_2026_events():
    """Smoke check that the hardcoded list isn't empty."""
    assert len(calendar_events.MACRO_CALENDAR_2026) > 10
    # Each entry is (date_str, name)
    for d, n in calendar_events.MACRO_CALENDAR_2026:
        assert d.startswith("2026-")
        assert n


def test_upcoming_macro_filters_by_window():
    today = date.today()
    fixture = [
        ((today - timedelta(days=1)).isoformat(), "Past event"),
        ((today + timedelta(days=2)).isoformat(), "Soon"),
        ((today + timedelta(days=10)).isoformat(), "Later"),
    ]
    with patch.object(calendar_events, "MACRO_CALENDAR_2026", fixture):
        out = calendar_events.upcoming_macro(days_ahead=5)
    names = [n for _, n in out]
    assert "Soon" in names
    assert "Past event" not in names
    assert "Later" not in names


def test_fmt_events_block_empty_returns_empty_string():
    assert calendar_events.fmt_events_block([], []) == ""


def test_fmt_events_block_includes_macro_and_earnings():
    block = calendar_events.fmt_events_block(
        [("2026-05-13", "CPI release")],
        [("2026-05-07", "AAPL"), ("2026-05-07", "MSFT")],
    )
    assert "CPI release" in block
    assert "AAPL" in block
    assert "MSFT" in block
