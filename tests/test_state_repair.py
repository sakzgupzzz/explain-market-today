"""State persistence + corruption recovery."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import state


def _with_temp_state(callback):
    tmp = Path(tempfile.mkdtemp())
    fake_path = tmp / ".state.json"
    with patch.object(state, "STATE_PATH", fake_path):
        callback(fake_path)


def test_load_state_when_missing_returns_empty():
    def go(p):
        s = state.load_state()
        assert s == {"covered": []}
    _with_temp_state(go)


def test_load_state_valid_json_round_trips():
    def go(p):
        p.write_text(json.dumps({"covered": [{"cluster_id": "abc", "first_covered": "2026-05-01T00:00:00Z"}]}))
        s = state.load_state()
        assert s["covered"][0]["cluster_id"] == "abc"
    _with_temp_state(go)


def test_load_state_corrupt_json_rotates_and_returns_fresh():
    def go(p):
        p.write_text("{this is not json")
        s = state.load_state()
        assert s.get("covered") == []
        assert s.get("recovered_from_corruption") is True
        assert (p.with_suffix(".json.broken")).exists()
    _with_temp_state(go)


def test_load_state_invalid_schema_rotates():
    def go(p):
        p.write_text(json.dumps(["not", "a", "dict"]))
        s = state.load_state()
        assert s.get("recovered_from_corruption") is True
    _with_temp_state(go)


def test_load_state_filters_malformed_entries():
    def go(p):
        p.write_text(json.dumps({
            "covered": [
                {"cluster_id": "good", "first_covered": "2026-05-01T00:00:00Z"},
                {"cluster_id": 123},  # malformed: int id
                {"first_covered": "2026-05-01T00:00:00Z"},  # malformed: no cluster_id
                "not a dict at all",
            ]
        }))
        s = state.load_state()
        assert len(s["covered"]) == 1
        assert s["covered"][0]["cluster_id"] == "good"
    _with_temp_state(go)
