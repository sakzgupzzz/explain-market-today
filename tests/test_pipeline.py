"""Tests for the new ingest → rank → render layers."""
from __future__ import annotations
import pytest
from cluster import cluster_headlines, _normalize, _jaccard, _tokens
from score import score_clusters, _keyword_score, _source_score, _recency_decay
from state import covered_within, mark_covered, annotate_clusters


# ─── cluster ────────────────────────────────────────────────────────────────

def test_normalize_strips_stopwords_and_punctuation():
    assert "the" not in _normalize("The Apple Stock Drops!")
    assert "apple" in _normalize("The Apple Stock Drops!")


def test_jaccard_identical_sets():
    assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_cluster_merges_near_duplicate_titles():
    headlines = [
        {"title": "Spirit Airlines shutting down operations", "source": "Reuters", "category": "business", "summary": "", "published": "2026-05-04T12:00:00Z"},
        {"title": "Spirit Airlines collapse leaves 4200 workers jobless", "source": "AP", "category": "business", "summary": "", "published": "2026-05-04T13:00:00Z"},
    ]
    clusters = cluster_headlines(headlines)
    assert len(clusters) == 1
    assert clusters[0]["cluster_size"] == 2
    assert "AP" in clusters[0]["sources"]
    assert "Reuters" in clusters[0]["sources"]


def test_cluster_keeps_unrelated_separate():
    headlines = [
        {"title": "Apple ships iOS 21", "source": "TechCrunch", "category": "tech", "summary": "", "published": ""},
        {"title": "Spirit Airlines bankruptcy", "source": "Reuters", "category": "business", "summary": "", "published": ""},
    ]
    clusters = cluster_headlines(headlines)
    assert len(clusters) == 2


def test_cluster_canonical_title_is_longest():
    headlines = [
        {"title": "FBI raids Trump tower", "source": "AP", "category": "world", "summary": "", "published": ""},
        {"title": "FBI executes search warrant at Donald Trump's tower in New York", "source": "Reuters", "category": "world", "summary": "", "published": ""},
    ]
    clusters = cluster_headlines(headlines)
    assert "Donald Trump" in clusters[0]["title"]


# ─── score ──────────────────────────────────────────────────────────────────

def test_keyword_score_picks_up_earnings():
    s = _keyword_score("Apple beat earnings expectations")
    assert s >= 3.0


def test_source_score_reuters_high():
    assert _source_score(["Reuters"]) == 1.0


def test_source_score_unknown_default():
    assert _source_score(["RandomBlog"]) == 0.6


def test_recency_decay_recent():
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    assert _recency_decay(now_iso) == 1.0


def test_score_clusters_sorts_by_score():
    clusters = [
        {"id": "a", "title": "Random local news", "summary": "", "sources": ["Blog"], "categories": ["culture"], "published": "", "cluster_size": 1, "headlines": []},
        {"id": "b", "title": "Apple earnings beat by 20% revenue surge", "summary": "", "sources": ["Reuters", "AP", "Bloomberg"], "categories": ["business"], "published": "", "cluster_size": 3, "headlines": []},
    ]
    market = {"gainers": [{"symbol": "AAPL", "pct": 4.5, "name": "Apple", "close": 200.0, "prev_close": 191.0}], "losers": []}
    ranked = score_clusters(clusters, market)
    assert ranked[0]["id"] == "b"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_score_clusters_drops_blocked_topics():
    clusters = [
        {"id": "a", "title": "Sports betting expansion in Nevada", "summary": "", "sources": ["AP"], "categories": ["culture"], "published": "", "cluster_size": 1, "headlines": []},
        {"id": "b", "title": "Apple ships iPad", "summary": "", "sources": ["TechCrunch"], "categories": ["tech"], "published": "", "cluster_size": 1, "headlines": []},
    ]
    interests = {"blocked": {"topics": ["sports betting"]}}
    ranked = score_clusters(clusters, market={}, interests=interests)
    assert all(c["id"] != "a" for c in ranked)


def test_score_clusters_watchlist_ticker_boost():
    clusters = [
        {"id": "a", "title": "Anthropic raises 1B in Series E", "summary": "", "sources": ["Bloomberg"], "categories": ["business"], "published": "", "cluster_size": 1, "headlines": []},
    ]
    no_interest = score_clusters(clusters, market={}, interests={})
    base = no_interest[0]["score"]
    interest = score_clusters(clusters, market={}, interests={"watchlist": {"keywords": ["series e"]}})
    boosted = interest[0]["score"]
    assert boosted > base


# ─── state ──────────────────────────────────────────────────────────────────

def test_state_mark_and_query():
    state = {"covered": []}
    mark_covered(state, ["abc", "def"])
    assert "abc" in covered_within(state, days=1)
    assert "def" in covered_within(state, days=1)


def test_annotate_clusters_flags_recently_seen():
    state = {"covered": []}
    mark_covered(state, ["xyz"])
    clusters = [{"id": "xyz"}, {"id": "new"}]
    annotated = annotate_clusters(clusters, state, suppress_days=2)
    by_id = {c["id"]: c["seen_recently"] for c in annotated}
    assert by_id["xyz"] is True
    assert by_id["new"] is False
