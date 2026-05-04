"""Cluster near-identical headlines from different RSS sources into single
story records. Pure-Python, no LLM, no network. Fast.

Algorithm: Jaccard similarity on normalized title token sets. Greedy single-link
clustering: each new headline joins the existing cluster with the highest
overlap if that overlap exceeds the threshold; otherwise it seeds a new cluster.
The threshold (default 0.5) is tuned to merge headlines like
'Spirit Airlines shutting down' / 'Spirit ceases operations' / 'Spirit Airlines
collapse leaves 4,200 jobless' into one story while keeping 'Spirit Halloween
files for IPO' separate.
"""
from __future__ import annotations
import hashlib
import re

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or", "by",
    "with", "after", "as", "is", "are", "was", "were", "be", "from", "that",
    "this", "it", "its", "but", "not", "have", "has", "had", "will", "would",
    "amid", "over", "under", "into", "out", "up", "down", "about",
}


def _normalize(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return " ".join(w for w in title.split() if w not in _STOPWORDS and len(w) > 1)


def _tokens(title: str) -> set[str]:
    return set(_normalize(title).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_headlines(headlines: list[dict], threshold: float = 0.2, min_shared: int = 2) -> list[dict]:
    """Group near-duplicate headlines into clusters.

    Returns: list of dicts with keys
        id              — sha1[:12] of canonical normalized title
        title           — longest title in the cluster
        summary         — first non-empty summary
        sources         — sorted list of source names
        categories      — sorted list of category tags (markets/business/...)
        published       — most recent ISO datetime among cluster members
        cluster_size    — number of headlines that landed in the cluster
        link            — first non-empty link
        headlines       — original headline dicts (for downstream needs)
    """
    buckets: list[dict] = []
    for h in headlines:
        title = h.get("title") or ""
        toks = _tokens(title)
        if not toks:
            continue
        best_idx, best_sim, best_shared = -1, 0.0, 0
        for i, b in enumerate(buckets):
            shared = len(toks & b["_tokens"])
            sim = _jaccard(toks, b["_tokens"])
            if sim > best_sim:
                best_idx, best_sim, best_shared = i, sim, shared
        if best_idx >= 0 and best_sim >= threshold and best_shared >= min_shared:
            b = buckets[best_idx]
            b["headlines"].append(h)
            b["sources"].add(h.get("source") or "unknown")
            b["categories"].add(h.get("category") or "unknown")
            b["_tokens"] |= toks
        else:
            buckets.append({
                "headlines": [h],
                "sources": {h.get("source") or "unknown"},
                "categories": {h.get("category") or "unknown"},
                "_tokens": set(toks),
            })

    clusters: list[dict] = []
    for b in buckets:
        titles = [h.get("title") or "" for h in b["headlines"]]
        canonical = max(titles, key=len) if titles else ""
        cid = hashlib.sha1(_normalize(canonical).encode()).hexdigest()[:12]
        summary = next((h.get("summary", "") for h in b["headlines"] if h.get("summary")), "")
        link = next((h.get("link", "") for h in b["headlines"] if h.get("link")), "")
        published = max((h.get("published") or "" for h in b["headlines"]), default="")
        clusters.append({
            "id": cid,
            "title": canonical,
            "summary": summary,
            "sources": sorted(b["sources"]),
            "categories": sorted(b["categories"]),
            "published": published,
            "cluster_size": len(b["headlines"]),
            "link": link,
            "headlines": b["headlines"],
        })
    return clusters


if __name__ == "__main__":
    import json
    import sys
    from fetch_news import fetch_headlines, flatten
    h = fetch_headlines()
    clusters = cluster_headlines(flatten(h))
    print(f"{sum(len(v) for v in h.values())} headlines → {len(clusters)} clusters")
    # show top by cluster size
    clusters.sort(key=lambda c: c["cluster_size"], reverse=True)
    for c in clusters[:10]:
        print(f"  [{c['cluster_size']}] {c['title'][:80]}  ({', '.join(c['sources'])})")
