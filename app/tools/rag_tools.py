"""
RAG tools: semantic search over `cloud_documents` with cross-encoder reranking.

Pipeline (both tools):
  question --embed_query--> vector search (RPC match_cloud_documents, top CANDIDATES)
           --rerank--> keep chunks scoring ≥ RERANK_MIN_SCORE --> top TOP_K, with sources.

The LLM chooses only the query (and a category / provider filter); thresholds and counts
are fixed here — the model used to pick values like match_count=10, threshold=0.5.

* search_cloud_documents — OUR documentation (SLA, security, features…); crawled
  competitor pages are excluded so a competitor's figures are never presented as ours.
* search_competitor_docs — crawled public pages of Vietnamese competitors (category
  "Competitor"), optionally for one provider.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_query
from app.tools.reranker import rerank

logger = logging.getLogger(__name__)

CANDIDATES = 15            # vector-search candidates handed to the reranker
TOP_K = 4                  # chunks returned to the agent
VECTOR_THRESHOLD = 0.3     # loose: the reranker does the real filtering
COMPETITOR_CATEGORY = "Competitor"
COMPETITOR_PROVIDERS = ("vng", "fpt", "bizfly", "vnpt", "viettelidc", "viettel")


def _vector_search(query: str, *, category: str | None = None, exclude: list[str] | None = None,
                   provider: str | None = None) -> list[dict]:
    """
    Candidates from the RPC. Uses the migration-003 signature (exclude_categories,
    filter_provider); if the database hasn't been migrated yet, falls back to the original
    signature and applies the same filters in Python.
    """
    params = {"query_embedding": embed_query(query), "match_threshold": VECTOR_THRESHOLD,
              "match_count": CANDIDATES, "filter_category": category}
    try:
        return supabase.rpc("match_cloud_documents", {
            **params, "exclude_categories": exclude, "filter_provider": provider}).execute().data or []
    except Exception as exc:  # noqa: BLE001 - old function signature (migration 003 not run)
        logger.warning("match_cloud_documents (v3) failed, using the old signature: %s", exc)
    params["match_count"] = CANDIDATES * 4
    rows = supabase.rpc("match_cloud_documents", params).execute().data or []
    rows = [r for r in rows if not exclude or r.get("category") not in exclude]
    if provider:
        rows = [r for r in rows if (r.get("metadata") or {}).get("provider") == provider]
    return rows[:CANDIDATES]


def _format(results: list[dict], title: str) -> str:
    lines = [f"### {title}:"]
    for idx, doc in enumerate(results, 1):
        meta = doc.get("metadata") or {}
        source = meta.get("source") or "?"
        fetched = f" | fetched {meta['fetched_at'][:10]}" if meta.get("fetched_at") else ""
        score = doc.get("rerank_score", doc.get("similarity", 0.0))
        lines.append(
            f"\n[{idx}] Service: {doc.get('service_name')} | Category: {doc.get('category')} | "
            f"Source: {source}{fetched} | Relevance: {score:.2f}\n{doc.get('content')}"
        )
    return "\n".join(lines)


@tool
def search_cloud_documents(query: str, category: str | None = None) -> str:
    """
    Searches OUR OWN technical, SLA, security, compliance and feature documentation.

    Args:
        query: The question or topic to search for (rephrase it as a full question).
        category: Optional filter: 'SLA', 'Security', 'Compliance', 'Feature', 'CaseStudy'.
            Leave empty to search everything (recommended unless the user names a category).

    Returns:
        The most relevant passages with service name, category and source, or a message
        that no relevant document exists.
    """
    candidates = _vector_search(query, category=category, exclude=[COMPETITOR_CATEGORY])
    results = rerank(query, candidates, TOP_K)
    if not results and category:
        # The LLM's category guess may not exist in the corpus (e.g. no "SLA" documents yet):
        # search everything rather than wrongly concluding there is no information.
        candidates = _vector_search(query, exclude=[COMPETITOR_CATEGORY])
        results = rerank(query, candidates, TOP_K)
    if not results:
        return "No relevant documents found in the database for this question."
    return _format(results, "DOCUMENT SEARCH RESULTS")


@tool
def search_competitor_docs(query: str, provider: str | None = None) -> str:
    """
    Searches crawled PUBLIC web pages of Vietnamese cloud competitors (product pages and
    price lists of VNG, FPT, Bizfly, VNPT, Viettel IDC) — for questions about a competitor's
    services, features, SLAs or list prices.

    Args:
        query: The question or topic to search for.
        provider: Optional: one of 'vng', 'fpt', 'bizfly', 'vnpt', 'viettelidc', 'viettel'.
            Leave empty to search all competitors.

    Returns:
        Relevant passages with provider, page URL and crawl date, or a message that nothing
        relevant was found.
    """
    provider = (provider or "").strip().lower() or None
    if provider and provider not in COMPETITOR_PROVIDERS:
        return f"Unknown provider {provider!r}. Valid options: {', '.join(COMPETITOR_PROVIDERS)}."
    candidates = _vector_search(query, category=COMPETITOR_CATEGORY, provider=provider)
    results = rerank(query, candidates, TOP_K)
    if not results:
        return "No relevant competitor pages found for this question."
    return _format(results, "COMPETITOR PAGE RESULTS (public website data — cite the source and date)")
