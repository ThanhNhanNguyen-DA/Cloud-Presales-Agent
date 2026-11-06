"""
RAG tools: semantic search over `cloud_documents` with cross-encoder reranking.

Pipeline (both tools):
  vector candidates + optional keyword candidates -> reciprocal rank fusion
  -> provenance/scope checks -> rerank -> relevance cutoff -> top TOP_K with sources.

The LLM chooses only the query (and a category / provider filter); thresholds and counts
are fixed here — the model used to pick values like match_count=10, threshold=0.5.

* search_cloud_documents — OUR documentation (SLA, security, features…); crawled
  competitor pages are excluded so a competitor's figures are never presented as ours.
* search_competitor_docs — crawled public pages of Vietnamese competitors (category
  "Competitor"), optionally for one provider.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from langchain_core.tools import tool

from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_query
from app.retrieval.ranking import eligible_documents, lexical_query, reciprocal_rank_fusion
from app.tools.reranker import RERANK_MIN_SCORE, rank_documents

logger = logging.getLogger(__name__)

CANDIDATES = 15            # vector-search candidates handed to the reranker
TOP_K = 4                  # chunks returned to the agent
VECTOR_THRESHOLD = 0.3     # loose: the reranker does the real filtering
COMPETITOR_CATEGORY = "Competitor"
COMPETITOR_PROVIDERS = ("vng", "fpt", "bizfly", "vnpt", "viettelidc", "viettel")


@dataclass
class RetrievalResult:
    documents: list[dict]
    method: str
    rerank_status: str
    threshold: float
    notes: list[str] = field(default_factory=list)


class VectorCandidates(list):
    def __init__(self, rows: list[dict], *, legacy: bool = False):
        super().__init__(rows)
        self.legacy = legacy


def _missing_rpc(exc: Exception) -> bool:
    text = str(exc).lower()
    return getattr(exc, "code", None) == "PGRST202" or any(
        marker in text for marker in ("function does not exist", "could not find the function")
    )


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
        rows = supabase.rpc("match_cloud_documents", {
            **params, "exclude_categories": exclude, "filter_provider": provider}).execute().data or []
        return VectorCandidates(rows)
    except Exception as exc:  # noqa: BLE001 - old function signature (migration 003 not run)
        if not _missing_rpc(exc):
            raise
        logger.warning("Legacy document RPC in use; apply migration 003 (code=%s).",
                       getattr(exc, "code", "missing_function"))
    params["match_count"] = CANDIDATES * 4
    rows = supabase.rpc("match_cloud_documents", params).execute().data or []
    rows = [r for r in rows if not exclude or r.get("category") not in exclude]
    if provider:
        rows = [r for r in rows if (r.get("metadata") or {}).get("provider") == provider]
    return VectorCandidates(rows[:CANDIDATES], legacy=True)


def retrieve_documents(query: str, *, category: str | None = None,
                       exclude: list[str] | None = None, provider: str | None = None,
                       hybrid: bool | None = None, top_k: int = TOP_K,
                       min_score: float = RERANK_MIN_SCORE) -> RetrievalResult:
    """The shared production/evaluation path: scoped candidates -> RRF -> rerank."""
    notes = []
    try:
        vector = _vector_search(query, category=category, exclude=exclude, provider=provider)
        if getattr(vector, "legacy", False):
            notes.append("Legacy vector search in use; migration 003 is needed for the current retrieval backend.")
    except Exception as exc:  # noqa: BLE001 - lexical retrieval may still find evidence
        logger.warning("Vector retrieval unavailable: %s", exc)
        vector = []
        notes.append("Semantic retrieval unavailable; coverage may be incomplete.")
    keyword = []
    method = "vector"
    enabled = hybrid if hybrid is not None else os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() not in {"0", "false", "no"}
    terms = lexical_query(query)
    if enabled and terms:
        try:
            keyword = supabase.rpc("search_cloud_documents_lexical", {
                "query_text": terms, "match_count": CANDIDATES, "filter_category": category,
                "exclude_categories": exclude, "filter_provider": provider,
            }).execute().data or []
            method = "hybrid"
        except Exception as exc:  # noqa: BLE001 - migration 004 is optional during rollout
            logger.warning("Keyword retrieval unavailable: %s", exc)
            notes.append("Keyword retrieval unavailable; using semantic candidates only.")
    vector = eligible_documents(vector, category=category, exclude=exclude, provider=provider)
    keyword = eligible_documents(keyword, category=category, exclude=exclude, provider=provider)
    candidates = reciprocal_rank_fusion(vector, keyword, limit=CANDIDATES * 2)
    ranked = rank_documents(query, candidates, top_k, min_score)
    if ranked.status == "disabled":
        notes.append("Relevance checking is disabled. These are unverified candidates, not confirmed answers.")
    elif ranked.status == "unavailable":
        notes.append("Relevance checking unavailable. No passages are accepted for factual claims.")
    return RetrievalResult(ranked.documents, method, ranked.status, ranked.threshold, notes)


def _format(result: RetrievalResult, title: str) -> str:
    lines = [f"### {title}:",
             f"Retrieval: {result.method}; relevance check: {result.rerank_status}.",
             "Relevance scores are not probabilities of factual correctness.",
             "Answer only the parts supported by cited passages; explicitly identify missing information."]
    if result.rerank_status == "scored":
        lines.append(f"Relevance cutoff: {result.threshold:g} (requires calibration on reviewed queries).")
    lines.extend(f"Coverage note: {note}" for note in result.notes)
    for idx, doc in enumerate(result.documents, 1):
        meta = doc.get("metadata") or {}
        source = meta.get("source") or "?"
        fetched = f" | fetched {meta['fetched_at'][:10]}" if meta.get("fetched_at") else ""
        score = f"{doc['rerank_score']:.2f}" if "rerank_score" in doc else "unscored"
        section = f" | Section: {meta['section_path']}" if meta.get("section_path") else ""
        lines.append(
            f"\n[{idx}] Service: {doc.get('service_name')} | Category: {doc.get('category')} | "
            f"Source: {source}{section}{fetched} | Relevance: {score}\n{doc.get('content')}"
        )
    return "\n".join(lines)


def _empty_result(result: RetrievalResult, message: str) -> str:
    notes = " ".join(result.notes)
    return f"{message} {notes} State what information is missing; do not guess.".strip()


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
    result = retrieve_documents(query, category=category, exclude=[COMPETITOR_CATEGORY])
    if not result.documents and category and result.rerank_status != "unavailable":
        # The LLM's category guess may not exist in the corpus (e.g. no "SLA" documents yet):
        # search everything rather than wrongly concluding there is no information.
        result = retrieve_documents(query, exclude=[COMPETITOR_CATEGORY])
        result.notes.append(f"No accepted evidence in category {category!r}; broader passages may answer only part of the question.")
    if not result.documents:
        return _empty_result(result, "No relevant documents found in the database for this question.")
    return _format(result, "DOCUMENT SEARCH RESULTS")


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
    result = retrieve_documents(query, category=COMPETITOR_CATEGORY, provider=provider)
    if not result.documents:
        return _empty_result(result, "No relevant competitor pages found for this question.")
    return _format(result, "COMPETITOR PAGE RESULTS (public website data — cite the source and date)")
