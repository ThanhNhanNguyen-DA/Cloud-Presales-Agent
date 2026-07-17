"""
RAG tools for semantic document search using Supabase vector similarity.
"""

from langchain_core.tools import tool
from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_text


@tool
def search_cloud_documents(
    query: str,
    category: str = None,
    match_threshold: float = 0.6,
    match_count: int = 3
) -> str:
    """
    Searches for technical, SLA, security, or legal documentation relevant to the question.
    
    Args:
        query: The question or topic to search for.
        category: Filter by document category ('SLA', 'Security', 'Compliance', 'Feature'), leave blank to search all.
        match_threshold: Minimum similarity threshold (0-1).
        match_count: Maximum number of results to return.
        
    Returns:
        Formatted search results with document content and similarity scores.
    """
    query_embedding = embed_text(query)
    response = supabase.rpc(
        "match_cloud_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_category": category
        }
    ).execute()

    results = response.data
    if not results:
        return "No relevant documents found in the database."

    formatted = "### DOCUMENT SEARCH RESULTS:\n"
    for idx, doc in enumerate(results):
        formatted += f"\n[{idx+1}] Service: {doc['service_name']} | Category: {doc['category']} | Similarity: {doc['similarity']:.2f}\n{doc['content']}\n"
    return formatted
