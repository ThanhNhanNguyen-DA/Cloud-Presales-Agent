"""RAG pipeline tests: query embedding, candidate search, reranking, competitor separation."""

from unittest.mock import MagicMock, patch

import pytest

from app.tools import rag_tools, reranker


def _doc(i, score=0.8, category="Feature", provider=None, content=None):
    return {"id": str(i), "service_name": f"svc{i}", "category": category, "similarity": score,
            "content": content or f"content {i}", "metadata": {"source": f"doc{i}.docx",
                                                              **({"provider": provider} if provider else {})}}


@pytest.fixture
def rpc(monkeypatch):
    monkeypatch.setenv("HYBRID_SEARCH_ENABLED", "false")
    with patch.object(rag_tools, "embed_query", return_value=[0.0] * 512), \
         patch.object(rag_tools, "supabase") as sb:
        yield sb


def test_document_search_excludes_competitor_pages(rpc):
    rpc.rpc.return_value.execute.return_value = MagicMock(data=[_doc(1)])
    with patch.object(reranker, "rerank_enabled", return_value=False):
        out = rag_tools.search_cloud_documents.invoke({"query": "SLA?"})
    params = rpc.rpc.call_args.args[1]
    assert params["exclude_categories"] == ["Competitor"] and params["match_count"] == 15
    assert "svc1" in out and "Source: doc1.docx" in out


def test_falls_back_to_old_rpc_signature_and_filters_in_python(rpc):
    old_rows = [_doc(1), _doc(2, category="Competitor"), _doc(3)]
    rpc.rpc.return_value.execute.side_effect = [RuntimeError("function does not exist"),
                                                MagicMock(data=old_rows)]
    with patch.object(reranker, "rerank_enabled", return_value=False):
        out = rag_tools.search_cloud_documents.invoke({"query": "q"})
    assert "exclude_categories" not in rpc.rpc.call_args.args[1]
    assert "svc2" not in out and "svc1" in out and "svc3" in out
    assert "Legacy vector search in use" in out


def test_reranker_reorders_and_drops_irrelevant(rpc):
    rpc.rpc.return_value.execute.return_value = MagicMock(data=[_doc(1), _doc(2), _doc(3)])
    fake = MagicMock()
    fake.predict.return_value = [0.02, 0.95, 0.40]
    with patch.object(reranker, "get_reranker", return_value=fake):
        out = rag_tools.search_cloud_documents.invoke({"query": "q"})
    assert out.index("svc2") < out.index("svc3") and "svc1" not in out
    assert "Relevance: 0.95" in out


def test_nothing_relevant_says_so(rpc):
    rpc.rpc.return_value.execute.return_value = MagicMock(data=[_doc(1)])
    fake = MagicMock()
    fake.predict.return_value = [0.001]
    with patch.object(reranker, "get_reranker", return_value=fake):
        assert "No relevant documents" in rag_tools.search_cloud_documents.invoke({"query": "weather?"})


def test_reranker_failure_does_not_bypass_relevance_check():
    docs = [_doc(1), _doc(2)]
    with patch.object(reranker, "get_reranker", side_effect=RuntimeError("CUDA OOM")):
        result = reranker.rank_documents("q", docs, top_k=1)
        assert result.documents == [] and result.status == "unavailable"


def test_competitor_search_filters_category_and_provider(rpc):
    rpc.rpc.return_value.execute.return_value = MagicMock(
        data=[{**_doc(1, category="Competitor", provider="bizfly"),
               "metadata": {"source": "https://bizflycloud.vn/cloud-server/bang-gia",
                            "provider": "bizfly", "fetched_at": "2026-11-05T01:00:00"}}])
    with patch.object(reranker, "rerank_enabled", return_value=False):
        out = rag_tools.search_competitor_docs.invoke({"query": "giá cloud server", "provider": "Bizfly"})
    params = rpc.rpc.call_args.args[1]
    assert params["filter_category"] == "Competitor" and params["filter_provider"] == "bizfly"
    assert "fetched 2026-11-05" in out and "cite the source" in out


def test_competitor_search_rejects_unknown_provider(rpc):
    out = rag_tools.search_competitor_docs.invoke({"query": "q", "provider": "aws"})
    assert "Unknown provider" in out
    rpc.rpc.assert_not_called()


def test_llm_cannot_set_thresholds_any_more():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    props = convert_to_openai_tool(rag_tools.search_cloud_documents)["function"]["parameters"]["properties"]
    assert set(props) == {"query", "category"}


def test_embed_query_uses_the_query_prompt():
    from app.tools import embedding_tool
    model = MagicMock()
    model.prompts = {"query": "Instruct: ..."}
    model.encode.return_value = MagicMock(tolist=lambda: [0.1])
    with patch.object(embedding_tool, "get_embedding_model", return_value=model):
        embedding_tool.embed_query("hello")
    assert model.encode.call_args.kwargs["prompt_name"] == "query"


def test_competitor_agent_has_the_new_tool():
    from app.agents.competitor_agent import TOOLS
    assert "search_competitor_docs" in {t.name for t in TOOLS}


def test_empty_category_filter_retries_without_it(rpc):
    rpc.rpc.return_value.execute.side_effect = [MagicMock(data=[]), MagicMock(data=[_doc(7)])]
    with patch.object(reranker, "rerank_enabled", return_value=False):
        out = rag_tools.search_cloud_documents.invoke({"query": "SLA?", "category": "SLA"})
    first, second = (c.args[1] for c in rpc.rpc.call_args_list)
    assert first["filter_category"] == "SLA" and second["filter_category"] is None
    assert "svc7" in out
