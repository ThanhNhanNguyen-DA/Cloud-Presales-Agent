"""Hybrid recall, evidence provenance and partial-answer behaviour."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.graders import self_rag_grader as grader
from app.retrieval.ranking import eligible_documents, lexical_query, reciprocal_rank_fusion
from app.tools import memory_tools, rag_tools, reranker


def doc(identifier, **overrides):
    row = {"id": identifier, "service_name": "ELB", "category": "Feature",
           "content": f"Passage {identifier}", "similarity": 0.6,
           "metadata": {"source": f"{identifier}.docx"}}
    return {**row, **overrides}


def test_rrf_rewards_agreement_without_duplicate_votes():
    one, two = doc("one"), doc("two")
    fused = reciprocal_rank_fusion([one, two], [two, two])
    assert [d["id"] for d in fused] == ["two", "one"]
    assert fused[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1]["similarity"] == 0.6


def test_keyword_terms_are_literal_multilingual_and_keep_product_codes():
    query = lexical_query('ELB có hỗ trợ TLS 1.3 không? "OR" -AWS')
    assert '"elb"' in query and '"tls"' in query and '"aws"' in query
    assert '-AWS' not in query and '"không"' not in query
    assert lexical_query("Bạn có giúp tôi không?") == ""


def test_provenance_and_scope_rechecked_after_retrieval():
    rows = [doc("own"), doc("sample", metadata={"source": "sample_data"}),
            doc("case-study", category="CaseStudy"), doc("missing", metadata={}),
            doc("pending", metadata={"source": "draft.pdf", "review_status": "pending"})]
    assert [d["id"] for d in eligible_documents(rows, exclude=["CaseStudy"])] == ["own"]


def test_hybrid_recovers_a_passage_missed_by_vector_and_reranks_it(monkeypatch):
    monkeypatch.setenv("HYBRID_SEARCH_ENABLED", "true")
    vector, keyword = doc("vector"), doc("keyword")
    model = MagicMock()
    model.predict.side_effect = lambda pairs, **kwargs: [0.02 if "vector" in text else 0.94 for _, text in pairs]
    with patch.object(rag_tools, "_vector_search", return_value=[vector]), \
         patch.object(rag_tools, "supabase") as sb, patch.object(reranker, "get_reranker", return_value=model):
        sb.rpc.return_value.execute.return_value = MagicMock(data=[keyword])
        result = rag_tools.retrieve_documents("ELB TLS", exclude=["CaseStudy"])
    assert result.method == "hybrid" and result.rerank_status == "scored"
    assert [d["id"] for d in result.documents] == ["keyword"]
    assert sb.rpc.call_args.args[1]["exclude_categories"] == ["CaseStudy"]


def test_keyword_failure_is_visible_and_vector_results_still_checked():
    model = MagicMock()
    model.predict.return_value = [0.8]
    with patch.object(rag_tools, "_vector_search", return_value=[doc("vector")]), \
         patch.object(rag_tools, "supabase") as sb, patch.object(reranker, "get_reranker", return_value=model):
        sb.rpc.side_effect = RuntimeError("function does not exist")
        result = rag_tools.retrieve_documents("ELB TLS", hybrid=True)
    assert result.method == "vector" and result.documents
    assert "Keyword retrieval unavailable" in result.notes[0]


def test_embedding_failure_can_use_scored_keyword_evidence():
    model = MagicMock()
    model.predict.return_value = [0.8]
    with patch.object(rag_tools, "_vector_search", side_effect=RuntimeError("embedding unavailable")), \
         patch.object(rag_tools, "supabase") as sb, patch.object(reranker, "get_reranker", return_value=model):
        sb.rpc.return_value.execute.return_value = MagicMock(data=[doc("keyword")])
        result = rag_tools.retrieve_documents("ELB", hybrid=True)
    assert result.documents and "Semantic retrieval unavailable" in result.notes[0]


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [2.0], []])
def test_invalid_scores_do_not_accept_evidence(scores):
    model = MagicMock()
    model.predict.return_value = scores
    with patch.object(reranker, "get_reranker", return_value=model):
        result = reranker.rank_documents("ELB", [doc("one")], 4)
    assert result.documents == [] and result.status == "unavailable"


def test_duplicates_do_not_crowd_out_distinct_evidence():
    model = MagicMock()
    model.predict.return_value = [0.95, 0.94, 0.9]
    with patch.object(reranker, "get_reranker", return_value=model):
        result = reranker.rank_documents("ELB", [doc("one"), doc("copy", content="Passage one"), doc("two")], 2)
    assert [d["id"] for d in result.documents] == ["one", "two"]
    assert result.documents[0]["duplicate_ids"] == ["copy"]


def test_disabled_relevance_check_is_disclosed_not_presented_as_probability():
    with patch.object(rag_tools, "_vector_search", return_value=[doc("one")]), \
         patch.object(reranker, "rerank_enabled", return_value=False):
        result = rag_tools.retrieve_documents("ELB", hybrid=False)
    text = rag_tools._format(result, "RESULTS")
    assert "Relevance: unscored" in text and "not probabilities" in text
    assert "disabled" in text and "missing information" in text


def test_unrelated_database_error_does_not_retry_legacy_rpc():
    with patch.object(rag_tools, "embed_query", return_value=[0.0] * 512), \
         patch.object(rag_tools, "supabase") as sb:
        sb.rpc.side_effect = RuntimeError("database unavailable")
        with pytest.raises(RuntimeError):
            rag_tools._vector_search("ELB")
    assert sb.rpc.call_count == 1


def test_specialist_without_tool_evidence_is_not_verified():
    with patch.object(grader, "grader_llm") as judge:
        result = grader.grounding_grader_node({"route": "document", "final_response": "SLA 100%"})
    assert result["grading_passed"] is False
    assert result["retry_count"] == grader.MAX_GRADING_ATTEMPTS
    assert grader.UNVERIFIED_WARNING in result["final_response"]
    judge.invoke.assert_not_called()


def test_grader_receives_the_same_reviewed_facts_as_the_agent():
    with patch.object(grader, "grader_llm") as judge:
        judge.invoke.return_value = AIMessage(content='{"verdict":"PASS","issues":[]}')
        grader.grounding_grader_node({"raw_tool_output": "Old SLA 99.9%", "final_response": "SLA 99.95%",
                                     "retrieved_experience": "Reviewed SLA is 99.95%"})
    assert "Reviewed SLA is 99.95%" in judge.invoke.call_args.args[0]


def test_missing_review_status_never_enters_memory_prompt():
    with patch.object(memory_tools, "embed_text", return_value=[0.0] * 512), \
         patch.object(memory_tools, "supabase") as sb:
        sb.rpc.return_value.execute.return_value = MagicMock(data=[{
            "is_positive": False, "user_query": "SLA?", "system_response": "old",
            "user_correction": "unreviewed correction"}])
        assert memory_tools.retrieve_past_experiences("SLA?") == ""


# --------------------------------------------------------------------------- Vietnamese lexical query
from app.retrieval.ranking import lexical_query as _lexical  # noqa: E402


def test_lexical_query_uses_vietnamese_phrases_not_single_syllables():
    q = _lexical("Những thuật toán cân bằng tải nào được hỗ trợ?")
    assert '"cân bằng"' in q and '"bằng tải"' in q
    assert '"hỗ"' not in q and '"trợ"' not in q and '"hỗ trợ"' not in q   # common, meaningless


def test_identifiers_take_over_the_keyword_channel():
    # ts_rank_cd has no IDF: a common phrase next to a rare price would drown it.
    assert _lexical("Gói nào giá 116.820 đ/tháng?") == '"116.820"'
    assert _lexical("ELB hỗ trợ những thuật toán cân bằng tải nào?") == '"elb"'


def test_lexical_query_keeps_product_names_and_dotted_numbers():
    assert _lexical("Gói Micro 116.820 đ/tháng") == '"micro" OR "116.820"'
    assert _lexical("SLA uptime của Object Storage") == '"sla" OR "uptime" OR "object" OR "storage"'


def test_lexical_query_phrases_never_cross_stop_words():
    assert '"tải của"' not in _lexical("cân bằng tải của hệ thống")


def test_lexical_query_falls_back_to_lone_syllables():
    assert _lexical("Cân") == '"cân"'
    assert _lexical("là gì của") == ""


def test_expand_aliases():
    from app.retrieval.ranking import expand_aliases
    assert expand_aliases("KMS là gì?") == "KMS (Key Management Service) là gì?"
    assert expand_aliases("Key Management Service KMS") == "Key Management Service KMS"   # already spelled out
    assert expand_aliases("SMS marketing") == "SMS marketing"                              # whole words only
    assert expand_aliases("cấu hình elb") == "cấu hình elb (Elastic Load Balancer)"


def test_lexical_and_mode():
    assert _lexical("Gói 770.000đ của Bizfly", mode="and") == '"770.000" "bizfly"'
    assert _lexical("Gói 770.000đ của Bizfly") == '"770.000" OR "bizfly"'


def test_keyword_channel_tries_and_then_or():
    from unittest.mock import MagicMock, patch

    from app.tools import rag_tools, reranker
    calls = []

    def fake_rpc(name, params):
        if name == "search_cloud_documents_lexical":
            calls.append(params["query_text"])
            data = [] if len(calls) == 1 else [{"id": "k1", "service_name": "s", "category": "Feature",
                                                "content": "c", "metadata": {"source": "x"}}]
        else:
            data = []
        return MagicMock(execute=MagicMock(return_value=MagicMock(data=data)))

    with patch.object(rag_tools, "embed_query", return_value=[0.0] * 512), \
         patch.object(rag_tools, "supabase") as sb, \
         patch.object(reranker, "rerank_enabled", return_value=False):
        sb.rpc.side_effect = fake_rpc
        result = rag_tools.retrieve_documents("Gói 770.000đ Bizfly", hybrid=True)
    assert calls == ['"770.000" "bizfly"', '"770.000" OR "bizfly"']
    assert [d["id"] for d in result.documents] == ["k1"]


# --------------------------------------------------------------------------- clarifications & warning language
def test_clarifying_question_without_tools_gets_no_warning():
    with patch.object(grader, "grader_llm") as judge:
        result = grader.grounding_grader_node({
            "route": "calculator", "user_query": "Backup mất bao lâu?",
            "final_response": "Bạn cho mình biết dung lượng dữ liệu và băng thông nhé?"})
    assert result == {"grading_passed": True, "grader_feedback": None}
    judge.invoke.assert_not_called()


@pytest.mark.parametrize("answer", [
    "Thời gian backup khoảng 16,7 phút. Bạn cần thêm gì không?",   # has a figure
    "Có, dịch vụ hỗ trợ mã hóa dữ liệu.",                           # a claim, not a question
])
def test_unsupported_claims_without_tools_still_warn(answer):
    result = grader.grounding_grader_node({"route": "document", "user_query": "Có mã hóa không?",
                                           "final_response": answer})
    assert result["final_response"].endswith(grader.UNVERIFIED_WARNING_VI)


def test_warning_follows_the_question_language():
    vi = grader.grounding_grader_node({"route": "document", "user_query": "SLA là bao nhiêu?",
                                       "final_response": "SLA 100%"})
    en = grader.grounding_grader_node({"route": "document", "user_query": "What is the SLA?",
                                       "final_response": "SLA 100%"})
    assert vi["final_response"].endswith(grader.UNVERIFIED_WARNING_VI)
    assert en["final_response"].endswith(grader.UNVERIFIED_WARNING)


def test_activity_log_reports_unverified_even_without_tool_data():
    from app.progress import describe_step
    update = {"grading_passed": False, "final_response": "SLA 100%" + grader.UNVERIFIED_WARNING_VI}
    assert describe_step("grader", update, None)["verdict"] == "unverified"
    assert describe_step("grader", {"grading_passed": True, "grader_feedback": None}, None)["verdict"] == "skipped"


def test_store_down_is_an_error_not_an_empty_result():
    """Both channels unreachable must not read as "no documents on this" (it used to)."""
    from app.agents.tool_agent import _run_tool

    down = RuntimeError("[Errno -2] Name or service not known")
    with patch.object(rag_tools, "_vector_search", side_effect=down), patch.object(rag_tools, "supabase") as sb:
        sb.rpc.side_effect = down
        with pytest.raises(RuntimeError, match="Document store unreachable"):
            rag_tools.retrieve_documents("CMC Cloud có những dịch vụ gì?", hybrid=True)
        result = _run_tool({"search_cloud_documents": rag_tools.search_cloud_documents},
                           {"name": "search_cloud_documents", "args": {"query": "CMC Cloud có những dịch vụ gì?"}})
    assert "Document store unreachable" in result["error"]
    with patch.object(rag_tools, "_vector_search", side_effect=down):
        with pytest.raises(RuntimeError):
            rag_tools.retrieve_documents("ELB", hybrid=False)
