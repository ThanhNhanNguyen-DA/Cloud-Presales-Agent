"""Certifications named in an answer must come from the evidence (found live: an invented ISO 27018 passed the LLM grader)."""

from unittest.mock import MagicMock, patch

import pytest

from app.graders import self_rag_grader as grader

EVIDENCE = "search_cloud_documents(...) -> Cloud Server đạt ISO/IEC 27001 và ISO 9001. S3 Standard: PCI DSS."


@pytest.mark.parametrize("answer, missing", [
    ("Cloud Server đạt ISO 27001, ISO 9001 và PCI-DSS", []),                 # spelling variants match
    ("Cloud Server đạt ISO 27001 và ISO 27018", ["ISO 27018"]),
    ("Đạt SOC 2, HIPAA và Tier III", ["SOC 2", "HIPAA", "Tier III"]),
    ("Uptime 99,99%, hỗ trợ mã hóa", []),
])
def test_unsupported_standards(answer, missing):
    assert grader.unsupported_standards(answer, EVIDENCE) == missing


@pytest.mark.parametrize("evidence", ["ISO 9001, ISO 27017 - 27018", "ISO 27017/27018", "ISO 27017 và 27018"])
def test_iso_ranges_in_the_evidence_cover_each_number(evidence):
    # Found live: the docs write "ISO 27017 - 27018"; a correct "ISO 27018" was flagged as invented.
    assert grader.unsupported_standards("Đạt ISO 27017 và ISO\u202f27018", evidence) == []
    assert grader.unsupported_standards("Đạt ISO 27001", evidence) == ["ISO 27001"]


def test_standards_named_in_the_question_may_be_repeated():
    assert grader.unsupported_standards("Tài liệu không đề cập SOC 2.", EVIDENCE, "Có SOC 2 không?") == []


def _grade(answer, llm_reply='{"verdict": "PASS", "issues": []}', retry_count=0):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=llm_reply)
    with patch.object(grader, "grader_llm", llm), patch.object(grader, "jev_enabled", return_value=False):
        return grader.grounding_grader_node({"route": "document", "user_query": "Cloud Server đạt chuẩn nào?",
                                             "raw_tool_output": EVIDENCE, "final_response": answer,
                                             "retry_count": retry_count})


def test_invented_certification_overrides_an_llm_pass():
    result = _grade("Cloud Server đạt ISO 27001 và ISO 27018.")
    assert result["grading_passed"] is False and result["retry_count"] == 1
    assert "ISO 27018" in result["grader_feedback"]


def test_invented_certification_on_the_last_attempt_warns():
    result = _grade("Cloud Server đạt ISO 27018.", retry_count=1)
    assert result["final_response"].endswith(grader.GRADING_WARNING_VI)


def test_supported_certifications_still_pass():
    assert _grade("Cloud Server đạt ISO 27001 và ISO 9001.") == {"grading_passed": True, "grader_feedback": None}


def test_general_prompt_refuses_competitor_questions_first():
    from app.graph.builder import GENERAL_SYSTEM_PROMPT
    rule = GENERAL_SYSTEM_PROMPT.index("RULE 1")
    assert rule < GENERAL_SYSTEM_PROMPT.index("Otherwise:")
    for name in ("AWS", "VNPT", "VNG", "FPT", "Viettel", "Bizfly"):
        assert name in GENERAL_SYSTEM_PROMPT
