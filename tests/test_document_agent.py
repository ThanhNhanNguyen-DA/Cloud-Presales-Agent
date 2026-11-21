"""
Tests for the Document Agent.
Uses mocking to avoid requiring a valid Groq API key and Supabase data.
"""

import re
from unittest.mock import patch, MagicMock
from app.graph.builder import build_graph


def mock_supervisor_for_document(prompt):
    """Mock supervisor routes document-related queries to document agent."""
    if "sla" in prompt.lower() or "documentation" in prompt.lower() or "security" in prompt.lower() or "compliance" in prompt.lower():
        return MagicMock(content="document")
    return MagicMock(content="general")


def mock_document_llm_invoke(messages):
    """Mock document agent LLM response - returns answer without tool calls."""
    result = MagicMock()
    result.content = "Theo tài liệu, Object Storage đảm bảo 99.9% thời gian hoạt động hàng tháng. Nếu không đạt được, khách hàng sẽ nhận được khoản tín dụng phí dịch vụ 10% cho tháng đó."
    result.tool_calls = None
    return result


def test_document_agent_sla_question():
    """Test document agent with SLA question."""
    with patch('app.graph.supervisor.llm') as mock_sup, \
         patch('app.graph.builder.llm') as mock_gen, \
         patch('app.agents.document_agent.llm_with_tools') as mock_doc_llm:
        
        mock_sup.invoke = mock_supervisor_for_document
        mock_gen.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_doc_llm.invoke = mock_document_llm_invoke
        
        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "user_query": "What is the SLA documentation for Object Storage?",
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result["route"] == "document"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert re.search(r'\d', result["final_response"]), "Response should contain numbers"
        print(f"✓ Document SLA test passed. Response: {result['final_response'][:100]}...")


def test_document_agent_security_question():
    """Test document agent with security question."""
    with patch('app.graph.supervisor.llm') as mock_sup, \
         patch('app.graph.builder.llm') as mock_gen, \
         patch('app.agents.document_agent.llm_with_tools') as mock_doc_llm:
        
        mock_sup.invoke = mock_supervisor_for_document
        mock_gen.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_doc_llm.invoke = mock_document_llm_invoke
        
        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "user_query": "What security certifications does Cloud Server have?",
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result["route"] == "document"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        print(f"✓ Document security test passed. Response: {result['final_response'][:100]}...")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
