"""
Smoke test for the LangGraph skeleton.
Tests that the graph compiles and runs end-to-end with 4 different query types.
Uses mocking to avoid requiring a valid Groq API key.
"""

from unittest.mock import patch, MagicMock
from app.graph.builder import build_graph


def test_graph_runs_end_to_end():
    """Test that the graph compiles and processes all 4 query types."""
    
    # Mock the LLM responses
    def mock_supervisor_response(prompt):
        """Mock supervisor routing based on the LATEST question only (not the prompt template,
        which itself mentions "backup duration" and used to send every query to calculator)."""
        question = prompt.split("LATEST question:", 1)[1].splitlines()[0].lower()
        if "backup" in question or "duration" in question:
            return MagicMock(content="calculator")
        elif "sla" in question or "documentation" in question:
            return MagicMock(content="document")
        elif "aws" in question or "equivalent" in question:
            return MagicMock(content="competitor")
        else:
            return MagicMock(content="general")
    
    def mock_general_response(messages):
        """Mock general node response. `messages` is [System, *history, Human]."""
        query = getattr(messages[-1], "content", str(messages)) if isinstance(messages, list) else str(messages)
        if "backup" in query.lower():
            return MagicMock(content="The backup will take approximately 0.78 hours (about 47 minutes) for a 10GB backup over a 100Mbps connection.")
        elif "sla" in query.lower():
            return MagicMock(content="Object Storage SLA guarantees 99.95% availability with 99.99% durability for stored objects.")
        elif "aws" in query.lower() or "s3" in query.lower():
            return MagicMock(content="AWS S3 is equivalent to our Object Storage service. Both provide scalable object storage with similar pricing and features.")
        else:
            return MagicMock(content="Hello! How can I assist you with your cloud infrastructure questions today?")
    
    test_queries = [
        ("Hello", "general"),
        ("How long does a 10GB backup take over a 100Mbps connection?", "calculator"),
        ("What is the SLA documentation for Object Storage?", "document"),
        ("What service is equivalent to AWS S3 on our platform?", "competitor")
    ]
    
    with patch('app.graph.supervisor.llm') as mock_supervisor_llm, \
         patch('app.graph.builder.llm') as mock_builder_llm, \
         patch('app.agents.calculator_agent.llm_with_tools') as mock_calc_llm, \
         patch('app.agents.document_agent.llm_with_tools') as mock_doc_llm, \
         patch('app.agents.competitor_agent.llm_with_tools') as mock_comp_llm:
        
        mock_supervisor_llm.invoke = mock_supervisor_response
        mock_builder_llm.invoke = mock_general_response
        
        # Mock calculator agent LLM
        mock_calc_response = MagicMock()
        mock_calc_response.content = "Based on the provided information, the backup will take approximately 0.39 hours (about 23 minutes) and cost 0.23 USD per month."
        mock_calc_response.tool_calls = None
        mock_calc_llm.invoke = MagicMock(return_value=mock_calc_response)
        mock_doc_llm.invoke = MagicMock(return_value=MagicMock(
            content="The Object Storage SLA is 99.9% monthly uptime.", tool_calls=None))
        mock_comp_llm.invoke = MagicMock(return_value=MagicMock(
            content="**AWS S3 → Object Storage**.", tool_calls=None))
        
        graph = build_graph()
        
        for query, expected_route in test_queries:
            result = graph.invoke({
                "messages": [],
                "user_query": query,
                "route": None,
                "retrieved_experience": None,
                "agent_response": None,
                "final_response": None
            })
            
            # Assertions
            assert result["final_response"] is not None, f"final_response is None for query: {query}"
            assert len(result["final_response"]) > 0, f"final_response is empty for query: {query}"
            assert result["route"] == expected_route, f"wrong route for query: {query}"
            
            print(f"✓ Query: '{query}'")
            print(f"  Route: {result['route']}")
            print(f"  Response (first 100 chars): {result['final_response'][:100]}...")
            print()


if __name__ == "__main__":
    test_graph_runs_end_to_end()
    print("All smoke tests passed! ✅")

