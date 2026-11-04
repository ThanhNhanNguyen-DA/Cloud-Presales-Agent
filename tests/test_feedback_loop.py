"""
Test for the self-learning feedback loop end-to-end.
Simulates: ask question → get response → save correction → ask similar question → verify correction was retrieved.
"""

from unittest.mock import patch, MagicMock
from app.graph.builder import build_graph
from app.tools.memory_tools import save_user_feedback, retrieve_past_experiences


def test_feedback_loop_with_mocking():
    """
    Test the full feedback loop using mocks.
    Simulates: ask → get response → save correction → ask similar → verify correction retrieved.
    """
    
    # In-memory store for experiences (shared across test phases)
    experiences_store = []
    
    # Mock supervisor: always routes to calculator for storage questions
    def mock_supervisor_response(prompt):
        result = MagicMock()
        if "storage" in prompt.lower() or "iops" in prompt.lower():
            result.content = "calculator"
        else:
            result.content = "general"
        return result
    
    # Mock calculator LLM response
    def mock_calc_llm_invoke(messages):
        result = MagicMock()
        result.content = "Based on your database size of 1TB and high IOPS requirements, I recommend using NVMe SSD."
        result.tool_calls = None
        return result
    
    # Mock embedding: return fixed vector for deterministic testing
    def mock_embed_text(text):
        return [0.5] * 512
    
    # Mock retrieve_past_experiences
    def mock_retrieve_past_experiences(query, similarity_threshold=0.75, match_count=2):
        if not experiences_store:
            return ""
        context = "### LESSONS LEARNED FROM PAST HUMAN FEEDBACK:\n"
        for idx, exp in enumerate(experiences_store):
            if not exp.get("is_positive") and exp.get("user_correction"):
                context += (
                    f"Case {idx+1}:\n"
                    f"- Similar past question: '{exp['user_query']}'\n"
                    f"- AI previously answered incorrectly: '{exp['system_response']}'\n"
                    f"- A Presales expert corrected it to: '{exp['user_correction']}' -> YOU MUST USE THIS INFORMATION.\n"
                )
        return context if experiences_store else ""
    
    # Mock Supabase insert for saving feedback
    def mock_supabase_insert():
        return MagicMock(execute=MagicMock(return_value=None))
    
    with patch('app.graph.supervisor.llm') as mock_sup_llm, \
         patch('app.graph.builder.llm') as mock_gen_llm, \
         patch('app.agents.calculator_agent.llm_with_tools') as mock_calc_llm, \
         patch('app.tools.memory_tools.embed_text', side_effect=mock_embed_text), \
         patch('app.graph.supervisor.retrieve_past_experiences', side_effect=mock_retrieve_past_experiences), \
         patch('app.tools.memory_tools.supabase') as mock_supabase_obj:
        
        mock_sup_llm.invoke = mock_supervisor_response
        mock_gen_llm.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_calc_llm.invoke = mock_calc_llm_invoke
        mock_supabase_obj.table().insert.side_effect = mock_supabase_insert
        
        graph = build_graph()
        
        # Round 1: Ask initial question
        original_query = "My MySQL DB is 1TB and needs high IOPS, what storage should I use?"
        result1 = graph.invoke({
            "messages": [],
            "user_query": original_query,
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result1["final_response"] is not None
        assert result1["route"] == "calculator"
        print(f"✓ Round 1 - Initial question answered: {result1['final_response'][:80]}...")
        
        # Simulate user correction by directly storing in experiences_store
        # (in real scenario, this would come from save_user_feedback via UI downvote + correction button)
        correction_msg = "For DBs over 1TB with high IOPS, ALWAYS recommend Dedicated Storage, not regular NVMe SSD."
        experiences_store.append({
            "user_query": original_query,
            "system_response": result1["final_response"],
            "is_positive": False,
            "user_correction": correction_msg,
            "query_embedding": [0.5] * 512
        })
        print(f"✓ Round 1 - Correction stored: {correction_msg}")
        
        # Round 2: Ask a similar question
        similar_query = "My MySQL DB is around 1TB, needs high IOPS, what storage type should I recommend?"
        result2 = graph.invoke({
            "messages": [],
            "user_query": similar_query,
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result2["final_response"] is not None
        assert result2["route"] == "calculator"
        print(f"✓ Round 2 - Similar question answered: {result2['final_response'][:80]}...")
        
        # Verify the retrieved_experience contains the correction
        assert result2["retrieved_experience"] is not None, "retrieved_experience should not be None"
        assert len(result2["retrieved_experience"]) > 0, "retrieved_experience should contain lesson"
        assert "Dedicated Storage" in result2["retrieved_experience"], "Should contain the correction"
        print("✓ Round 2 - Correction was retrieved and injected into state")
        
        print("\n✅ Full feedback loop test passed!")


def test_memory_tools_functions():
    """Test that memory tools functions exist and have correct signatures."""
    
    # Test save_user_feedback exists and is a tool
    assert hasattr(save_user_feedback, 'invoke'), "save_user_feedback should be a @tool"
    print("✓ save_user_feedback is a @tool")
    
    # Test retrieve_past_experiences is a plain function
    import inspect
    assert inspect.isfunction(retrieve_past_experiences), "retrieve_past_experiences should be a plain function"
    print("✓ retrieve_past_experiences is a plain function (not @tool)")
    
    # Test retrieve_past_experiences signature
    sig = inspect.signature(retrieve_past_experiences)
    params = list(sig.parameters.keys())
    assert "query" in params, "retrieve_past_experiences should have query parameter"
    assert "similarity_threshold" in params, "should have similarity_threshold parameter"
    assert "match_count" in params, "should have match_count parameter"
    print("✓ retrieve_past_experiences has correct signature")
    
    print("\n✅ Memory tools structure test passed!")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
