"""
Tests for the Calculator Agent using the real tools.
Uses mocking to avoid requiring a valid Groq API key.
"""

import re
from unittest.mock import patch, MagicMock
from app.graph.builder import build_graph


def mock_supervisor_for_calculator(prompt):
    """Mock supervisor always routes to calculator."""
    result = MagicMock()
    result.content = "calculator"
    return result


def mock_llm_with_tools_backup(messages):
    """
    Mock LLM response for backup question.
    Returns tool call to calculate_backup_and_cost.
    """
    result = MagicMock()
    result.content = None
    result.tool_calls = [
        {
            "id": "call_1",
            "name": "calculate_backup_and_cost",
            "args": {
                "data_volume_gb": 50,
                "bandwidth_mbps": 200,
                "storage_class": "Standard"
            }
        }
    ]
    return result


def mock_llm_with_tools_backup_response(messages):
    """
    Mock LLM second invocation after tool result - returns final answer.
    """
    result = MagicMock()
    result.content = "Dựa trên tính toán, việc sao lưu 50GB dữ liệu qua kết nối 200Mbps sẽ mất khoảng 1.95 giờ (khoảng 117 phút). Chi phí lưu trữ hàng tháng cho Standard tier là 1.15 USD."
    result.tool_calls = None
    return result


def mock_llm_with_tools_vm(messages):
    """
    Mock LLM response for VM sizing question.
    Returns tool call to size_web_app_vm.
    """
    result = MagicMock()
    result.content = None
    result.tool_calls = [
        {
            "id": "call_2",
            "name": "size_web_app_vm",
            "args": {
                "ccu": 3000,
                "app_type": "java"
            }
        }
    ]
    return result


def mock_llm_with_tools_vm_response(messages):
    """
    Mock LLM second invocation after tool result - returns final answer for VM sizing.
    """
    result = MagicMock()
    result.content = "Cho ứng dụng Java với 3000 người dùng đồng thời, tôi khuyên dùng máy ảo có 48 vCPU và 48GB RAM."
    result.tool_calls = None
    return result


def mock_llm_with_tools_storage(messages):
    """
    Mock LLM response for storage question.
    Returns tool call to size_database_storage.
    """
    result = MagicMock()
    result.content = None
    result.tool_calls = [
        {
            "id": "call_3",
            "name": "size_database_storage",
            "args": {
                "current_size_gb": 500,
                "monthly_growth_rate_percent": 15,
                "months": 6,
                "required_iops": 5000
            }
        }
    ]
    return result


def mock_llm_with_tools_storage_response(messages):
    """
    Mock LLM second invocation after tool result - returns final answer for storage.
    """
    result = MagicMock()
    result.content = "Cơ sở dữ liệu MySQL của bạn dự kiến sẽ phát triển lên 1,206.39GB sau 6 tháng. Với yêu cầu 5000 IOPS, tôi khuyên dùng NVMe SSD có hiệu suất cao."
    result.tool_calls = None
    return result


def test_calculator_agent_backup_question():
    """Test calculator agent with backup question."""
    call_count = [0]
    
    def mock_llm_invoke(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_llm_with_tools_backup(messages)
        else:
            return mock_llm_with_tools_backup_response(messages)
    
    with patch('app.graph.supervisor.llm') as mock_sup, \
         patch('app.graph.builder.llm') as mock_gen, \
         patch('app.agents.calculator_agent.llm_with_tools') as mock_tools_llm, \
         patch('app.graders.self_rag_grader.grader_llm') as mock_grader_llm:

        mock_sup.invoke = mock_supervisor_for_calculator
        mock_gen.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_tools_llm.invoke = mock_llm_invoke
        mock_grader_llm.invoke = MagicMock(return_value=MagicMock(content="PASS"))
        
        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "user_query": "I have 50GB of data to back up over a 200Mbps connection, how long will it take and what's the monthly Standard storage cost?",
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result["route"] == "calculator"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert re.search(r'\d', result["final_response"]), "Response should contain numbers"
        print(f"✓ Backup test passed. Response: {result['final_response'][:100]}...")


def test_calculator_agent_vm_sizing_question():
    """Test calculator agent with VM sizing question."""
    call_count = [0]
    
    def mock_llm_invoke(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_llm_with_tools_vm(messages)
        else:
            return mock_llm_with_tools_vm_response(messages)
    
    with patch('app.graph.supervisor.llm') as mock_sup, \
         patch('app.graph.builder.llm') as mock_gen, \
         patch('app.agents.calculator_agent.llm_with_tools') as mock_tools_llm, \
         patch('app.graders.self_rag_grader.grader_llm') as mock_grader_llm:

        mock_sup.invoke = mock_supervisor_for_calculator
        mock_gen.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_tools_llm.invoke = mock_llm_invoke
        mock_grader_llm.invoke = MagicMock(return_value=MagicMock(content="PASS"))
        
        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "user_query": "My Java web app has 3000 CCU, what VM configuration should I use?",
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result["route"] == "calculator"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert re.search(r'\d', result["final_response"]), "Response should contain numbers"
        print(f"✓ VM sizing test passed. Response: {result['final_response'][:100]}...")


def test_calculator_agent_storage_sizing_question():
    """Test calculator agent with storage sizing question."""
    call_count = [0]
    
    def mock_llm_invoke(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_llm_with_tools_storage(messages)
        else:
            return mock_llm_with_tools_storage_response(messages)
    
    with patch('app.graph.supervisor.llm') as mock_sup, \
         patch('app.graph.builder.llm') as mock_gen, \
         patch('app.agents.calculator_agent.llm_with_tools') as mock_tools_llm, \
         patch('app.graders.self_rag_grader.grader_llm') as mock_grader_llm:

        mock_sup.invoke = mock_supervisor_for_calculator
        mock_gen.invoke = MagicMock(return_value=MagicMock(content="test"))
        mock_tools_llm.invoke = mock_llm_invoke
        mock_grader_llm.invoke = MagicMock(return_value=MagicMock(content="PASS"))
        
        graph = build_graph()
        result = graph.invoke({
            "messages": [],
            "user_query": "My MySQL DB is currently 500GB, growing 15%/month over 6 months, needs 5000 IOPS, what storage should I recommend?",
            "route": None,
            "retrieved_experience": None,
            "agent_response": None,
            "final_response": None
        })
        
        assert result["route"] == "calculator"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert re.search(r'\d', result["final_response"]), "Response should contain numbers"
        print(f"✓ Storage sizing test passed. Response: {result['final_response'][:100]}...")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
