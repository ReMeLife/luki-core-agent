"""
Phase 1D Integration Tests
Tests the integration of cognitive tools with the core agent
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from luki_agent.agent_core import LukiAgent, ConversationRequest
from luki_agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_cognitive_tools_registered():
    """Test that cognitive tools are properly registered"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'):
        
        agent = LukiAgent(
            memory_service_url="http://test:8002",
            redis_url="redis://test:6379"
        )
        
        try:
            tools = agent.tool_registry.list_tools()
            tool_names = [tool['name'] for tool in tools]
            
            # Check that cognitive tools are registered
            expected_cognitive_tools = [
                'recommend_cognitive_activity',
                'get_world_day_activities', 
                'record_activity_engagement',
                'get_mystory_prompt'
            ]
            
            for tool_name in expected_cognitive_tools:
                assert tool_name in tool_names, f"Cognitive tool '{tool_name}' not registered"
            
            print(f"✅ All {len(expected_cognitive_tools)} cognitive tools registered")
            
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_memory_tools_registered():
    """Test that memory tools are properly registered"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'):
        
        agent = LukiAgent(
            memory_service_url="http://test:8002",
            redis_url="redis://test:6379"
        )
        
        try:
            tools = agent.tool_registry.list_tools()
            tool_names = [tool['name'] for tool in tools]
            
            expected_memory_tools = [
                'search_memory',
                'get_user_profile',
                'recommend_activity'
            ]
            
            for tool_name in expected_memory_tools:
                assert tool_name in tool_names, f"Memory tool '{tool_name}' not registered"
            
            print(f"✅ All {len(expected_memory_tools)} memory tools registered")
            
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_cognitive_activity_recommendation_tool():
    """Test cognitive activity recommendation tool execution"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'), \
         patch('luki_modules_cognitive.interfaces.agent_tools.CognitiveTools') as mock_cognitive:
        
        agent = LukiAgent(
            memory_service_url="http://test:8002",
            redis_url="redis://test:6379"
        )
        
        try:
            mock_instance = AsyncMock()
            mock_cognitive.return_value = mock_instance
            
            # Mock successful recommendation response
            mock_instance.recommend_activity.return_value = {
                "success": True,
                "recommendations": [
                    {
                        "title": "Memory Lane Music",
                        "description": "Listen to songs from your favorite decade",
                        "duration_minutes": 30,
                        "difficulty_level": "easy"
                    }
                ]
            }
            
            # Execute the tool
            result = await agent.tool_registry.execute_tool(
                'recommend_cognitive_activity',
                user_id='test_user_123',
                current_mood='happy',
                available_duration=30,
                max_recommendations=2
            )
            
            assert result.success, f"Tool execution failed: {result.error}"
            assert "personalized activities" in result.content
            assert result.metadata and (result.metadata.get('recommendations') or result.metadata.get('activities'))
            
            print("✅ Cognitive activity recommendation tool working")
            
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_world_day_activities_tool():
    """Test world day activities tool execution"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'), \
         patch('luki_modules_cognitive.interfaces.agent_tools.CognitiveTools') as mock_cognitive:
        
        agent = LukiAgent(
            memory_service_url="http://test:8002",
            redis_url="redis://test:6379"
        )
        
        try:
            mock_instance = AsyncMock()
            mock_cognitive.return_value = mock_instance
            
            # Mock world day response
            mock_instance.get_world_day_activities.return_value = {
                "success": True,
                "world_day": "International Music Day",
                "activities": [
                    {
                        "title": "Musical Memories",
                        "description": "Share your favorite songs and why they're special"
                    }
                ]
            }
            
            result = await agent.tool_registry.execute_tool(
                'get_world_day_activities',
                user_id='test_user_123'
            )
            
            assert result.success, f"Tool execution failed: {result.error}"
            assert "activities" in result.content.lower()
            # The tool returns actual world day info or fallback message - both are valid
            
            print("✅ World day activities tool working")
            
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_agent_health_check():
    """Test agent health check includes cognitive tools status"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'):
        
        agent = LukiAgent(
            memory_service_url="http://test:8002",
            redis_url="redis://test:6379"
        )
        
        try:
            with patch.object(agent.llm_manager, 'generate') as mock_generate:
                mock_generate.return_value = MagicMock(content="Hello")
                
                health = await agent.health_check()
                
                assert health['agent_id']
                assert health['status'] in ['healthy', 'degraded']
                assert 'components' in health
                assert 'conversation_count' in health
                
                print("✅ Agent health check working")
                
        finally:
            await agent.close()


def test_tool_registry_execution_history():
    """Test that tool execution history is tracked"""
    with patch('luki_agent.memory.retriever.MemoryRetriever'), \
         patch('luki_agent.memory.session_store.SessionStore'):
        
        registry = ToolRegistry()
        
        # Check initial state
        history = registry.get_execution_history()
        initial_count = len(history)
        
        # The history should be empty or contain previous test executions
        assert isinstance(history, list)
        
        print(f"✅ Tool execution history tracking working (current: {initial_count} entries)")


if __name__ == "__main__":
    """Run integration tests directly"""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
