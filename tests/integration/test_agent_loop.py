"""Integration tests for krolik agent loop."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from krolik.agent.loop import AgentLoop
from krolik.bus.queue import MessageBus
from krolik.providers.base import LLMProvider


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_bus():
    """Create a mock message bus."""
    bus = MagicMock(spec=MessageBus)
    bus.publish_outbound = AsyncMock()
    return bus


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=MagicMock(
        has_tool_calls=False,
        content="Test response",
        tool_calls=[]
    ))
    return provider


@pytest.mark.asyncio
async def test_agent_loop_initialization(temp_workspace, mock_bus, mock_provider):
    """Test agent loop initializes correctly."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    assert loop.workspace == temp_workspace
    assert loop.model == "test-model"
    assert loop.tools is not None


@pytest.mark.asyncio
async def test_agent_loop_memory_tools_registered(temp_workspace, mock_bus, mock_provider):
    """Test that memory tools are registered."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    # Check memory tools are registered
    assert loop.tools.get("remember") is not None
    assert loop.tools.get("recall") is not None
    assert loop.tools.get("search_memory") is not None


@pytest.mark.asyncio
async def test_agent_loop_intent_retriever_exists(temp_workspace, mock_bus, mock_provider):
    """Test that intent retriever is initialized."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    assert loop.intent_retriever is not None
    assert loop.proactive_suggestions is not None


@pytest.mark.asyncio
async def test_agent_loop_mcp_initialization(temp_workspace, mock_bus, mock_provider):
    """Test MCP initialization in agent loop."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    assert loop.mcp_manager is None  # Not initialized by default
    
    # Initialize MCP
    with patch('krolik.mcp.client.create_mcp_manager') as mock_create:
        mock_manager = MagicMock()
        mock_manager._clients = {"memos": MagicMock(is_available=MagicMock(return_value=True))}
        mock_create.return_value = mock_manager
        
        results = await loop.initialize_mcp({"memos": "http://localhost:3001"})
        
        assert "memos" in results


@pytest.mark.asyncio
async def test_agent_loop_memorize_conversation(temp_workspace, mock_bus, mock_provider):
    """Test background conversation memorization."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    with patch.object(loop.context.memory, 'memorize', new_callable=AsyncMock) as mock_memorize:
        await loop._memorize_conversation("User message", "Assistant response")
        
        mock_memorize.assert_called_once()
        call_args = mock_memorize.call_args[1]
        assert call_args["category"] == "conversation"
        assert call_args["messages"][0]["content"] == "User message"


def test_agent_loop_get_mcp_status(temp_workspace, mock_bus, mock_provider):
    """Test getting MCP status."""
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    # Before initialization
    assert loop.get_mcp_status() == {}


@pytest.mark.asyncio
async def test_agent_loop_process_message_with_memory(temp_workspace, mock_bus, mock_provider):
    """Test message processing with memory integration."""
    from krolik.bus.events import InboundMessage
    
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    # Mock intent retriever
    with patch.object(loop.intent_retriever, 'retrieve_if_needed', new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = [{"content": "Previous context", "score": 0.9}]
        
        with patch.object(loop.intent_retriever, 'format_for_context', return_value="## Context"):
            msg = InboundMessage(
                channel="telegram",
                sender_id="user123",
                chat_id="456",
                content="What did we discuss?"
            )
            
            response = await loop._process_message(msg)
            
            assert response is not None
            mock_retrieve.assert_called_once_with("What did we discuss?")


@pytest.mark.asyncio
async def test_agent_loop_proactive_suggestion(temp_workspace, mock_bus, mock_provider):
    """Test proactive suggestion integration."""
    from krolik.bus.events import InboundMessage
    
    loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=temp_workspace
    )
    
    with patch.object(loop.proactive_suggestions, 'check_for_suggestions', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = "Want me to help with your task?"
        
        msg = InboundMessage(
            channel="telegram",
            sender_id="user123",
            chat_id="456",
            content="I'm working on..."
        )
        
        await loop._process_message(msg)
        
        mock_check.assert_called_once_with("I'm working on...")
