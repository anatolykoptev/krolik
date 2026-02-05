"""Tests for krolik memory tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from krolik.memory.tools import RememberTool, RecallTool, SearchMemoryTool
from krolik.memory.store import EnhancedMemoryStore


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store."""
    store = MagicMock(spec=EnhancedMemoryStore)
    store.memorize = AsyncMock(return_value=True)
    store.retrieve = AsyncMock(return_value=[])
    return store


@pytest.mark.asyncio
async def test_remember_tool_success(mock_memory_store):
    """Test successful remember operation."""
    tool = RememberTool(mock_memory_store)
    
    result = await tool.execute(
        content="User likes Python programming",
        category="preference",
        context="Mentioned during conversation"
    )
    
    assert result.success is True
    assert "Remembered" in result.output
    assert "preference" in result.output
    mock_memory_store.memorize.assert_called_once()


@pytest.mark.asyncio
async def test_remember_tool_failure(mock_memory_store):
    """Test remember tool when storage fails."""
    mock_memory_store.memorize = AsyncMock(return_value=False)
    tool = RememberTool(mock_memory_store)
    
    result = await tool.execute(content="Test fact")
    
    assert result.success is False
    assert "Failed" in result.error


@pytest.mark.asyncio
async def test_recall_tool_with_results(mock_memory_store):
    """Test recall with found memories."""
    mock_memory_store.retrieve = AsyncMock(return_value=[
        {"content": "Memory 1", "category": "fact", "score": 0.9},
        {"content": "Memory 2", "category": "fact", "score": 0.8}
    ])
    
    tool = RecallTool(mock_memory_store)
    result = await tool.execute(query="test query")
    
    assert result.success is True
    assert "Found 2 relevant memories" in result.output
    assert "Memory 1" in result.output


@pytest.mark.asyncio
async def test_recall_tool_empty_results(mock_memory_store):
    """Test recall with no results."""
    mock_memory_store.retrieve = AsyncMock(return_value=[])
    
    tool = RecallTool(mock_memory_store)
    result = await tool.execute(query="unknown")
    
    assert result.success is True
    assert "No relevant memories" in result.output


@pytest.mark.asyncio
async def test_search_memory_tool(mock_memory_store):
    """Test advanced memory search."""
    mock_memory_store.retrieve = AsyncMock(return_value=[
        {"content": "Task 1", "category": "task"},
        {"content": "Task 2", "category": "task"}
    ])
    
    tool = SearchMemoryTool(mock_memory_store)
    result = await tool.execute(
        query="pending tasks",
        category="task",
        limit=10
    )
    
    assert result.success is True
    assert "results_count" in result.output or "Task 1" in result.output


@pytest.mark.asyncio
async def test_recall_tool_with_category_filter(mock_memory_store):
    """Test recall with category filtering."""
    tool = RecallTool(mock_memory_store)
    
    await tool.execute(query="test", category="preference")
    
    # Verify retrieve was called with category filter
    call_args = mock_memory_store.retrieve.call_args
    assert call_args[1]["category"] == "preference"


def test_tool_parameters_structure():
    """Test that tools have correct parameter schemas."""
    mock_store = MagicMock()
    
    remember = RememberTool(mock_store)
    assert "content" in remember.parameters["properties"]
    assert "category" in remember.parameters["properties"]
    
    recall = RecallTool(mock_store)
    assert "query" in recall.parameters["properties"]
    
    search = SearchMemoryTool(mock_store)
    assert "query" in search.parameters["properties"]
    assert "days" in search.parameters["properties"]
