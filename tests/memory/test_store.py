"""Tests for krolik memory store with fallback."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from krolik.memory.store import EnhancedMemoryStore


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_store(temp_workspace):
    """Create an EnhancedMemoryStore instance for testing."""
    return EnhancedMemoryStore(
        workspace=temp_workspace,
        memu_url="http://localhost:8000"
    )


@pytest.mark.asyncio
async def test_memory_store_file_fallback_read_write(temp_workspace):
    """Test file-based fallback operations."""
    store = EnhancedMemoryStore(temp_workspace)
    
    # Test writing to today's file
    store.append_today("Test memory content")
    
    # Test reading back
    content = store.read_today()
    assert "Test memory content" in content
    
    # Test long-term memory
    store.write_long_term("Important fact: User likes Python")
    long_term = store.read_long_term()
    assert "likes Python" in long_term


@pytest.mark.asyncio
async def test_memory_store_memu_available(temp_workspace):
    """Test that memU is used when available."""
    store = EnhancedMemoryStore(temp_workspace)
    
    with patch.object(store._memu, 'health_check', return_value=True):
        with patch.object(store._memu, 'memorize', return_value=True) as mock_memorize:
            messages = [{"role": "user", "content": "Hello"}]
            
            result = await store.memorize(messages, category="conversation")
            
            assert result is True
            mock_memorize.assert_called_once()


@pytest.mark.asyncio
async def test_memory_store_fallback_when_memu_unavailable(temp_workspace):
    """Test file fallback when memU is down."""
    store = EnhancedMemoryStore(temp_workspace)
    
    with patch.object(store._memu, 'health_check', return_value=False):
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ]
        
        result = await store.memorize(messages, category="conversation")
        
        assert result is True
        # Check file was written
        today_content = store.read_today()
        assert "Test message" in today_content


@pytest.mark.asyncio
async def test_memory_store_retrieve_with_memu(temp_workspace):
    """Test retrieval via memU."""
    store = EnhancedMemoryStore(temp_workspace)
    
    mock_results = [
        {"content": "Memory 1", "score": 0.9, "category": "fact"},
        {"content": "Memory 2", "score": 0.8, "category": "fact"}
    ]
    
    with patch.object(store._memu, 'health_check', return_value=True):
        with patch.object(store._memu, 'retrieve', return_value=mock_results):
            results = await store.retrieve("test query")
            
            assert len(results) == 2
            assert results[0]["content"] == "Memory 1"


@pytest.mark.asyncio
async def test_memory_store_retrieve_fallback(temp_workspace):
    """Test file-based retrieval fallback."""
    store = EnhancedMemoryStore(temp_workspace)
    
    # Pre-populate with some content
    store.append_today("Important meeting at 3 PM")
    store.append_today("Remember to call John")
    
    with patch.object(store._memu, 'health_check', return_value=False):
        results = await store.retrieve("meeting")
        
        # Should return file content as fallback
        assert len(results) > 0
        assert "meeting" in results[0]["content"].lower()


@pytest.mark.asyncio
async def test_memory_store_recent_memories(temp_workspace):
    """Test getting recent memories."""
    store = EnhancedMemoryStore(temp_workspace)
    
    # Add content for today and yesterday (simulated)
    store.append_today("Today's note")
    store.write_long_term("Long term fact")
    
    recent = store.get_recent_memories(days=7)
    
    assert "Today's note" in recent


def test_memory_store_list_files(temp_workspace):
    """Test listing memory files."""
    store = EnhancedMemoryStore(temp_workspace)
    
    # Create some memory files
    store.append_today("Note 1")
    
    files = store.list_memory_files()
    
    assert len(files) >= 1


def test_memory_context_formatting(temp_workspace):
    """Test memory context formatting."""
    store = EnhancedMemoryStore(temp_workspace)
    
    store.write_long_term("User prefers dark mode")
    store.append_today("Working on krolik tests")
    
    context = store.get_memory_context()
    
    assert "Long-term Memory" in context
    assert "Today's Notes" in context
    assert "dark mode" in context
