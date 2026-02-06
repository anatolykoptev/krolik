"""Tests for krolik memory client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from krolik.memory.client import MemUClient


@pytest.fixture
def memu_client(tmp_path):
    """Create a MemUClient in HTTP-only mode (embedded service disabled)."""
    client = MemUClient(base_url="http://localhost:8000", api_key="test-key", data_dir=tmp_path, pg_dsn="postgresql://fake:fake@localhost/fake")
    # Force HTTP-only mode: skip embedded service init
    client._service_attempted = True
    client._service = None
    return client


@pytest.mark.asyncio
async def test_memu_client_health_check_success(memu_client):
    """Test health check when memU HTTP service is available."""
    with patch.object(memu_client._http, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = await memu_client.health_check()
        
        assert result is True
        mock_get.assert_called_once_with("http://localhost:8000/health")


@pytest.mark.asyncio
async def test_memu_client_health_check_failure(memu_client):
    """Test health check when memU is unavailable."""
    with patch.object(memu_client._http, 'get') as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        
        result = await memu_client.health_check()
        
        assert result is False


@pytest.mark.asyncio
async def test_memu_client_memorize_success(memu_client):
    """Test successful memory storage via HTTP."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    
    with patch.object(memu_client._http, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await memu_client.memorize(messages, category="conversation")
        
        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8000/memorize"


@pytest.mark.asyncio
async def test_memu_client_memorize_failure(memu_client):
    """Test memory storage failure handling."""
    messages = [{"role": "user", "content": "Test"}]
    
    with patch.object(memu_client._http, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response
        
        result = await memu_client.memorize(messages)
        
        assert result is False


@pytest.mark.asyncio
async def test_memu_client_retrieve_success(memu_client):
    """Test successful memory retrieval via HTTP."""
    with patch.object(memu_client._http, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"content": "Memory 1", "category": "fact", "score": 0.9},
                {"content": "Memory 2", "category": "fact", "score": 0.8}
            ]
        }
        mock_post.return_value = mock_response
        
        results = await memu_client.retrieve("test query", category="fact", limit=5)
        
        assert len(results) == 2
        assert results[0]["content"] == "Memory 1"


@pytest.mark.asyncio
async def test_memu_client_retrieve_empty(memu_client):
    """Test retrieval with no results."""
    with patch.object(memu_client._http, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        results = await memu_client.retrieve("unknown query")
        
        assert results == []


@pytest.mark.asyncio
async def test_memu_client_with_api_key(tmp_path):
    """Test that API key is included in HTTP headers."""
    client = MemUClient(base_url="http://localhost:8000", api_key="secret-key", data_dir=tmp_path, pg_dsn="postgresql://fake:fake@localhost/fake")
    client._service_attempted = True
    client._service = None
    
    with patch.object(client._http, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        await client.retrieve("test")
        
        call_kwargs = mock_post.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer secret-key"
