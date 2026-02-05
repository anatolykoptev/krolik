"""Tests for krolik MCP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from krolik.mcp.client import MCPClient, MCPManager, create_mcp_manager


@pytest.fixture
def mcp_client():
    """Create an MCPClient instance for testing."""
    return MCPClient(name="test-server", url="http://localhost:3001")


@pytest.mark.asyncio
async def test_mcp_client_initialize_success(mcp_client):
    """Test successful MCP connection."""
    with patch.object(mcp_client._client, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = await mcp_client.initialize()
        
        assert result is True
        assert mcp_client.is_available() is True


@pytest.mark.asyncio
async def test_mcp_client_initialize_failure(mcp_client):
    """Test MCP connection failure."""
    with patch.object(mcp_client._client, 'get') as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        
        result = await mcp_client.initialize()
        
        assert result is False
        assert mcp_client.is_available() is False


@pytest.mark.asyncio
async def test_mcp_client_call_tool_success(mcp_client):
    """Test successful tool call."""
    # Initialize first
    mcp_client._initialized = True
    
    with patch.object(mcp_client._client, 'post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Success",
            "content": "Tool executed successfully"
        }
        mock_post.return_value = mock_response
        
        result = await mcp_client.call_tool("test_tool", {"arg1": "value1"})
        
        assert "error" not in result
        assert result["result"] == "Success"


@pytest.mark.asyncio
async def test_mcp_client_call_tool_not_initialized(mcp_client):
    """Test tool call when not initialized."""
    mcp_client._initialized = False
    
    # Should raise RuntimeError
    try:
        result = await mcp_client.call_tool("test_tool", {})
        # If no exception, should return error dict
        assert "error" in result
    except RuntimeError as e:
        # Expected behavior - not initialized
        assert "not initialized" in str(e).lower()


@pytest.mark.asyncio
async def test_mcp_manager_add_and_initialize():
    """Test MCP manager with multiple servers."""
    manager = MCPManager()
    
    # Add servers
    client1 = manager.add_server("server1", "http://localhost:3001")
    client2 = manager.add_server("server2", "http://localhost:3002")
    
    # Mock initialization
    with patch.object(client1, 'initialize', return_value=True):
        with patch.object(client2, 'initialize', return_value=True):
            with patch.object(client1, '_discover_tools'):
                with patch.object(client2, '_discover_tools'):
                    results = await manager.initialize_all()
    
    assert "server1" in results
    assert "server2" in results
    assert results["server1"] is True
    assert results["server2"] is True


@pytest.mark.asyncio
async def test_mcp_manager_call_tool():
    """Test calling tool via manager."""
    manager = MCPManager()
    client = manager.add_server("memos", "http://localhost:3001")
    
    with patch.object(client, 'call_tool', return_value={"result": "ok"}) as mock_call:
        client._initialized = True
        
        result = await manager.call_tool("memos.search", {"query": "test"})
        
        assert result["result"] == "ok"
        mock_call.assert_called_once_with("search", {"query": "test"})


@pytest.mark.asyncio
async def test_mcp_manager_call_tool_not_found():
    """Test calling non-existent tool."""
    manager = MCPManager()
    
    result = await manager.call_tool("unknown.tool", {})
    
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_mcp_manager_factory():
    """Test MCP manager factory function."""
    config = {
        "memos": "http://localhost:3001",
        "gdrive": "http://localhost:3002"
    }
    
    with patch.object(MCPManager, 'initialize_all', return_value={"memos": True, "gdrive": True}):
        manager = await create_mcp_manager(config)
    
    assert "memos" in manager._clients
    assert "gdrive" in manager._clients


def test_mcp_manager_get_available_servers():
    """Test getting list of connected servers."""
    manager = MCPManager()
    client1 = manager.add_server("server1", "http://localhost:3001")
    client2 = manager.add_server("server2", "http://localhost:3002")
    
    # Mock availability
    client1._initialized = True
    client2._initialized = False
    
    available = manager.get_available_servers()
    
    assert "server1" in available
    assert "server2" not in available


def test_mcp_manager_get_all_tools():
    """Test getting all tools from all servers."""
    manager = MCPManager()
    client = manager.add_server("server1", "http://localhost:3001")
    
    # Mock tools
    client._tools = [
        {"name": "tool1", "description": "Tool 1"},
        {"name": "tool2", "description": "Tool 2"}
    ]
    
    all_tools = manager.get_all_tools()
    
    assert len(all_tools) == 2


@pytest.mark.asyncio
async def test_mcp_manager_close_all():
    """Test closing all connections."""
    manager = MCPManager()
    client = manager.add_server("server1", "http://localhost:3001")
    
    with patch.object(client, 'close') as mock_close:
        await manager.close_all()
        
        mock_close.assert_called_once()
