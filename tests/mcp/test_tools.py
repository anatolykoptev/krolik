"""Tests for krolik MCP tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from krolik.mcp.tools import MCPTool, MCPProxyTool, MCPListTool, register_mcp_tools
from krolik.mcp.client import MCPManager


@pytest.fixture
def mock_mcp_manager():
    """Create a mock MCP manager."""
    manager = MagicMock(spec=MCPManager)
    manager.call_tool = AsyncMock(return_value={"result": "success"})
    manager.get_all_tools = MagicMock(return_value=[])
    manager.get_available_servers = MagicMock(return_value=["memos", "gdrive"])
    return manager


@pytest.mark.asyncio
async def test_mcp_tool_execute_success(mock_mcp_manager):
    """Test successful MCP tool execution."""
    tool_schema = {
        "name": "search_memories",
        "description": "Search memories",
        "parameters": {"type": "object", "properties": {}}
    }
    
    tool = MCPTool("memos", "search_memories", tool_schema, mock_mcp_manager)
    
    result = await tool.execute(query="test")
    
    assert result.success is True
    mock_mcp_manager.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_tool_execute_with_error(mock_mcp_manager):
    """Test MCP tool execution with error."""
    mock_mcp_manager.call_tool = AsyncMock(return_value={"error": "Tool failed"})
    
    tool_schema = {"name": "test_tool", "description": "Test", "parameters": {}}
    tool = MCPTool("server", "test_tool", tool_schema, mock_mcp_manager)
    
    result = await tool.execute()
    
    assert result.success is False
    assert "Tool failed" in result.error


@pytest.mark.asyncio
async def test_mcp_proxy_tool_execute(mock_mcp_manager):
    """Test MCP proxy tool."""
    mock_mcp_manager.call_tool = AsyncMock(return_value={"content": "Result"})
    
    tool = MCPProxyTool(mock_mcp_manager)
    result = await tool.execute(server="memos", tool="search", args={"q": "test"})
    
    assert result.success is True
    mock_mcp_manager.call_tool.assert_called_once_with("memos.search", {"q": "test"})


@pytest.mark.asyncio
async def test_mcp_list_tool_execute(mock_mcp_manager):
    """Test MCP list tool."""
    mock_mcp_manager.get_all_tools.return_value = [
        {"name": "memos.tool1", "description": "Tool 1"},
        {"name": "gdrive.tool2", "description": "Tool 2"}
    ]
    
    tool = MCPListTool(mock_mcp_manager)
    result = await tool.execute()
    
    assert result.success is True
    assert "memos" in result.output
    assert "gdrive" in result.output


def test_register_mcp_tools(mock_mcp_manager):
    """Test registering MCP tools with registry."""
    registry = MagicMock()
    registry.register = MagicMock()
    
    mock_mcp_manager.get_all_tools.return_value = [
        {"name": "server.tool1", "description": "Tool 1", "original_server": "server"}
    ]
    
    register_mcp_tools(registry, mock_mcp_manager, register_individual=True)
    
    # Should register proxy tools + individual tools
    assert registry.register.call_count >= 2  # mcp_call, mcp_list + at least one individual


def test_mcp_tool_name_formatting():
    """Test that MCP tool names are formatted correctly."""
    tool_schema = {"name": "search", "description": "Search tool"}
    tool = MCPTool("memos", "search", tool_schema, MagicMock())
    
    assert tool.name == "memos_search"


def test_mcp_tool_description():
    """Test MCP tool description extraction."""
    tool_schema = {"name": "test", "description": "Test description", "parameters": {}}
    tool = MCPTool("server", "test", tool_schema, MagicMock())
    
    assert tool.description == "Test description"


def test_mcp_tool_parameters():
    """Test MCP tool parameter schema."""
    tool_schema = {
        "name": "test",
        "description": "Test",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            }
        }
    }
    tool = MCPTool("server", "test", tool_schema, MagicMock())
    
    assert "properties" in tool.parameters
    assert "query" in tool.parameters["properties"]
