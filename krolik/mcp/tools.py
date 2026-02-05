"""MCP tool wrapper for nanobot's tool registry."""

import json
from typing import Any, Optional

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult
from krolik.mcp.client import MCPManager


class MCPTool(Tool):
    """
    Wrapper for MCP tools to work with nanobot's tool registry.
    
    Each MCPTool represents one tool from an MCP server.
    """
    
    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_schema: dict[str, Any],
        mcp_manager: MCPManager
    ):
        self.server_name = server_name
        self.tool_name = tool_name
        self.mcp_manager = mcp_manager
        
        # Extract tool metadata from schema
        self._name = f"{server_name}_{tool_name}"
        self._description = tool_schema.get("description", f"MCP tool: {tool_name}")
        self._parameters = tool_schema.get("parameters", {
            "type": "object",
            "properties": {}
        })
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters
    
    async def execute(self, **arguments) -> ToolResult:
        """Execute the MCP tool."""
        try:
            # Call via MCP manager
            full_tool_name = f"{self.server_name}.{self.tool_name}"
            result = await self.mcp_manager.call_tool(full_tool_name, arguments)
            
            # Check for errors
            if "error" in result:
                return ToolResult(
                    success=False,
                    error=result["error"]
                )
            
            # Format result
            output = result.get("content", result.get("result", result))
            if isinstance(output, (dict, list)):
                output = json.dumps(output, indent=2, ensure_ascii=False)
            
            return ToolResult(
                success=True,
                output=str(output)
            )
            
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return ToolResult(success=False, error=str(e))


class MCPProxyTool(Tool):
    """
    Generic proxy tool for dynamic MCP calls.
    
    Allows calling any MCP tool via a single tool interface.
    """
    
    name = "mcp_call"
    description = """Call any MCP (Model Context Protocol) server tool.

Usage: Provide server name and tool name with arguments.
Example: {"server": "memos", "tool": "search_memories", "args": {"query": "important"}}

Available servers will be listed in tool parameters.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "MCP server name (e.g., 'memos', 'gdrive')"
            },
            "tool": {
                "type": "string",
                "description": "Tool name on the server (e.g., 'search_memories')"
            },
            "args": {
                "type": "object",
                "description": "Tool arguments as JSON object",
                "default": {}
            }
        },
        "required": ["server", "tool"]
    }
    
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
    
    async def execute(self, server: str, tool: str, args: dict = None) -> ToolResult:
        """Execute MCP call via proxy."""
        try:
            args = args or {}
            full_tool_name = f"{server}.{tool}"
            
            result = await self.mcp_manager.call_tool(full_tool_name, args)
            
            if "error" in result:
                return ToolResult(
                    success=False,
                    error=result["error"]
                )
            
            output = result.get("content", result.get("result", result))
            if isinstance(output, (dict, list)):
                output = json.dumps(output, indent=2, ensure_ascii=False)
            
            return ToolResult(
                success=True,
                output=str(output)
            )
            
        except Exception as e:
            logger.error(f"MCP proxy call failed: {e}")
            return ToolResult(success=False, error=str(e))


class MCPListTool(Tool):
    """Tool to list available MCP servers and their tools."""
    
    name = "mcp_list"
    description = """List all available MCP servers and their tools.

Use this to discover what MCP tools are available before calling them.
"""
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
    
    async def execute(self) -> ToolResult:
        """List all MCP tools."""
        try:
            servers = self.mcp_manager.get_available_servers()
            tools = self.mcp_manager.get_all_tools()
            
            output_parts = ["Available MCP Servers:", ""]
            
            for server in servers:
                output_parts.append(f"📡 {server}")
            
            if tools:
                output_parts.extend(["", "Available Tools:", ""])
                for tool in tools:
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "No description")
                    output_parts.append(f"• {name}: {desc}")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts)
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register_mcp_tools(
    tool_registry,
    mcp_manager: MCPManager,
    register_individual: bool = True
) -> None:
    """
    Register MCP tools with nanobot's tool registry.
    
    Args:
        tool_registry: The ToolRegistry to register with
        mcp_manager: The MCPManager instance
        register_individual: Also register individual MCP tools
    """
    # Always register proxy tools
    tool_registry.register(MCPProxyTool(mcp_manager))
    tool_registry.register(MCPListTool(mcp_manager))
    
    # Optionally register individual tools
    if register_individual:
        for tool_schema in mcp_manager.get_all_tools():
            full_name = tool_schema.get("name", "")
            original_name = tool_schema.get("original_name", "")
            
            # Parse server from prefixed name "server.tool_name"
            if "." in full_name and original_name:
                server = full_name.split(".", 1)[0]
                tool = MCPTool(server, original_name, tool_schema, mcp_manager)
                tool_registry.register(tool)
    
    logger.info("MCP tools registered")
