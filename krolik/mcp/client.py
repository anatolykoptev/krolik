"""MCP (Model Context Protocol) client for krolik.

Allows krolik to connect to external MCP servers (like memos, google-drive, etc.)
Similar to OpenClaw's mcporter integration.
"""

import asyncio
import json
from typing import Any, Optional, Callable
from urllib.parse import urlparse

import httpx
from loguru import logger


class MCPClient:
    """
    Client for Model Context Protocol (MCP) servers.
    
    Supports:
    - SSE (Server-Sent Events) transport
    - HTTP POST for tool calls
    - Dynamic tool discovery
    """
    
    def __init__(self, name: str, url: str, timeout: float = 30.0):
        self.name = name
        self.url = url
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._tools: list[dict[str, Any]] = []
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Connect to MCP server and discover tools."""
        try:
            # Try to connect via SSE endpoint
            parsed = urlparse(self.url)
            sse_url = f"{self.url}/sse" if not self.url.endswith('/sse') else self.url
            
            # For now, do a simple health check
            # Full SSE implementation would stream events
            response = await self._client.get(sse_url, timeout=5.0)
            
            if response.status_code in [200, 405]:  # 405 = method not allowed, but server exists
                self._initialized = True
                logger.info(f"MCP client '{self.name}' connected to {self.url}")
                
                # Discover tools (if endpoint available)
                await self._discover_tools()
                return True
            else:
                logger.warning(f"MCP server {self.name} returned {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"MCP client '{self.name}' failed to connect: {e}")
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from MCP server."""
        try:
            # Try to get tools list via common endpoints
            endpoints = [
                f"{self.url}/tools",
                f"{self.url}/list_tools",
                f"{self.url}/mcp/tools"
            ]
            
            for endpoint in endpoints:
                try:
                    response = await self._client.get(endpoint, timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        self._tools = data.get("tools", data.get("result", []))
                        logger.info(f"Discovered {len(self._tools)} tools from {self.name}")
                        return
                except Exception:
                    continue
            
            # If no discovery endpoint, assume standard MCP tools
            self._tools = []
            
        except Exception as e:
            logger.debug(f"Tool discovery failed for {self.name}: {e}")
            self._tools = []
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if not self._initialized:
            raise RuntimeError(f"MCP client '{self.name}' not initialized")
        
        try:
            # Standard MCP tool call format
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            
            # Try common endpoints
            endpoints = [
                f"{self.url}/call",
                f"{self.url}/tools/{tool_name}",
                f"{self.url}/mcp/call",
                f"{self.url}/invoke"
            ]
            
            for endpoint in endpoints:
                try:
                    response = await self._client.post(
                        endpoint,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 404:
                        continue  # Try next endpoint
                    else:
                        return {
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "status_code": response.status_code
                        }
                        
                except httpx.RequestError:
                    continue  # Try next endpoint
            
            return {"error": "No valid MCP endpoint found"}
            
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {"error": str(e)}
    
    def get_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools."""
        return self._tools
    
    def is_available(self) -> bool:
        """Check if client is connected."""
        return self._initialized
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


class MCPManager:
    """Manages multiple MCP server connections."""
    
    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._tool_to_client: dict[str, str] = {}  # tool_name -> client_name
    
    def add_server(self, name: str, url: str) -> MCPClient:
        """Add an MCP server."""
        client = MCPClient(name, url)
        self._clients[name] = client
        return client
    
    async def initialize_all(self) -> dict[str, bool]:
        """Initialize all MCP connections."""
        results = {}
        
        for name, client in self._clients.items():
            success = await client.initialize()
            results[name] = success
            
            if success:
                # Map tools to client
                for tool in client.get_tools():
                    tool_name = tool.get("name", tool.get("id"))
                    if tool_name:
                        full_name = f"{name}.{tool_name}"
                        self._tool_to_client[full_name] = name
        
        return results
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool by name (with server prefix)."""
        
        # Parse server.tool_name format
        if "." in tool_name:
            server_name, actual_tool = tool_name.split(".", 1)
            
            if server_name in self._clients:
                client = self._clients[server_name]
                if not client.is_available():
                    return {"error": f"MCP server '{server_name}' is not connected"}
                return await client.call_tool(actual_tool, arguments)
        
        # Try to find in tool mapping
        if tool_name in self._tool_to_client:
            client_name = self._tool_to_client[tool_name]
            client = self._clients[client_name]
            if not client.is_available():
                return {"error": f"MCP server '{client_name}' is not connected"}
            return await client.call_tool(tool_name, arguments)
        
        return {"error": f"Tool '{tool_name}' not found in any MCP server"}
    
    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all available tools from all servers."""
        all_tools = []
        
        for client_name, client in self._clients.items():
            for tool in client.get_tools():
                # Prefix tool name with server
                tool_copy = tool.copy()
                tool_name = tool.get("name", tool.get("id"))
                if tool_name:
                    tool_copy["name"] = f"{client_name}.{tool_name}"
                    tool_copy["original_name"] = tool_name
                    all_tools.append(tool_copy)
        
        return all_tools
    
    def get_available_servers(self) -> list[str]:
        """Get list of connected servers."""
        return [
            name for name, client in self._clients.items() 
            if client.is_available()
        ]
    
    async def close_all(self):
        """Close all connections."""
        for client in self._clients.values():
            await client.close()


# Convenience factory
async def create_mcp_manager(config: dict[str, str]) -> MCPManager:
    """
    Create MCP manager from config.
    
    Args:
        config: Dict of {server_name: url}
        
    Returns:
        Initialized MCPManager
    """
    manager = MCPManager()
    
    for name, url in config.items():
        manager.add_server(name, url)
    
    await manager.initialize_all()
    return manager
