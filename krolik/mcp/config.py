"""MCP configuration for krolik."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    
    url: str  # Server URL (e.g., http://localhost:8001/sse)
    enabled: bool = True
    timeout: float = 30.0
    
    # Optional authentication
    api_key: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    
    # Tool filtering
    include_tools: List[str] = Field(default_factory=list)  # Empty = all
    exclude_tools: List[str] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""
    
    enabled: bool = True
    
    # MCP servers
    servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)
    
    # Default servers to auto-connect
    auto_connect: List[str] = Field(default_factory=list)
    
    # Global settings
    default_timeout: float = 30.0
    
    # Tool registration
    register_individual_tools: bool = True  # Register each tool separately
    register_proxy_tools: bool = True  # Register mcp_call, mcp_list
    
    # Examples of common MCP servers:
    # servers = {
    #     "memos": {"url": "http://localhost:8001/sse"},
    #     "gdrive": {"url": "http://localhost:3002/sse"},
    #     "notion": {"url": "http://localhost:3003/sse"}
    # }
