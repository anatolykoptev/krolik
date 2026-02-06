"""MCP (Model Context Protocol) integration for krolik."""

from krolik.mcp.client import MCPClient, MCPManager, create_mcp_manager
from krolik.mcp.config import MCPConfig, MCPServerConfig

# Lazy imports to avoid circular dependency with krolik.agent.loop
def __getattr__(name):
    if name in ("MCPTool", "MCPProxyTool", "MCPListTool", "register_mcp_tools"):
        from krolik.mcp import tools as _tools
        return getattr(_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MCPClient",
    "MCPManager",
    "create_mcp_manager",
    "MCPTool",
    "MCPProxyTool",
    "MCPListTool",
    "register_mcp_tools",
    "MCPConfig",
    "MCPServerConfig"
]
