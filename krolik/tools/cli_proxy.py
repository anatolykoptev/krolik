"""CLI Proxy API for agent connection."""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from krolik.agent.tools.base import Tool, ToolResult


class CLIProxyTool(Tool):
    """
    Tool for connecting to agents via CLI Proxy API.
    
    This allows krolik to act as a proxy for other CLI-based agents,
    forwarding commands and receiving responses through a standardized interface.
    """
    
    name = "cli_proxy"
    description = """Connect to and control external agents via CLI Proxy API.

Use this to:
- Spawn external CLI agents as sub-processes
- Send commands to connected agents
- Receive responses from agent processes
- Manage agent lifecycle (start/stop/restart)

The CLI Proxy API provides a standardized interface for communicating
with CLI-based AI agents through stdin/stdout or socket connections.

Example: cli_proxy(agent_type="mcp-server", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/path"])
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["spawn", "send", "receive", "status", "kill"],
                "description": "Action to perform"
            },
            "agent_id": {
                "type": "string",
                "description": "Agent identifier (required for send/receive/kill)"
            },
            "agent_type": {
                "type": "string",
                "description": "Type of agent to spawn (e.g., 'mcp-server', 'subagent')"
            },
            "command": {
                "type": "string",
                "description": "Command to execute when spawning"
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments for the command",
                "default": []
            },
            "message": {
                "type": "string",
                "description": "Message to send to agent"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds",
                "default": 30
            }
        },
        "required": ["action"]
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agents: dict[str, dict[str, Any]] = {}
        self._counter = 0
    
    async def execute(
        self,
        action: str,
        agent_id: str | None = None,
        agent_type: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        message: str | None = None,
        timeout: int = 30
    ) -> ToolResult:
        """Execute CLI proxy action."""
        try:
            if action == "spawn":
                return await self._spawn_agent(agent_type or "cli-agent", command, args or [], timeout)
            elif action == "send":
                return await self._send_to_agent(agent_id or "", message or "", timeout)
            elif action == "receive":
                return await self._receive_from_agent(agent_id or "", timeout)
            elif action == "status":
                return self._get_status()
            elif action == "kill":
                return await self._kill_agent(agent_id or "")
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"CLI proxy error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _spawn_agent(self, agent_type: str, command: str | None, args: list[str], timeout: int) -> ToolResult:
        """Spawn a new CLI agent process."""
        import subprocess
        
        self._counter += 1
        agent_id = f"{agent_type}-{self._counter}"
        
        if not command:
            return ToolResult(success=False, error="Command is required to spawn agent")
        
        try:
            # Start process
            process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.agents[agent_id] = {
                "type": agent_type,
                "process": process,
                "status": "running",
                "created_at": asyncio.get_running_loop().time()
            }
            
            logger.info(f"Spawned agent {agent_id} with PID {process.pid}")
            
            return ToolResult(
                success=True,
                output=f"✅ Spawned agent '{agent_id}' (PID: {process.pid})\nType: {agent_type}\nCommand: {command} {' '.join(args)}"
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to spawn agent: {e}")
    
    async def _send_to_agent(self, agent_id: str, message: str, timeout: int) -> ToolResult:
        """Send message to agent stdin."""
        if agent_id not in self.agents:
            return ToolResult(success=False, error=f"Agent '{agent_id}' not found")
        
        agent = self.agents[agent_id]
        process = agent.get("process")
        
        if not process or process.poll() is not None:
            return ToolResult(success=False, error=f"Agent '{agent_id}' is not running")
        
        try:
            # Send message to stdin
            if process.stdin:
                process.stdin.write(message + "\n")
                process.stdin.flush()
            
            return ToolResult(
                success=True,
                output=f"📤 Sent to {agent_id}: {message[:100]}..."
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to send: {e}")
    
    async def _receive_from_agent(self, agent_id: str, timeout: int) -> ToolResult:
        """Receive output from agent stdout."""
        if agent_id not in self.agents:
            return ToolResult(success=False, error=f"Agent '{agent_id}' not found")
        
        agent = self.agents[agent_id]
        process = agent.get("process")
        
        if not process or process.poll() is not None:
            return ToolResult(success=False, error=f"Agent '{agent_id}' is not running")
        
        try:
            # Read available output (non-blocking)
            output_lines = []
            
            # Use select for non-blocking read if available
            import select
            if hasattr(select, 'select') and process.stdout:
                ready, _, _ = select.select([process.stdout], [], [], timeout)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
            
            output = "\n".join(output_lines) if output_lines else "(no output yet)"
            
            return ToolResult(
                success=True,
                output=f"📥 Received from {agent_id}:\n{output}"
            )
            
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to receive: {e}")
    
    def _get_status(self) -> ToolResult:
        """Get status of all agents."""
        if not self.agents:
            return ToolResult(
                success=True,
                output="No active agents. Spawn one with cli_proxy(action='spawn', ...)"
            )
        
        lines = ["Active agents:", ""]
        for agent_id, info in self.agents.items():
            process = info.get("process")
            pid = process.pid if process else "N/A"
            status = "running" if process and process.poll() is None else "stopped"
            lines.append(f"• {agent_id} (PID: {pid}) - {status}")
        
        return ToolResult(success=True, output="\n".join(lines))
    
    async def _kill_agent(self, agent_id: str) -> ToolResult:
        """Kill an agent process."""
        if agent_id not in self.agents:
            return ToolResult(success=False, error=f"Agent '{agent_id}' not found")
        
        agent = self.agents[agent_id]
        process = agent.get("process")
        
        if process and process.poll() is None:
            try:
                process.terminate()
                await asyncio.sleep(1)
                if process.poll() is None:
                    process.kill()
                
                del self.agents[agent_id]
                
                return ToolResult(
                    success=True,
                    output=f"🛑 Killed agent '{agent_id}'"
                )
                
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to kill agent: {e}")
        else:
            del self.agents[agent_id]
            return ToolResult(
                success=True,
                output=f"Agent '{agent_id}' was already stopped"
            )


class AgentConnectTool(Tool):
    """
    Tool to connect to external agent services via API.
    
    Provides HTTP/REST API connectivity for agent communication.
    """
    
    name = "agent_connect"
    description = """Connect to external agent services via HTTP API.

Use this to:
- Connect to remote agent endpoints
- Send requests to agent services
- Receive responses from API-based agents
- Manage agent service connections

Supports REST APIs, SSE streams, and WebSocket connections.

Example: agent_connect(url="http://localhost:3000/agent", method="POST", payload={"task": "analyze"})
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Agent service URL"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET"
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers",
                "default": {}
            },
            "payload": {
                "type": "object",
                "description": "Request body for POST/PUT",
                "default": {}
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds",
                "default": 30
            },
            "stream": {
                "type": "boolean",
                "description": "Stream response (for SSE)",
                "default": False
            }
        },
        "required": ["url"]
    }
    
    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        payload: dict | None = None,
        timeout: int = 30,
        stream: bool = False
    ) -> ToolResult:
        """Connect to agent service."""
        try:
            import httpx
            
            headers = headers or {}
            payload = payload or {}
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, json=payload, headers=headers)
                elif method == "PUT":
                    response = await client.put(url, json=payload, headers=headers)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return ToolResult(success=False, error=f"Unsupported method: {method}")
                
                response.raise_for_status()
                
                # Try to parse as JSON
                try:
                    data = response.json()
                    output = json.dumps(data, indent=2, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    output = response.text
                
                return ToolResult(
                    success=True,
                    output=f"✅ {method} {url}\n\nStatus: {response.status_code}\n\n{output[:2000]}"
                )
                
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            )
        except Exception as e:
            logger.error(f"Agent connect error: {e}")
            return ToolResult(success=False, error=str(e))
