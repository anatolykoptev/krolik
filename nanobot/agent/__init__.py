"""Agent core module."""

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader

# Lazy import for AgentLoop to avoid circular dependency:
# loop.py imports krolik.llm/mcp tools → which import nanobot.agent.tools.base
# → which triggers this __init__ → back to loop.py
def __getattr__(name):
    if name == "AgentLoop":
        from nanobot.agent.loop import AgentLoop
        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]
