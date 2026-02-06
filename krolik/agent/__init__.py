"""Agent core module."""

from krolik.agent.context import ContextBuilder
from krolik.agent.skills import SkillsLoader

# Lazy import for AgentLoop to avoid circular dependency:
# loop.py imports krolik.llm/mcp tools → which import krolik.agent.tools.base
# → which triggers this __init__ → back to loop.py
def __getattr__(name):
    if name == "AgentLoop":
        from krolik.agent.loop import AgentLoop
        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AgentLoop", "ContextBuilder", "SkillsLoader"]
