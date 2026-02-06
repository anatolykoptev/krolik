"""
krolik - Proactive AI bot with memU memory and dynamic LLM routing.
"""

__version__ = "0.2.0"
__logo__ = "🐰"


def __getattr__(name):
    """Lazy imports for heavy modules to keep startup fast."""
    if name == "AgentLoop":
        from krolik.agent.loop import AgentLoop
        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "__logo__", "AgentLoop"]
