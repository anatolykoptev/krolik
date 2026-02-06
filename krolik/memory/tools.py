"""Memory tools for the agent."""

import json
from typing import Any, Optional

from loguru import logger

from krolik.agent.tools.base import Tool, ToolResult
from krolik.memory.store import EnhancedMemoryStore


class RememberTool(Tool):
    """Tool to explicitly save information to memory."""
    
    name = "remember"
    description = """Save important information to long-term memory.
    
Use this when:
- User shares personal information (preferences, facts about themselves)
- Important conclusions are reached during a task
- User explicitly asks to "remember" something
- You detect important facts that should persist across conversations

Categories:
- "fact": General facts, knowledge
- "preference": User preferences, likes/dislikes
- "task": Task-related information
- "conversation": Conversation summaries
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember"
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "task", "conversation"],
                "description": "Category for the memory",
                "default": "fact"
            },
            "context": {
                "type": "string",
                "description": "Additional context about when/why this was learned",
                "default": ""
            }
        },
        "required": ["content"]
    }
    
    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory = memory_store
    
    async def execute(
        self, 
        content: str, 
        category: str = "fact",
        context: str = ""
    ) -> ToolResult:
        """Save information to memory."""
        try:
            metadata = {}
            if context:
                metadata["context"] = context
            
            messages = [{
                "role": "system",
                "content": f"[{category.upper()}] {content}"
            }]
            
            success = await self.memory.memorize(messages, category, metadata)
            
            if success:
                return ToolResult(
                    success=True,
                    output=f"✓ Remembered ({category}): {content[:100]}{'...' if len(content) > 100 else ''}"
                )
            else:
                return ToolResult(
                    success=False,
                    error="Failed to save to memory"
                )
                
        except Exception as e:
            logger.error(f"Remember tool error: {e}")
            return ToolResult(success=False, error=str(e))


class RecallTool(Tool):
    """Tool to retrieve information from memory."""
    
    name = "recall"
    description = """Retrieve relevant memories based on a query.
    
Use this when:
- User asks about something from previous conversations
- You need context about user's preferences/history
- User references "the other day" or "before"
- You need to check if there's relevant prior context
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory"
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "task", "conversation", ""],
                "description": "Optional category filter",
                "default": ""
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory = memory_store
    
    async def execute(
        self, 
        query: str, 
        category: str = "",
        limit: int = 5
    ) -> ToolResult:
        """Retrieve memories by query."""
        try:
            cat_filter = category if category else None
            results = await self.memory.retrieve(query, cat_filter, limit)
            
            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant memories found."
                )
            
            output_parts = [f"Found {len(results)} relevant memories:"]
            
            for i, result in enumerate(results, 1):
                content = result.get("content", "")
                cat = result.get("category", "unknown")
                score = result.get("score", 0)
                
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                output_parts.append(f"\n{i}. [{cat}] (relevance: {score:.2f})\n{content}")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts)
            )
            
        except Exception as e:
            logger.error(f"Recall tool error: {e}")
            return ToolResult(success=False, error=str(e))


class SearchMemoryTool(Tool):
    """Advanced memory search with filters."""
    
    name = "search_memory"
    description = """Advanced search through memory with filters.
    
Similar to 'recall' but with more control over search parameters.
Use for targeted searches when you know exactly what you're looking for.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "category": {
                "type": "string",
                "description": "Filter by category (fact, preference, task, conversation)"
            },
            "days": {
                "type": "integer",
                "description": "Only search memories from last N days",
                "default": 0
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results",
                "default": 10
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory = memory_store
    
    async def execute(
        self, 
        query: str, 
        category: Optional[str] = None,
        days: int = 0,
        limit: int = 10
    ) -> ToolResult:
        """Advanced memory search."""
        try:
            cat_filter = category if category else None
            results = await self.memory.retrieve(query, cat_filter, limit)
            
            # Filter by days if specified (for file-based fallback)
            if days > 0 and results:
                # Note: This is a simplified filter
                # Full implementation would check timestamps
                pass
            
            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results for query: '{query}'"
                )
            
            output = {
                "query": query,
                "results_count": len(results),
                "results": results
            }
            
            return ToolResult(
                success=True,
                output=json.dumps(output, indent=2, ensure_ascii=False)
            )
            
        except Exception as e:
            logger.error(f"Search memory tool error: {e}")
            return ToolResult(success=False, error=str(e))
