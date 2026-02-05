"""Proactive memory-based suggestions for the agent."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class ProactiveMemorySuggestions:
    """
    Generate proactive suggestions based on stored memories.
    
    Example:
    - User mentioned wanting to learn Rust
    - Bot proactively suggests: "Want me to find Rust learning resources?"
    """
    
    def __init__(self, memory_store, min_relevance: float = 0.7):
        self.memory = memory_store
        self.min_relevance = min_relevance
        self._suggestion_cache: dict[str, Any] = {}
    
    async def check_for_suggestions(
        self, 
        user_message: str,
        categories: Optional[list[str]] = None
    ) -> Optional[str]:
        """
        Check if there's a relevant proactive suggestion.
        
        Args:
            user_message: Current user message
            categories: Memory categories to check
            
        Returns:
            Suggestion text or None
        """
        categories = categories or ["preference", "task", "fact"]
        
        # Search for relevant past memories
        for category in categories:
            results = await self.memory.retrieve(
                query=user_message,
                category=category,
                limit=3
            )
            
            for result in results:
                score = result.get("score", 0)
                content = result.get("content", "")
                
                if score >= self.min_relevance:
                    # Check if we haven't already suggested this
                    cache_key = f"{category}:{content[:50]}"
                    if cache_key not in self._suggestion_cache:
                        self._suggestion_cache[cache_key] = datetime.now()
                        
                        suggestion = self._generate_suggestion(
                            category, content, user_message
                        )
                        if suggestion:
                            return suggestion
        
        return None
    
    def _generate_suggestion(
        self, 
        category: str, 
        memory_content: str,
        user_message: str
    ) -> Optional[str]:
        """Generate a contextual suggestion based on memory."""
        
        if category == "preference":
            # User preferences - offer related help
            if "rust" in memory_content.lower() and "learn" in memory_content.lower():
                if "project" in user_message.lower():
                    return "I see you were learning Rust. Want me to help with your Rust project?"
            
            if "python" in memory_content.lower():
                if "script" in user_message.lower() or "code" in user_message.lower():
                    return "Since you like Python, want me to write a script for this?"
        
        elif category == "task":
            # Ongoing tasks - check status
            if "deadline" in memory_content.lower() or "due" in memory_content.lower():
                return "I remember you had a task with a deadline. Want me to check its status?"
        
        elif category == "fact":
            # Known facts - connect dots
            if "birthday" in memory_content.lower() or "anniversary" in memory_content.lower():
                return "I recall you mentioned an important date. Is it coming up soon?"
        
        return None
    
    async def get_daily_digest(self) -> Optional[str]:
        """
        Generate a daily memory digest for proactive cron jobs.
        
        Returns:
            Digest text or None if nothing relevant
        """
        # Get recent preferences and tasks
        preferences = await self.memory.retrieve(
            query="preferences interests goals",
            category="preference",
            limit=5
        )
        
        tasks = await self.memory.retrieve(
            query="tasks deadlines pending",
            category="task",
            limit=5
        )
        
        if not preferences and not tasks:
            return None
        
        parts = ["📊 Daily Memory Digest\n"]
        
        if preferences:
            parts.append("\n🎯 Based on your preferences:")
            for p in preferences[:3]:
                content = p.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                parts.append(f"  • {content}")
        
        if tasks:
            parts.append("\n📋 Active tasks:")
            for t in tasks[:3]:
                content = t.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                parts.append(f"  • {content}")
        
        parts.append("\n💡 Want me to help with any of these?")
        
        return "\n".join(parts)
    
    def clear_cache(self):
        """Clear suggestion cache (call periodically)."""
        self._suggestion_cache.clear()
