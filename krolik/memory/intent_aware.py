"""Intent-aware retrieval with pre-retrieval decision."""

from typing import Any, Optional
from loguru import logger


class IntentAwareRetriever:
    """
    Intent-aware retrieval system.
    
    Based on memU's pre-retrieval decision pattern:
    1. Analyze if retrieval is needed
    2. Rewrite query if needed
    3. Retrieve only when beneficial
    """
    
    # Simple keyword-based intent detection (can be replaced with LLM)
    NO_RETRIEVE_PATTERNS = [
        "hello", "hi", "hey", "good morning", "good evening",
        "thanks", "thank you", "ok", "okay", "bye",
        "how are you", "what's up", "help me"
    ]
    
    RETRIEVE_PATTERNS = [
        "remember", "recall", "what did", "when did",
        "you said", "we discussed", "my preference",
        "last time", "before", "previously"
    ]
    
    def __init__(self, memory_store):
        self.memory = memory_store
    
    def should_retrieve(self, query: str) -> tuple[bool, str]:
        """
        Decide if retrieval is needed and rewrite query.
        
        Returns:
            (should_retrieve, rewritten_query)
        """
        query_lower = query.lower()
        
        # Check if query explicitly references memory
        for pattern in self.RETRIEVE_PATTERNS:
            if pattern in query_lower:
                return True, query
        
        # Check if query is casual (no retrieval needed)
        for pattern in self.NO_RETRIEVE_PATTERNS:
            if pattern in query_lower:
                return False, query
        
        # Default: try retrieval for unknown queries
        return True, query
    
    async def retrieve_if_needed(
        self, 
        query: str, 
        category: Optional[str] = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Smart retrieval - only fetches when beneficial.
        
        Args:
            query: User query
            category: Optional category filter
            limit: Max results
            
        Returns:
            List of memories or empty list if not needed
        """
        should_retrieve, rewritten_query = self.should_retrieve(query)
        
        if not should_retrieve:
            logger.debug(f"Skipping retrieval for casual query: {query}")
            return []
        
        # Perform retrieval
        results = await self.memory.retrieve(rewritten_query, category, limit)
        
        # Sufficiency check - filter low-relevance results
        if results:
            # Only return results with reasonable relevance
            results = [r for r in results if r.get("score", 0) > 0.3]
        
        return results
    
    def format_for_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieval results for agent context."""
        if not results:
            return ""
        
        parts = ["## Relevant Memories\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            category = result.get("category", "unknown")
            
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            
            parts.append(f"{i}. [{category}] {content}")
        
        return "\n".join(parts)
