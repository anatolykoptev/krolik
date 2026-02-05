"""Memory configuration for krolik."""

from pydantic import BaseModel, Field
from typing import List, Optional


class MemoryCategoriesConfig(BaseModel):
    """Configuration for memory categories."""
    
    conversation: bool = True
    task: bool = True
    fact: bool = True
    preference: bool = True
    
    # Auto-categorization settings
    auto_categorize: bool = True
    
    # Category-specific settings
    conversation_ttl_days: int = 30  # How long to keep conversations
    task_priority_keywords: List[str] = Field(default_factory=lambda: [
        "deadline", "due", "urgent", "important", "asap", "today", "tomorrow"
    ])
    preference_keywords: List[str] = Field(default_factory=lambda: [
        "like", "love", "prefer", "hate", "dislike", "favorite", "want", "need"
    ])


class IntentAwareConfig(BaseModel):
    """Configuration for intent-aware retrieval."""
    
    enabled: bool = True
    min_relevance_score: float = 0.3
    max_results: int = 5
    
    # Patterns that trigger retrieval
    retrieve_patterns: List[str] = Field(default_factory=lambda: [
        "remember", "recall", "what did", "when did",
        "you said", "we discussed", "my preference",
        "last time", "before", "previously", "ago"
    ])
    
    # Patterns that skip retrieval (casual chat)
    skip_patterns: List[str] = Field(default_factory=lambda: [
        "hello", "hi", "hey", "good morning", "good evening",
        "thanks", "thank you", "ok", "okay", "bye",
        "how are you", "what's up"
    ])


class ProactiveSuggestionsConfig(BaseModel):
    """Configuration for proactive memory suggestions."""
    
    enabled: bool = True
    min_relevance: float = 0.7
    cooldown_minutes: int = 60  # Don't suggest same thing within this time
    
    # Categories to check for suggestions
    check_categories: List[str] = Field(default_factory=lambda: [
        "preference", "task", "fact"
    ])
    
    # Daily digest settings (for cron jobs)
    daily_digest_enabled: bool = True
    digest_categories: List[str] = Field(default_factory=lambda: [
        "preference", "task"
    ])


class MemoryConfig(BaseModel):
    """Complete memory configuration."""
    
    # memU connection
    memu_url: str = "http://localhost:8000"
    memu_api_key: Optional[str] = None
    
    # Fallback settings
    file_fallback: bool = True
    
    # Auto-memorization
    auto_memorize: bool = True
    
    # Sub-configs
    categories: MemoryCategoriesConfig = Field(default_factory=MemoryCategoriesConfig)
    intent_aware: IntentAwareConfig = Field(default_factory=IntentAwareConfig)
    proactive: ProactiveSuggestionsConfig = Field(default_factory=ProactiveSuggestionsConfig)
