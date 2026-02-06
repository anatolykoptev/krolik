"""Enhanced memory store with memU integration and file fallback."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from krolik.memory.client import MemUClient
from krolik.utils.helpers import ensure_dir, today_date


class EnhancedMemoryStore:
    """
    Enhanced memory system with memU integration.
    
    Uses memU for vector-based semantic memory when available,
    falls back to file-based storage (original file-based behavior).
    """
    
    def __init__(
        self, 
        workspace: Path,
        memu_url: str = "http://localhost:8000",
        memu_api_key: Optional[str] = None,
        pg_dsn: Optional[str] = None,
    ):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        
        # memU client — PostgreSQL+pgvector → HTTP → file fallback
        self._memu = MemUClient(
            base_url=memu_url,
            api_key=memu_api_key,
            data_dir=self.memory_dir / "memu_data",
            pg_dsn=pg_dsn,
        )
        self._memu_available: Optional[bool] = None
        
    async def _check_memu(self) -> bool:
        """Check if memU is available (cached)."""
        if self._memu_available is None:
            self._memu_available = await self._memu.health_check()
            if self._memu_available:
                logger.info("memU memory service connected")
            else:
                logger.info("memU unavailable, using file-based memory")
        return self._memu_available
    
    # ========== File-based methods (original interface) ==========
    
    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"
    
    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""
    
    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        today_file = self.get_today_file()
        
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            # Add header for new day
            header = f"# {today_date()}\n\n"
            content = header + content
        
        today_file.write_text(content, encoding="utf-8")
    
    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""
    
    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        self.memory_file.write_text(content, encoding="utf-8")
    
    def get_recent_memories(self, days: int = 7) -> str:
        """Get memories from the last N days."""
        memories = []
        today = datetime.now().date()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)
        
        return "\n\n---\n\n".join(memories)
    
    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []
        
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)
    
    def get_memory_context(self) -> str:
        """Get memory context for the agent."""
        parts = []
        
        # Long-term memory
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        
        # Today's notes
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)
        
        return "\n\n".join(parts) if parts else ""
    
    # ========== memU-enhanced methods ==========
    
    async def memorize(
        self, 
        messages: list[dict[str, Any]], 
        category: str = "conversation",
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Memorize messages using memU or file fallback.
        
        Args:
            messages: List of messages with 'role' and 'content'
            category: Memory category (conversation, task, fact, preference)
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        # Try memU first
        if await self._check_memu():
            return await self._memu.memorize(messages, category, metadata)
        
        # Fallback: append to today's file
        try:
            content_parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_parts.append(f"**{role}**: {content}")
            
            content = "\n\n".join(content_parts)
            self.append_today(f"\n\n[{category}]\n{content}")
            return True
        except Exception as e:
            logger.error(f"File fallback memorize failed: {e}")
            return False
    
    async def retrieve(
        self, 
        query: str, 
        category: Optional[str] = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories by query using memU or file fallback.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Max results
            
        Returns:
            List of memory results
        """
        # Try memU first
        if await self._check_memu():
            return await self._memu.retrieve(query, category, limit)
        
        # Fallback: simple text search in recent files
        results = []
        recent = self.get_recent_memories(days=30)
        
        if query.lower() in recent.lower():
            # Return simple match structure
            results.append({
                "content": recent[:2000],  # Truncate for display
                "category": "conversation",
                "score": 0.5,
                "source": "file_fallback"
            })
        
        return results[:limit]
    
    async def close(self):
        """Close memU client."""
        await self._memu.close()
