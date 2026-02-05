"""Proactive scheduler for memory-based actions.

Integrates with nanobot's cron system to provide:
- Daily memory digest
- Reminder checking
- Proactive suggestions based on stored memories
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable

from loguru import logger

from krolik.memory.proactive import ProactiveMemorySuggestions
from krolik.memory.store import EnhancedMemoryStore
from krolik.memory.config import MemoryConfig


class ProactiveMemoryScheduler:
    """
    Scheduler for proactive memory-based actions.
    
    Runs as part of nanobot's cron system.
    """
    
    def __init__(
        self,
        memory_store: EnhancedMemoryStore,
        config: MemoryConfig,
        send_callback: Optional[Callable[[str, str], Any]] = None
    ):
        self.memory = memory_store
        self.config = config
        self.proactive = ProactiveMemorySuggestions(memory_store)
        self.send_callback = send_callback  # (channel, message) -> None
        
        # Track last runs
        self._last_daily_digest: Optional[datetime] = None
        self._last_reminder_check: Optional[datetime] = None
    
    async def run_daily_digest(self, channel: str = "telegram", to: str = "default") -> bool:
        """
        Generate and send daily memory digest.
        
        Called by cron job. Returns True if digest was sent.
        """
        if not self.config.proactive.daily_digest_enabled:
            return False
        
        # Check if already sent today
        now = datetime.now()
        if self._last_daily_digest and self._last_daily_digest.date() == now.date():
            logger.debug("Daily digest already sent today")
            return False
        
        try:
            digest = await self.proactive.get_daily_digest()
            
            if digest and self.send_callback:
                await self.send_callback(channel, digest, to)
                self._last_daily_digest = now
                logger.info(f"Daily digest sent to {channel}:{to}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Daily digest failed: {e}")
            return False
    
    async def check_reminders(self) -> list[dict[str, Any]]:
        """
        Check for task reminders (deadlines, etc).
        
        Returns list of reminders to send.
        """
        reminders = []
        
        try:
            # Search for tasks with deadline keywords
            deadline_keywords = self.config.categories.task_priority_keywords
            
            for keyword in deadline_keywords[:3]:  # Check top keywords
                results = await self.memory.retrieve(
                    query=f"task {keyword}",
                    category="task",
                    limit=5
                )
                
                for result in results:
                    content = result.get("content", "")
                    score = result.get("score", 0)
                    
                    # High relevance = likely important
                    if score > 0.6:
                        reminders.append({
                            "type": "task",
                            "content": content,
                            "priority": "high" if keyword in ["urgent", "asap"] else "normal",
                            "suggested_action": f"Check task status: {content[:100]}..."
                        })
            
            return reminders
            
        except Exception as e:
            logger.error(f"Reminder check failed: {e}")
            return []
    
    async def check_memory_triggers(self) -> list[dict[str, Any]]:
        """
        Check for memory-based triggers.
        
        Examples:
        - User mentioned wanting to learn X → suggest resources
        - User has birthday stored → remind when approaching
        - User has recurring task → remind when due
        """
        triggers = []
        
        try:
            # Check preferences for opportunities
            prefs = await self.memory.retrieve(
                query="interests goals learning",
                category="preference",
                limit=5
            )
            
            for pref in prefs:
                content = pref.get("content", "").lower()
                
                # Detect learning goals
                if "learn" in content or "want to" in content:
                    triggers.append({
                        "type": "learning_opportunity",
                        "memory": pref,
                        "suggestion": f"I see you wanted to {content}. Need help with that?"
                    })
                
                # Detect preferences that could be relevant
                if "like" in content or "love" in content:
                    triggers.append({
                        "type": "preference",
                        "memory": pref,
                        "suggestion": f"Remembered: {content}. Want to explore this more?"
                    })
            
            return triggers
            
        except Exception as e:
            logger.error(f"Memory trigger check failed: {e}")
            return []
    
    async def run_proactive_check(
        self,
        channel: str = "telegram",
        to: str = "default"
    ) -> dict[str, Any]:
        """
        Run full proactive check (reminders + triggers).
        
        Called periodically by cron.
        """
        results = {
            "reminders_sent": 0,
            "triggers_sent": 0,
            "errors": []
        }
        
        if not self.send_callback:
            results["errors"].append("No send callback configured")
            return results
        
        # Check reminders
        reminders = await self.check_reminders()
        for reminder in reminders:
            try:
                message = f"⏰ {reminder['suggested_action']}"
                await self.send_callback(channel, message, to)
                results["reminders_sent"] += 1
            except Exception as e:
                results["errors"].append(f"Reminder send failed: {e}")
        
        # Check triggers (but don't spam - limit to 1 per check)
        triggers = await self.check_memory_triggers()
        if triggers and results["reminders_sent"] == 0:  # Only if no reminders sent
            try:
                trigger = triggers[0]  # Just top one
                message = f"💡 {trigger['suggestion']}"
                await self.send_callback(channel, message, to)
                results["triggers_sent"] += 1
            except Exception as e:
                results["errors"].append(f"Trigger send failed: {e}")
        
        return results


# Factory for creating scheduler with nanobot's send mechanism
async def create_memory_scheduler(
    workspace: Path,
    send_callback: Callable[[str, str, str], Any],
    memu_url: str = "http://localhost:8000"
) -> ProactiveMemoryScheduler:
    """Create scheduler integrated with nanobot."""
    
    from krolik.memory.config import MemoryConfig
    
    config = MemoryConfig(memu_url=memu_url)
    memory = EnhancedMemoryStore(workspace, memu_url=memu_url)
    
    return ProactiveMemoryScheduler(
        memory_store=memory,
        config=config,
        send_callback=send_callback
    )
