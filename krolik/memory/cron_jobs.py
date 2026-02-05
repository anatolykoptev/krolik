"""Memory-based cron jobs for proactive bot behavior.

These jobs are registered with nanobot's cron system.
"""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from krolik.memory.scheduler import create_memory_scheduler


async def memory_daily_digest_job(
    workspace: Path,
    channel: str = "telegram",
    to: str = "default"
) -> str:
    """
    Cron job: Send daily memory digest.
    
    Schedule: 0 9 * * * (9 AM daily)
    """
    async def send_message(ch: str, msg: str, recipient: str):
        # This will be injected by nanobot's cron runner
        # For now, return the message for logging
        logger.info(f"[DAILY_DIGEST] Would send to {ch}:{recipient}: {msg[:100]}...")
    
    try:
        scheduler = await create_memory_scheduler(
            workspace=workspace,
            send_callback=send_message,
            memu_url="http://localhost:8000"
        )
        
        sent = await scheduler.run_daily_digest(channel, to)
        
        if sent:
            return "Daily digest sent successfully"
        else:
            return "No digest to send (empty or already sent)"
            
    except Exception as e:
        logger.error(f"Daily digest job failed: {e}")
        return f"Error: {e}"


async def memory_proactive_check_job(
    workspace: Path,
    channel: str = "telegram", 
    to: str = "default"
) -> str:
    """
    Cron job: Run proactive memory check (reminders, triggers).
    
    Schedule: 0 */4 * * * (every 4 hours)
    """
    async def send_message(ch: str, msg: str, recipient: str):
        logger.info(f"[PROACTIVE] Would send to {ch}:{recipient}: {msg[:100]}...")
    
    try:
        scheduler = await create_memory_scheduler(
            workspace=workspace,
            send_callback=send_message,
            memu_url="http://localhost:8000"
        )
        
        results = await scheduler.run_proactive_check(channel, to)
        
        summary = (
            f"Proactive check complete: "
            f"{results['reminders_sent']} reminders, "
            f"{results['triggers_sent']} triggers"
        )
        
        if results['errors']:
            summary += f" | Errors: {len(results['errors'])}"
        
        return summary
        
    except Exception as e:
        logger.error(f"Proactive check job failed: {e}")
        return f"Error: {e}"


async def memory_morning_briefing_job(
    workspace: Path,
    channel: str = "telegram",
    to: str = "default"
) -> str:
    """
    Cron job: Morning briefing with memory context.
    
    Schedule: 0 8 * * * (8 AM daily)
    
    Combines daily digest with proactive suggestions.
    """
    # First, get the digest
    digest_result = await memory_daily_digest_job(workspace, channel, to)
    
    # Then check for any urgent reminders
    proactive_result = await memory_proactive_check_job(workspace, channel, to)
    
    return f"Morning briefing: {digest_result} | {proactive_result}"


# Job registry for easy access
MEMORY_CRON_JOBS = {
    "memory-daily-digest": {
        "func": memory_daily_digest_job,
        "schedule": "0 9 * * *",
        "description": "Send daily memory digest at 9 AM"
    },
    "memory-proactive-check": {
        "func": memory_proactive_check_job,
        "schedule": "0 */4 * * *",
        "description": "Check for reminders and triggers every 4 hours"
    },
    "memory-morning-briefing": {
        "func": memory_morning_briefing_job,
        "schedule": "0 8 * * *",
        "description": "Morning briefing at 8 AM with digest + proactive"
    }
}
