"""MemU client — tries direct Python import first, HTTP fallback second."""

import httpx
from typing import Any, Optional
from loguru import logger


def _try_import_memu():
    """Try to import memu-py package directly (optional dependency)."""
    try:
        from memu.app.service import MemUService
        return MemUService
    except ImportError:
        return None


class MemUClient:
    """Client for memU memory service.
    
    Strategy:
    1. If memu-py is installed, use direct Python API (no HTTP overhead)
    2. Otherwise, fall back to HTTP client against memU service
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        
        # Try direct import
        self._service_cls = _try_import_memu()
        self._service = None
        if self._service_cls:
            logger.info("memU: using direct Python API (memu-py installed)")
        else:
            logger.debug("memU: using HTTP client (memu-py not installed)")
        
    async def health_check(self) -> bool:
        """Check if memU service is available."""
        # Direct mode: always available if importable
        if self._service_cls:
            return True
        
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"MemU health check failed: {e}")
            return False
    
    async def memorize(
        self, 
        messages: list[dict[str, Any]], 
        category: str = "conversation",
        metadata: Optional[dict] = None
    ) -> bool:
        """Save messages to memory.
        
        Args:
            messages: List of message dicts with role and content
            category: Memory category (conversation, task, fact, preference)
            metadata: Optional metadata to attach
            
        Returns:
            True if successful
        """
        try:
            payload = {
                "messages": messages,
                "category": category,
                "metadata": metadata or {}
            }
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            response = await self._client.post(
                f"{self.base_url}/memorize",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"Memorized {len(messages)} messages to category '{category}'")
                return True
            else:
                logger.warning(f"Memorize failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Memorize error: {e}")
            return False
    
    async def retrieve(
        self, 
        query: str, 
        category: Optional[str] = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve memories by query.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Max results to return
            
        Returns:
            List of memory results
        """
        try:
            payload = {
                "query": query,
                "limit": limit
            }
            if category:
                payload["category"] = category
                
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            response = await self._client.post(
                f"{self.base_url}/retrieve",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                logger.debug(f"Retrieved {len(results)} memories for query: {query}")
                return results
            else:
                logger.warning(f"Retrieve failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            return []
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()
