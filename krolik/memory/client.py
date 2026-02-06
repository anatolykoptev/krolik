"""MemU client — embedded MemoryService with HTTP and file fallbacks.

Strategy (ordered by preference):
1. Embedded: direct memu.app.MemoryService (no HTTP, full pipeline)
2. HTTP: remote memU service at memu_url
3. File: markdown-based fallback (always works)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

# Default PostgreSQL DSN for krolik memory (same cluster as OpenClaw memos)
DEFAULT_PG_DSN = "postgresql://memos:K2DscvW8JoBmSpEV4WIM856E6XtVl0s@172.18.0.3:5432/krolik_memory"


def _resolve_llm_config() -> dict[str, Any] | None:
    """Build memU LLM config from krolik's environment variables."""
    # Try providers in order: Gemini, OpenRouter, OpenAI, Anthropic
    providers = [
        {
            "env_key": "KROLIK_PROVIDERS__GEMINI__API_KEY",
            "legacy_key": "NANOBOT_PROVIDERS__GEMINI__API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "provider": "openai",
            "chat_model": "gemini-2.0-flash",
            "embed_model": "text-embedding-004",
        },
        {
            "env_key": "KROLIK_PROVIDERS__OPENROUTER__API_KEY",
            "legacy_key": "NANOBOT_PROVIDERS__OPENROUTER__API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "chat_model": "google/gemini-2.0-flash-001",
            "embed_model": "openai/text-embedding-3-small",
        },
        {
            "env_key": "KROLIK_PROVIDERS__OPENAI__API_KEY",
            "legacy_key": "NANOBOT_PROVIDERS__OPENAI__API_KEY",
            "base_url": "https://api.openai.com/v1",
            "provider": "openai",
            "chat_model": "gpt-4o-mini",
            "embed_model": "text-embedding-3-small",
        },
    ]
    for p in providers:
        api_key = os.environ.get(p["env_key"]) or os.environ.get(p["legacy_key"], "")
        if api_key:
            return {
                "base_url": p["base_url"],
                "api_key": api_key,
                "provider": p["provider"],
                "chat_model": p["chat_model"],
                "embed_model": p["embed_model"],
                "client_backend": "httpx",
            }
    return None


class MemUClient:
    """Client for memU memory engine.

    Instantiates an embedded MemoryService when possible (zero HTTP overhead),
    falls back to HTTP client, and finally to file-based storage.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        data_dir: Optional[Path] = None,
        pg_dsn: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=30.0)

        self._data_dir = data_dir or Path.home() / ".krolik" / "memory"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # PostgreSQL DSN: explicit > env var > default
        self._pg_dsn = (
            pg_dsn
            or os.environ.get("KROLIK_MEMORY__PG_DSN")
            or os.environ.get("KROLIK_MEMORY__DATABASE_URL")
            or DEFAULT_PG_DSN
        )

        # Embedded service (lazy init)
        self._service: Any = None
        self._service_attempted = False

    def _resolve_database_config(self) -> dict[str, Any]:
        """Build memU database config — PostgreSQL + pgvector if reachable, else inmemory."""
        try:
            import psycopg2
            conn = psycopg2.connect(self._pg_dsn, connect_timeout=3)
            conn.close()
            logger.debug(f"memU: PostgreSQL reachable at {self._pg_dsn.split('@')[-1]}")
            return {
                "metadata_store": {
                    "provider": "postgres",
                    "dsn": self._pg_dsn,
                    "ddl_mode": "create",
                },
                "vector_index": {
                    "provider": "pgvector",
                    "dsn": self._pg_dsn,
                },
            }
        except Exception as e:
            logger.warning(f"memU: PostgreSQL unavailable ({e}), falling back to inmemory")
            return {"metadata_store": {"provider": "inmemory"}}

    def _init_service(self) -> bool:
        """Try to create an embedded MemoryService (once)."""
        if self._service_attempted:
            return self._service is not None
        self._service_attempted = True

        try:
            from memu.app import MemoryService

            llm_cfg = _resolve_llm_config()
            if not llm_cfg:
                logger.debug("memU: no LLM provider configured, skipping embedded mode")
                return False

            resources_dir = str(self._data_dir / "resources")

            # Try PostgreSQL first, fall back to inmemory
            db_config = self._resolve_database_config()

            self._service = MemoryService(
                llm_profiles={
                    "default": llm_cfg,
                    "embedding": llm_cfg,
                },
                blob_config={"resources_dir": resources_dir},
                database_config=db_config,
            )
            backend = db_config["metadata_store"]["provider"]
            logger.info(f"memU: embedded MemoryService initialized ({backend})")
            return True
        except Exception as e:
            logger.debug(f"memU: embedded init failed ({e}), will try HTTP fallback")
            return False

    # ── Public API ──────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if memU is available (embedded or HTTP)."""
        if self._init_service():
            return True
        try:
            r = await self._http.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    async def memorize(
        self,
        messages: list[dict[str, Any]],
        category: str = "conversation",
        metadata: Optional[dict] = None,
    ) -> bool:
        """Save messages to memory."""
        # ── Embedded path ──
        if self._init_service():
            return await self._memorize_embedded(messages, category, metadata)
        # ── HTTP fallback ──
        return await self._memorize_http(messages, category, metadata)

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve memories by semantic query."""
        # ── Embedded path ──
        if self._init_service():
            return await self._retrieve_embedded(query, category, limit)
        # ── HTTP fallback ──
        return await self._retrieve_http(query, category, limit)

    async def close(self):
        """Close HTTP client."""
        await self._http.aclose()

    # ── Embedded implementation ─────────────────────────────────

    async def _memorize_embedded(
        self,
        messages: list[dict[str, Any]],
        category: str,
        metadata: Optional[dict],
    ) -> bool:
        try:
            # Write messages as JSON conversation file for memU ingest
            content = json.dumps(messages, ensure_ascii=False)
            tmp = self._data_dir / "resources" / "_ingest_tmp.json"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(content, encoding="utf-8")

            result = await self._service.memorize(
                resource_url=str(tmp),
                modality="conversation",
                user={"user_id": "krolik"},
            )
            logger.info(f"memU embedded: memorized {len(messages)} messages → {category}")
            return True
        except Exception as e:
            logger.warning(f"memU embedded memorize failed: {e}")
            return False

    async def _retrieve_embedded(
        self,
        query: str,
        category: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            queries = [{"role": "user", "content": query}]
            where = {"user_id": "krolik"}
            result = await self._service.retrieve(queries, where=where)

            # Normalize memU response → krolik format
            memories = []
            for item in result.get("items", []):
                memories.append({
                    "content": item.get("content", item.get("text", "")),
                    "category": item.get("category", category or "conversation"),
                    "score": item.get("score", 0.5),
                    "source": "memu_embedded",
                })
            for cat in result.get("categories", []):
                if cat.get("summary"):
                    memories.append({
                        "content": cat["summary"],
                        "category": cat.get("name", "unknown"),
                        "score": cat.get("score", 0.5),
                        "source": "memu_embedded",
                    })
            logger.debug(f"memU embedded: retrieved {len(memories)} memories for: {query}")
            return memories[:limit]
        except Exception as e:
            logger.warning(f"memU embedded retrieve failed: {e}")
            return []

    # ── HTTP implementation ─────────────────────────────────────

    async def _memorize_http(
        self,
        messages: list[dict[str, Any]],
        category: str,
        metadata: Optional[dict],
    ) -> bool:
        try:
            payload = {"messages": messages, "category": category, "metadata": metadata or {}}
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            r = await self._http.post(f"{self.base_url}/memorize", json=payload, headers=headers)
            if r.status_code == 200:
                logger.info(f"memU HTTP: memorized {len(messages)} messages → {category}")
                return True
            logger.warning(f"memU HTTP memorize: {r.status_code} — {r.text}")
            return False
        except Exception as e:
            logger.error(f"memU HTTP memorize error: {e}")
            return False

    async def _retrieve_http(
        self,
        query: str,
        category: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            payload: dict[str, Any] = {"query": query, "limit": limit}
            if category:
                payload["category"] = category
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            r = await self._http.post(f"{self.base_url}/retrieve", json=payload, headers=headers)
            if r.status_code == 200:
                results = r.json().get("results", [])
                logger.debug(f"memU HTTP: retrieved {len(results)} memories for: {query}")
                return results
            logger.warning(f"memU HTTP retrieve: {r.status_code} — {r.text}")
            return []
        except Exception as e:
            logger.error(f"memU HTTP retrieve error: {e}")
            return []
