"""Async LLM Gateway — unified interface to multiple LLM providers.

Supports:
- Direct API calls (OpenAI-compatible, Anthropic, Gemini)
- CLIProxyAPI (local OAuth gateway for free Gemini/Claude access)
- Streaming and non-streaming modes
- Automatic retries with exponential backoff
- Fallback chains (primary → fallback1 → fallback2)
- Request/response logging for observability
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
from loguru import logger


@dataclass
class GatewayResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    latency_ms: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    delta: str
    finish_reason: str | None = None


class LLMGateway:
    """Production-grade async LLM gateway.

    Usage::

        gw = LLMGateway(providers={
            "cliproxy": ProviderEndpoint(base_url="http://127.0.0.1:8317/v1", api_key="..."),
            "openrouter": ProviderEndpoint(base_url="https://openrouter.ai/api/v1", api_key="sk-or-..."),
            "gemini": ProviderEndpoint(base_url="https://generativelanguage.googleapis.com/v1beta", api_key="AI..."),
        })
        resp = await gw.chat("cliproxy", "gemini-2.5-flash", [{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        providers: dict[str, ProviderEndpoint] | None = None,
        default_timeout: float = 60.0,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._providers: dict[str, ProviderEndpoint] = providers or {}
        self._timeout = default_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client: httpx.AsyncClient | None = None
        self._stats: dict[str, _ProviderStats] = {}

    # ── Provider management ───────────────────────────────────────

    def add_provider(self, name: str, endpoint: ProviderEndpoint) -> None:
        self._providers[name] = endpoint
        logger.debug(f"LLMGateway: added provider '{name}' → {endpoint.base_url}")

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    # ── Core chat API ─────────────────────────────────────────────

    async def chat(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GatewayResponse:
        """Send a chat completion request (non-streaming).

        Args:
            provider: Provider name (e.g. "cliproxy", "openrouter").
            model: Model identifier understood by that provider.
            messages: OpenAI-format message list.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt prepended to messages.
            timeout: Per-request timeout override.
            extra_body: Provider-specific extra fields.

        Returns:
            GatewayResponse with content, usage, latency.

        Raises:
            LLMGatewayError on unrecoverable failure.
        """
        endpoint = self._get_endpoint(provider)
        full_messages = self._prepare_messages(messages, system_prompt)

        body: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if extra_body:
            body.update(extra_body)

        t0 = time.monotonic()
        data = await self._post_with_retry(
            endpoint,
            "/chat/completions",
            body,
            timeout=timeout or self._timeout,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        return self._parse_response(data, provider, model, latency_ms)

    async def chat_stream(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request.

        Yields StreamChunk objects as they arrive.
        """
        endpoint = self._get_endpoint(provider)
        full_messages = self._prepare_messages(messages, system_prompt)

        body: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if extra_body:
            body.update(extra_body)

        client = await self._get_client()
        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        headers = self._build_headers(endpoint)

        async with client.stream(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=timeout or self._timeout * 2,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    return

                import json
                try:
                    chunk_data = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                finish = choices[0].get("finish_reason")

                if content or finish:
                    yield StreamChunk(delta=content or "", finish_reason=finish)

    async def chat_with_fallbacks(
        self,
        fallback_chain: list[tuple[str, str]],
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> GatewayResponse:
        """Try providers in order until one succeeds.

        Args:
            fallback_chain: List of (provider, model) tuples to try in order.
            messages: OpenAI-format message list.
            **kwargs: Passed to chat().

        Returns:
            GatewayResponse from first successful provider.

        Raises:
            LLMGatewayError if all providers fail.
        """
        last_error: Exception | None = None

        for provider, model in fallback_chain:
            if not self.has_provider(provider):
                logger.debug(f"Skipping unavailable provider '{provider}'")
                continue
            try:
                return await self.chat(provider, model, messages, **kwargs)
            except LLMGatewayError as e:
                last_error = e
                logger.warning(f"Fallback: {provider}/{model} failed: {e}")
                continue

        raise LLMGatewayError(
            f"All providers in fallback chain failed. Last error: {last_error}"
        )

    # ── Statistics ────────────────────────────────────────────────

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-provider stats (requests, errors, avg latency)."""
        return {
            name: {
                "requests": s.requests,
                "errors": s.errors,
                "avg_latency_ms": int(s.total_latency_ms / max(s.requests, 1)),
                "total_tokens": s.total_tokens,
            }
            for name, s in self._stats.items()
        }

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Internal helpers ──────────────────────────────────────────

    def _get_endpoint(self, provider: str) -> ProviderEndpoint:
        ep = self._providers.get(provider)
        if not ep:
            raise LLMGatewayError(
                f"Provider '{provider}' not configured. "
                f"Available: {list(self._providers.keys())}"
            )
        return ep

    @staticmethod
    def _prepare_messages(
        messages: list[dict[str, Any]], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + messages
        return messages

    @staticmethod
    def _build_headers(endpoint: ProviderEndpoint) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        if endpoint.extra_headers:
            headers.update(endpoint.extra_headers)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )
        return self._client

    async def _post_with_retry(
        self,
        endpoint: ProviderEndpoint,
        path: str,
        body: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """POST with exponential-backoff retries."""
        import json as _json

        client = await self._get_client()
        url = endpoint.base_url.rstrip("/") + path
        headers = self._build_headers(endpoint)

        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(
                    url, json=body, headers=headers, timeout=timeout
                )

                if resp.status_code == 429:
                    # Rate limited — always retry
                    retry_after = float(resp.headers.get("Retry-After", "2"))
                    logger.warning(f"Rate limited by {endpoint.base_url}, retry in {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    # Server error — retry
                    logger.warning(f"Server error {resp.status_code} from {endpoint.base_url}")
                    await asyncio.sleep(self._retry_base_delay * (2 ** attempt))
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException as e:
                last_exc = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{self._max_retries + 1}")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_base_delay * (2 ** attempt))
                continue

            except httpx.ConnectError as e:
                last_exc = e
                logger.warning(f"Connection error: {e}")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_base_delay * (2 ** attempt))
                continue

            except httpx.HTTPStatusError as e:
                # Client errors (4xx except 429) — don't retry
                raise LLMGatewayError(
                    f"HTTP {e.response.status_code}: {e.response.text[:500]}"
                ) from e

        raise LLMGatewayError(
            f"Failed after {self._max_retries + 1} attempts. Last error: {last_exc}"
        )

    def _parse_response(
        self,
        data: dict[str, Any],
        provider: str,
        model: str,
        latency_ms: int,
    ) -> GatewayResponse:
        """Parse OpenAI-compatible response JSON."""
        choices = data.get("choices", [])
        if not choices:
            raise LLMGatewayError(f"Empty response from {provider}/{model}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        finish_reason = choices[0].get("finish_reason", "stop")

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }

        # Track stats
        stats = self._stats.setdefault(provider, _ProviderStats())
        stats.requests += 1
        stats.total_latency_ms += latency_ms
        stats.total_tokens += usage.get("total_tokens", 0)

        return GatewayResponse(
            content=content,
            model=data.get("model", model),
            provider=provider,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )


# ── Supporting types ──────────────────────────────────────────────


@dataclass
class ProviderEndpoint:
    """Configuration for a single LLM provider endpoint."""

    base_url: str
    api_key: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class _ProviderStats:
    requests: int = 0
    errors: int = 0
    total_latency_ms: int = 0
    total_tokens: int = 0


class LLMGatewayError(Exception):
    """Raised on unrecoverable gateway errors."""


# ── Factory ───────────────────────────────────────────────────────


def create_gateway_from_env() -> LLMGateway:
    """Create an LLMGateway pre-configured from environment variables.

    Reads:
        CLI_PROXY_API_KEY / CLI_PROXY_API_URL → cliproxy provider
        NANOBOT_PROVIDERS__OPENROUTER__API_KEY → openrouter provider
        NANOBOT_PROVIDERS__ANTHROPIC__API_KEY → anthropic provider
        NANOBOT_PROVIDERS__GEMINI__API_KEY → gemini provider
    """
    import os

    gw = LLMGateway()

    # CLIProxyAPI (free tier via local OAuth sessions)
    cliproxy_key = os.environ.get("CLI_PROXY_API_KEY", "")
    cliproxy_url = os.environ.get("CLI_PROXY_API_URL", "http://127.0.0.1:8317")
    if cliproxy_key:
        gw.add_provider(
            "cliproxy",
            ProviderEndpoint(
                base_url=cliproxy_url.rstrip("/") + "/v1",
                api_key=cliproxy_key,
            ),
        )

    # OpenRouter
    or_key = os.environ.get("NANOBOT_PROVIDERS__OPENROUTER__API_KEY", "")
    if or_key:
        gw.add_provider(
            "openrouter",
            ProviderEndpoint(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
            ),
        )

    # Anthropic (via OpenAI-compatible endpoint)
    ant_key = os.environ.get("NANOBOT_PROVIDERS__ANTHROPIC__API_KEY", "")
    if ant_key:
        gw.add_provider(
            "anthropic",
            ProviderEndpoint(
                base_url="https://api.anthropic.com/v1",
                api_key=ant_key,
                extra_headers={"anthropic-version": "2023-06-01"},
            ),
        )

    # Gemini (via OpenAI-compatible endpoint)
    gemini_key = os.environ.get("NANOBOT_PROVIDERS__GEMINI__API_KEY", "")
    if gemini_key:
        gw.add_provider(
            "gemini",
            ProviderEndpoint(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                api_key=gemini_key,
            ),
        )

    logger.info(f"LLMGateway initialized with providers: {gw.list_providers()}")
    return gw
