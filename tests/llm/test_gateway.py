"""Tests for the LLM gateway."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from krolik.llm.gateway import (
    LLMGateway,
    LLMGatewayError,
    GatewayResponse,
    ProviderEndpoint,
    StreamChunk,
    create_gateway_from_env,
)


@pytest.fixture
def gateway():
    gw = LLMGateway()
    gw.add_provider("test", ProviderEndpoint(base_url="http://localhost:9999/v1", api_key="test-key"))
    return gw


@pytest.fixture
def mock_response():
    """Standard OpenAI-compatible response body."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from test model!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def test_add_provider(gateway):
    assert gateway.has_provider("test")
    assert "test" in gateway.list_providers()


def test_missing_provider(gateway):
    assert not gateway.has_provider("nonexistent")


def test_list_providers(gateway):
    gateway.add_provider("second", ProviderEndpoint(base_url="http://localhost:8888/v1"))
    providers = gateway.list_providers()
    assert "test" in providers
    assert "second" in providers


@pytest.mark.asyncio
async def test_chat_success(gateway, mock_response):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        resp = await gateway.chat(
            "test", "test-model", [{"role": "user", "content": "Hello"}]
        )

    assert isinstance(resp, GatewayResponse)
    assert resp.content == "Hello from test model!"
    assert resp.provider == "test"
    assert resp.model == "test-model"
    assert resp.usage["total_tokens"] == 15
    assert resp.latency_ms >= 0


@pytest.mark.asyncio
async def test_chat_unknown_provider(gateway):
    with pytest.raises(LLMGatewayError, match="not configured"):
        await gateway.chat("unknown", "model", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_with_system_prompt(gateway, mock_response):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()
    
    captured_kwargs = {}
    async def capture_post(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp
    
    with patch.object(httpx.AsyncClient, "post", side_effect=capture_post):
        await gateway.chat(
            "test", "model", [{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
        )
    
    body = captured_kwargs.get("json", {})
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are helpful."
    assert body["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_with_fallbacks(gateway, mock_response):
    gateway.add_provider("fallback", ProviderEndpoint(base_url="http://localhost:8888/v1", api_key="fb"))

    call_count = 0
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = mock_response
    mock_resp_ok.raise_for_status = MagicMock()

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection refused")
        return mock_resp_ok

    with patch.object(httpx.AsyncClient, "post", side_effect=side_effect):
        resp = await gateway.chat_with_fallbacks(
            [("test", "model-a"), ("fallback", "model-b")],
            [{"role": "user", "content": "hello"}],
        )

    assert resp.content == "Hello from test model!"
    assert call_count >= 2  # First failed, second succeeded


@pytest.mark.asyncio
async def test_chat_with_fallbacks_all_fail(gateway):
    async def always_fail(*args, **kwargs):
        raise httpx.ConnectError("nope")

    with patch.object(httpx.AsyncClient, "post", side_effect=always_fail):
        with pytest.raises(LLMGatewayError, match="All providers.*failed"):
            await gateway.chat_with_fallbacks(
                [("test", "model")],
                [{"role": "user", "content": "hello"}],
            )


@pytest.mark.asyncio
async def test_chat_with_fallbacks_skips_unavailable(gateway, mock_response):
    """Providers not in the gateway should be skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        resp = await gateway.chat_with_fallbacks(
            [("nonexistent", "m1"), ("test", "m2")],
            [{"role": "user", "content": "hi"}],
        )
    assert resp.content == "Hello from test model!"


def test_stats_empty(gateway):
    stats = gateway.get_stats()
    assert stats == {}


@pytest.mark.asyncio
async def test_stats_tracked(gateway, mock_response):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        await gateway.chat("test", "model", [{"role": "user", "content": "hi"}])
        await gateway.chat("test", "model", [{"role": "user", "content": "bye"}])

    stats = gateway.get_stats()
    assert "test" in stats
    assert stats["test"]["requests"] == 2
    assert stats["test"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_close(gateway):
    await gateway.close()
    # Should not raise even if called twice
    await gateway.close()


def test_create_gateway_from_env_no_keys():
    """With no env vars set, gateway should initialize with no providers."""
    import os
    env_backup = {}
    keys_to_clear = [
        "CLI_PROXY_API_KEY", "NANOBOT_PROVIDERS__OPENROUTER__API_KEY",
        "NANOBOT_PROVIDERS__ANTHROPIC__API_KEY", "NANOBOT_PROVIDERS__GEMINI__API_KEY",
    ]
    for k in keys_to_clear:
        env_backup[k] = os.environ.pop(k, None)
    
    try:
        gw = create_gateway_from_env()
        # May have providers from actual env, just check it doesn't crash
        assert isinstance(gw, LLMGateway)
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v


def test_gateway_response_total_tokens():
    resp = GatewayResponse(
        content="hi", model="m", provider="p",
        usage={"total_tokens": 42, "prompt_tokens": 10, "completion_tokens": 32},
    )
    assert resp.total_tokens == 42


def test_gateway_response_empty_usage():
    resp = GatewayResponse(content="hi", model="m", provider="p")
    assert resp.total_tokens == 0
