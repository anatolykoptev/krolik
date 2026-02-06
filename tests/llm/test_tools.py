"""Tests for LLM tools (llm_call, coding_agent, list_models, discover_models)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from krolik.llm.gateway import LLMGateway, GatewayResponse, ProviderEndpoint, LLMGatewayError
from krolik.llm.router import ModelRouter
from krolik.llm.models import MODELS, Capability, Tier
from krolik.llm.tool import LLMCallTool, CodingAgentTool, ListModelsTool, DiscoverModelsTool
from krolik.agent.tools.base import ToolResult


@pytest.fixture
def gateway():
    gw = LLMGateway()
    gw.add_provider("cliproxy", ProviderEndpoint(base_url="http://localhost:8317/v1", api_key="test"))
    gw.add_provider("gemini", ProviderEndpoint(base_url="http://localhost:9999/v1", api_key="test"))
    return gw


@pytest.fixture
def router(gateway, tmp_path):
    return ModelRouter(
        available_providers=set(gateway.list_providers()),
        outcomes_path=tmp_path / "outcomes.json",
    )


@pytest.fixture
def mock_response():
    return GatewayResponse(
        content="Test response content",
        model="gemini-2.0-flash",
        provider="gemini",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        latency_ms=150,
    )


# ── LLMCallTool ──────────────────────────────────────────────────


class TestLLMCallTool:
    def test_tool_properties(self, gateway, router):
        tool = LLMCallTool(gateway, router)
        assert tool.name == "llm_call"
        assert "prompt" in tool.parameters["properties"]
        assert "prompt" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_auto_route_success(self, gateway, router, mock_response):
        tool = LLMCallTool(gateway, router)

        with patch.object(gateway, "chat_with_fallbacks", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(prompt="Translate hello to Russian")

        assert isinstance(result, ToolResult)
        assert result.success
        assert "Test response content" in result.output
        assert "gemini" in result.output

    @pytest.mark.asyncio
    async def test_explicit_model_by_alias(self, gateway, router, mock_response):
        tool = LLMCallTool(gateway, router)

        with patch.object(gateway, "chat", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(prompt="Hello", model="cliproxy-flash")

        assert result.success

    @pytest.mark.asyncio
    async def test_explicit_model_provider_slash_format(self, gateway, router, mock_response):
        tool = LLMCallTool(gateway, router)

        with patch.object(gateway, "chat", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(prompt="Hello", model="gemini/test-model")

        assert result.success

    @pytest.mark.asyncio
    async def test_unknown_model_error(self, gateway, router):
        tool = LLMCallTool(gateway, router)
        result = await tool.execute(prompt="Hello", model="nonexistent")
        assert not result.success
        assert "Unknown model" in result.error

    @pytest.mark.asyncio
    async def test_unavailable_provider_error(self, gateway, router, mock_response):
        tool = LLMCallTool(gateway, router)
        # Use a known model but whose provider isn't in gateway
        result = await tool.execute(prompt="Hello", model="openrouter/test")
        assert not result.success
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_gateway_error_handling(self, gateway, router):
        tool = LLMCallTool(gateway, router)

        with patch.object(
            gateway, "chat_with_fallbacks",
            new_callable=AsyncMock,
            side_effect=LLMGatewayError("Connection refused"),
        ):
            result = await tool.execute(prompt="Hello")

        assert not result.success
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_capability_routing(self, gateway, router, mock_response):
        tool = LLMCallTool(gateway, router)

        with patch.object(gateway, "chat_with_fallbacks", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(prompt="Write a function", capability="code")

        assert result.success


# ── CodingAgentTool ──────────────────────────────────────────────


class TestCodingAgentTool:
    def test_tool_properties(self, gateway, router):
        tool = CodingAgentTool(gateway, router)
        assert tool.name == "coding_agent"
        assert "task" in tool.parameters["properties"]
        assert "task" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_coding_task_success(self, gateway, router, mock_response):
        tool = CodingAgentTool(gateway, router)

        with patch.object(gateway, "chat", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(task="Write a Python function to validate emails")

        assert result.success
        assert "Test response content" in result.output
        assert "coding_agent" in result.output

    @pytest.mark.asyncio
    async def test_coding_with_language_and_context(self, gateway, router, mock_response):
        tool = CodingAgentTool(gateway, router)

        captured_messages = None

        async def capture_chat(provider, model, messages, **kwargs):
            nonlocal captured_messages
            captured_messages = messages
            return mock_response

        with patch.object(gateway, "chat", side_effect=capture_chat):
            result = await tool.execute(
                task="Implement a REST endpoint",
                language="typescript",
                context="Express.js, PostgreSQL",
            )

        assert result.success
        prompt_text = captured_messages[0]["content"]
        assert "typescript" in prompt_text.lower()
        assert "Express.js" in prompt_text

    @pytest.mark.asyncio
    async def test_coding_low_temperature(self, gateway, router, mock_response):
        """Coding agent should use low temperature."""
        tool = CodingAgentTool(gateway, router)

        captured_kwargs = {}

        async def capture_chat(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        with patch.object(gateway, "chat", side_effect=capture_chat):
            await tool.execute(task="Write a function")

        assert captured_kwargs.get("temperature", 1.0) <= 0.5

    @pytest.mark.asyncio
    async def test_coding_explicit_model(self, gateway, router, mock_response):
        tool = CodingAgentTool(gateway, router)

        with patch.object(gateway, "chat", new_callable=AsyncMock, return_value=mock_response):
            result = await tool.execute(task="Write code", model="cliproxy-flash")

        assert result.success

    @pytest.mark.asyncio
    async def test_coding_gateway_error(self, gateway, router):
        tool = CodingAgentTool(gateway, router)

        with patch.object(
            gateway, "chat",
            new_callable=AsyncMock,
            side_effect=LLMGatewayError("Timeout"),
        ):
            result = await tool.execute(task="Write code")

        assert not result.success
        assert "Timeout" in result.error


# ── ListModelsToolTool ───────────────────────────────────────────


class TestListModelsTool:
    def test_tool_properties(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        assert tool.name == "list_models"

    @pytest.mark.asyncio
    async def test_list_all_models(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        result = await tool.execute()
        assert result.success
        assert "FREE" in result.output

    @pytest.mark.asyncio
    async def test_list_by_tier(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        result = await tool.execute(tier="free")
        assert result.success
        assert "FREE" in result.output

    @pytest.mark.asyncio
    async def test_shows_availability(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        result = await tool.execute()
        assert result.success
        assert "✅" in result.output

    @pytest.mark.asyncio
    async def test_shows_total_count(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        result = await tool.execute()
        assert result.success
        assert "total" in result.output

    @pytest.mark.asyncio
    async def test_stale_cache_warning(self, gateway, router):
        tool = ListModelsTool(gateway, router)
        result = await tool.execute()
        assert result.success
        # Default registry has needs_discovery=True
        assert "discover_models" in result.output


class TestDiscoverModelsTool:
    def test_tool_properties(self):
        tool = DiscoverModelsTool()
        assert tool.name == "discover_models"

    @pytest.mark.asyncio
    async def test_no_api_key_error(self):
        import os
        env_backup = os.environ.pop("NANOBOT_PROVIDERS__OPENROUTER__API_KEY", None)
        try:
            tool = DiscoverModelsTool()
            result = await tool.execute()
            assert not result.success
            assert "API_KEY" in result.error
        finally:
            if env_backup:
                os.environ["NANOBOT_PROVIDERS__OPENROUTER__API_KEY"] = env_backup
