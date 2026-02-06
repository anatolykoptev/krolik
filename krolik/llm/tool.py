"""LLM tools for the agent — call other models, delegate coding tasks.

Provides four tools:
- llm_call: Direct LLM call with model routing or explicit model selection
- coding_agent: Spawn a focused coding task on a cheaper/faster model
- list_models: List available models with costs and capabilities
- discover_models: Fetch fresh models from OpenRouter API
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from krolik.agent.tools.base import Tool, ToolResult
from krolik.llm.gateway import LLMGateway, LLMGatewayError
from krolik.llm.models import MODELS, Capability, Tier
from krolik.llm.router import ModelRouter, RouteResult


class LLMCallTool(Tool):
    """Call any LLM model through the gateway with automatic routing."""

    name = "llm_call"
    description = """Call an LLM model for a subtask without using the main model's context.

Use this when:
- You need a second opinion or specialized model for a subtask
- The task is simple enough for a cheaper/faster model
- You want to offload work (translation, summarization, code generation)
- You need a research model for web search (Perplexity)

The system automatically routes to the best model by cost/capability,
or you can specify a model explicitly.

Examples:
  llm_call(prompt="Translate to Russian: Hello world")  → routes to free tier
  llm_call(prompt="Implement OAuth2 flow", capability="code")  → routes to standard tier
  llm_call(prompt="Latest AI news", capability="search")  → routes to research tier
  llm_call(prompt="Fix this code", model="cliproxy/gemini-2.5-flash")  → explicit model
"""

    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt/task to send to the LLM",
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional system prompt for context",
                "default": "",
            },
            "model": {
                "type": "string",
                "description": "Explicit model ID or alias (e.g. 'flash', 'sonnet', 'cliproxy/gemini-2.5-pro'). Leave empty for auto-routing.",
                "default": "",
            },
            "capability": {
                "type": "string",
                "enum": ["chat", "code", "vision", "reasoning", "search", ""],
                "description": "Required capability for auto-routing",
                "default": "",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum output tokens",
                "default": 4096,
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature (0.0-1.0)",
                "default": 0.7,
            },
        },
        "required": ["prompt"],
    }

    def __init__(self, gateway: LLMGateway, router: ModelRouter) -> None:
        self._gw = gateway
        self._router = router

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "",
        capability: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ToolResult:
        try:
            messages = [{"role": "user", "content": prompt}]

            if model:
                # Explicit model — resolve to provider
                spec = MODELS.get(model)
                if spec:
                    provider = spec.provider
                    model_id = spec.id
                    # Strip provider prefix for the API call if needed
                    if "/" in model_id and model_id.startswith(provider + "/"):
                        model_id = model_id[len(provider) + 1:]
                elif "/" in model:
                    # Format: provider/model
                    provider, model_id = model.split("/", 1)
                else:
                    return ToolResult(
                        success=False,
                        error=f"Unknown model '{model}'. Use full id (provider/model) or an alias.",
                    )

                if not self._gw.has_provider(provider):
                    return ToolResult(
                        success=False,
                        error=f"Provider '{provider}' not available. Available: {self._gw.list_providers()}",
                    )

                resp = await self._gw.chat(
                    provider,
                    model_id,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt or None,
                )
            else:
                # Auto-route
                cap = Capability(capability) if capability else None
                route = self._router.route(prompt, required_capability=cap)

                # Build fallback chain from route result
                chain = [(route.model.provider, route.model_id)]
                for fb in route.fallbacks:
                    chain.append((fb.provider, fb.id))

                resp = await self._gw.chat_with_fallbacks(
                    chain,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt or None,
                )

                # Record outcome for learning
                self._router.record_outcome(
                    route.model_id, prompt, success=True, latency_ms=resp.latency_ms
                )

            output = (
                f"[{resp.provider}/{resp.model} | {resp.latency_ms}ms"
                f" | {resp.usage.get('total_tokens', '?')} tokens]\n\n"
                f"{resp.content}"
            )
            return ToolResult(success=True, output=output)

        except LLMGatewayError as e:
            if model:
                self._router.record_outcome(model, prompt, success=False, error=str(e))
            return ToolResult(success=False, error=f"LLM call failed: {e}")
        except Exception as e:
            logger.error(f"llm_call error: {e}")
            return ToolResult(success=False, error=str(e))


class CodingAgentTool(Tool):
    """Delegate a coding task to a specialized model and get the result back."""

    name = "coding_agent"
    description = """Delegate a coding task to a specialized LLM coding agent.

The task runs on a cost-efficient model (defaults to free tier via CLIProxy or Gemini).
Results are returned directly — no subprocess management needed.

Use this when:
- You need to generate code (function, class, module, script)
- You need to refactor or fix existing code
- You need code review or bug analysis
- You need tests written for existing code

The coding agent has NO access to the filesystem. Provide all necessary
context (code snippets, file contents, requirements) in the task description.

Examples:
  coding_agent(task="Write a Python function that validates email addresses using regex")
  coding_agent(task="Refactor this class to use async/await: <code>", language="python")
  coding_agent(task="Write unit tests for: <code>", context="Uses pytest, Python 3.12")
"""

    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Detailed coding task description. Include all necessary context and code.",
            },
            "language": {
                "type": "string",
                "description": "Target programming language",
                "default": "python",
            },
            "context": {
                "type": "string",
                "description": "Additional context (project structure, dependencies, constraints)",
                "default": "",
            },
            "model": {
                "type": "string",
                "description": "Override model (leave empty for auto-routing to cheapest code-capable model)",
                "default": "",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum output tokens",
                "default": 8192,
            },
        },
        "required": ["task"],
    }

    _SYSTEM_PROMPT = """You are an expert coding agent. Your job is to write clean, production-quality code.

Rules:
1. Output ONLY code and brief explanations — no pleasantries
2. Follow the language's conventions and best practices
3. Include proper error handling and type hints where applicable
4. If the task is ambiguous, make reasonable assumptions and state them
5. Write code that is immediately runnable — no placeholders or TODOs
6. Use modern language features (Python 3.10+, ES2022+, etc.)"""

    def __init__(self, gateway: LLMGateway, router: ModelRouter) -> None:
        self._gw = gateway
        self._router = router

    async def execute(
        self,
        task: str,
        language: str = "python",
        context: str = "",
        model: str = "",
        max_tokens: int = 8192,
    ) -> ToolResult:
        try:
            # Build prompt
            parts = [f"Language: {language}", f"Task: {task}"]
            if context:
                parts.append(f"Context: {context}")
            prompt = "\n\n".join(parts)

            if model:
                spec = MODELS.get(model)
                if spec:
                    provider, model_id = spec.provider, spec.id
                elif "/" in model:
                    provider, model_id = model.split("/", 1)
                else:
                    return ToolResult(success=False, error=f"Unknown model: {model}")
            else:
                # Auto-route for code capability
                route = self._router.route(task, required_capability=Capability.CODE)
                provider = route.model.provider
                model_id = route.model_id

            if not self._gw.has_provider(provider):
                return ToolResult(
                    success=False,
                    error=f"Provider '{provider}' not available. Available: {self._gw.list_providers()}",
                )

            resp = await self._gw.chat(
                provider,
                model_id,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temp for code
                system_prompt=self._SYSTEM_PROMPT,
            )

            self._router.record_outcome(model_id, task, success=True)

            header = (
                f"[coding_agent: {resp.provider}/{resp.model} | "
                f"{resp.latency_ms}ms | {resp.usage.get('total_tokens', '?')} tokens]"
            )
            return ToolResult(success=True, output=f"{header}\n\n{resp.content}")

        except LLMGatewayError as e:
            return ToolResult(success=False, error=f"Coding agent failed: {e}")
        except Exception as e:
            logger.error(f"coding_agent error: {e}")
            return ToolResult(success=False, error=str(e))


class ListModelsTool(Tool):
    """List available models and their capabilities."""

    name = "list_models"
    description = """List all available LLM models with their tier, cost, and capabilities.

Use this to discover what models are available before making llm_call or coding_agent calls.
Models are dynamically loaded — run discover_models to fetch the latest from OpenRouter.
"""

    parameters = {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["free", "cheap", "standard", "premium", "research", ""],
                "description": "Filter by tier (leave empty for all)",
                "default": "",
            },
        },
        "required": [],
    }

    def __init__(self, gateway: LLMGateway, router: ModelRouter) -> None:
        self._gw = gateway
        self._router = router

    async def execute(self, tier: str = "") -> ToolResult:
        try:
            if tier:
                models = MODELS.list_by_tier(Tier(tier))
            else:
                models = MODELS.all()

            available_providers = set(self._gw.list_providers())

            lines = [f"Available LLM Models ({MODELS.count()} total):", ""]
            for t in [Tier.FREE, Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM, Tier.RESEARCH]:
                tier_models = sorted(
                    [m for m in models if m.tier == t],
                    key=lambda m: m.priority, reverse=True,
                )
                if not tier_models:
                    continue

                lines.append(f"── {t.value.upper()} ({len(tier_models)} models) ──")
                for m in tier_models[:15]:  # Cap display per tier
                    avail = "✅" if m.provider in available_providers else "❌"
                    cost = "FREE" if m.is_free else f"${m.cost_per_1m_input:.2f}/${m.cost_per_1m_output:.2f} /1M tok"
                    caps = ", ".join(sorted(c.value for c in m.capabilities))
                    aliases = f" ({', '.join(m.aliases)})" if m.aliases else ""
                    src = f" [{m.source}]" if m.source != "default" else ""
                    lines.append(f"  {avail} {m.id}{aliases}{src}")
                    lines.append(f"     {cost} | speed:{m.speed}/5 | pri:{m.priority} | {caps}")
                if len(tier_models) > 15:
                    lines.append(f"  ... and {len(tier_models) - 15} more")
                lines.append("")

            # Router stats
            router_stats = self._router.get_stats()
            if router_stats:
                lines.append("── ROUTING STATS ──")
                for mid, s in sorted(router_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]:
                    lines.append(
                        f"  {mid}: {s['total']} routes, "
                        f"{s['success_rate']:.0%} success, "
                        f"avg {s['avg_latency_ms']}ms"
                    )

            if MODELS.needs_discovery:
                lines.append("")
                lines.append("⚠️  Model cache is stale. Run discover_models to refresh.")

            return ToolResult(success=True, output="\n".join(lines))

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DiscoverModelsTool(Tool):
    """Fetch fresh models from OpenRouter API."""

    name = "discover_models"
    description = """Discover and sync available LLM models from OpenRouter API.

Fetches the latest model list with pricing, classifies them into tiers
(free/cheap/standard/premium), and saves to local cache.

Run this periodically (daily) or when you need access to new models.
Requires NANOBOT_PROVIDERS__OPENROUTER__API_KEY in environment.
"""

    parameters = {
        "type": "object",
        "properties": {
            "max_cost": {
                "type": "number",
                "description": "Max cost per 1M input tokens in USD (skip more expensive models)",
                "default": 20.0,
            },
        },
        "required": [],
    }

    async def execute(self, max_cost: float = 20.0) -> ToolResult:
        try:
            api_key = os.environ.get("NANOBOT_PROVIDERS__OPENROUTER__API_KEY", "")
            if not api_key:
                return ToolResult(
                    success=False,
                    error="NANOBOT_PROVIDERS__OPENROUTER__API_KEY not set. "
                          "Add it to .env to enable model discovery.",
                )

            result = await MODELS.discover(api_key=api_key, max_cost_per_1m=max_cost)

            lines = [
                f"Discovery complete:",
                f"  Added: {result['added']}",
                f"  Updated: {result['updated']}",
                f"  Skipped: {result['skipped']}",
                f"  Total models: {MODELS.count()}",
            ]

            if result.get("errors"):
                lines.append(f"  Errors: {len(result['errors'])}")
                for err in result["errors"][:5]:
                    lines.append(f"    - {err}")

            # Summary by tier
            lines.append("")
            lines.append("Models by tier:")
            for t in [Tier.FREE, Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM, Tier.RESEARCH]:
                count = len(MODELS.list_by_tier(t))
                if count:
                    lines.append(f"  {t.value}: {count}")

            return ToolResult(success=True, output="\n".join(lines))

        except Exception as e:
            logger.error(f"discover_models error: {e}")
            return ToolResult(success=False, error=str(e))
