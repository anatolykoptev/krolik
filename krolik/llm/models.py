"""Model registry with metadata for routing and cost optimization.

Each model entry contains:
- provider: which API backend to use
- tier: free / standard / premium / research
- cost_per_1k_input / cost_per_1k_output: in USD (0 = free)
- capabilities: set of tags (code, chat, vision, reasoning, search)
- context_window: max tokens
- speed: relative speed rating (1=slow, 5=fast)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    RESEARCH = "research"


class Capability(str, Enum):
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    REASONING = "reasoning"
    SEARCH = "search"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a single model."""

    id: str
    provider: str
    tier: Tier
    context_window: int = 128_000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    speed: int = 3
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    aliases: tuple[str, ...] = ()
    max_output_tokens: int = 8192

    @property
    def is_free(self) -> bool:
        return self.cost_per_1k_input == 0 and self.cost_per_1k_output == 0


# ---------------------------------------------------------------------------
# Built-in model catalogue
# Users can extend via config; this is the sensible default set.
# ---------------------------------------------------------------------------

_CATALOGUE: list[ModelSpec] = [
    # ── Free tier (Google) ────────────────────────────────────────
    ModelSpec(
        id="google/gemini-2.0-flash",
        provider="gemini",
        tier=Tier.FREE,
        context_window=1_000_000,
        speed=5,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION}),
        aliases=("gemini-flash", "flash"),
    ),
    ModelSpec(
        id="google/gemini-2.5-flash",
        provider="gemini",
        tier=Tier.FREE,
        context_window=1_000_000,
        speed=5,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("gemini-2.5-flash",),
    ),
    ModelSpec(
        id="google/gemini-2.5-pro",
        provider="gemini",
        tier=Tier.STANDARD,
        context_window=1_000_000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.01,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING, Capability.LONG_CONTEXT}),
        aliases=("gemini-pro", "pro"),
    ),

    # ── Free tier (via CLIProxy) ──────────────────────────────────
    ModelSpec(
        id="cliproxy/gemini-2.5-flash",
        provider="cliproxy",
        tier=Tier.FREE,
        context_window=1_000_000,
        speed=5,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION}),
        aliases=("cliproxy-flash",),
    ),
    ModelSpec(
        id="cliproxy/gemini-2.5-pro",
        provider="cliproxy",
        tier=Tier.FREE,
        context_window=1_000_000,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("cliproxy-pro",),
    ),
    ModelSpec(
        id="cliproxy/claude-sonnet-4-5",
        provider="cliproxy",
        tier=Tier.FREE,
        context_window=200_000,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.REASONING}),
        aliases=("cliproxy-sonnet",),
    ),

    # ── Standard tier ─────────────────────────────────────────────
    ModelSpec(
        id="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        tier=Tier.STANDARD,
        context_window=200_000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("sonnet", "claude-sonnet"),
    ),
    ModelSpec(
        id="openrouter/anthropic/claude-sonnet-4-5",
        provider="openrouter",
        tier=Tier.STANDARD,
        context_window=200_000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("or-sonnet",),
    ),

    # ── Premium tier ──────────────────────────────────────────────
    ModelSpec(
        id="anthropic/claude-opus-4-5",
        provider="anthropic",
        tier=Tier.PREMIUM,
        context_window=200_000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        speed=2,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("opus", "claude-opus"),
    ),
    ModelSpec(
        id="openrouter/anthropic/claude-opus-4-5",
        provider="openrouter",
        tier=Tier.PREMIUM,
        context_window=200_000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        speed=2,
        capabilities=frozenset({Capability.CHAT, Capability.CODE, Capability.VISION, Capability.REASONING}),
        aliases=("or-opus",),
    ),

    # ── Research tier ─────────────────────────────────────────────
    ModelSpec(
        id="openrouter/perplexity/sonar-pro",
        provider="openrouter",
        tier=Tier.RESEARCH,
        context_window=128_000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        speed=3,
        capabilities=frozenset({Capability.CHAT, Capability.SEARCH}),
        aliases=("perplexity", "sonar"),
    ),
    ModelSpec(
        id="openrouter/perplexity/sonar-deep-research",
        provider="openrouter",
        tier=Tier.RESEARCH,
        context_window=128_000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        speed=1,
        capabilities=frozenset({Capability.CHAT, Capability.SEARCH, Capability.REASONING}),
        aliases=("deep-research",),
    ),
]


class ModelRegistry:
    """Thread-safe model registry with lookup by id or alias."""

    def __init__(self) -> None:
        self._by_id: dict[str, ModelSpec] = {}
        self._by_alias: dict[str, ModelSpec] = {}
        for spec in _CATALOGUE:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        self._by_id[spec.id] = spec
        for alias in spec.aliases:
            self._by_alias[alias.lower()] = spec

    def get(self, id_or_alias: str) -> Optional[ModelSpec]:
        """Lookup by full id or alias (case-insensitive for aliases)."""
        return self._by_id.get(id_or_alias) or self._by_alias.get(id_or_alias.lower())

    def list_by_tier(self, tier: Tier) -> list[ModelSpec]:
        return [s for s in self._by_id.values() if s.tier == tier]

    def list_by_capability(self, cap: Capability) -> list[ModelSpec]:
        return [s for s in self._by_id.values() if cap in s.capabilities]

    def list_by_provider(self, provider: str) -> list[ModelSpec]:
        return [s for s in self._by_id.values() if s.provider == provider]

    def all(self) -> list[ModelSpec]:
        return list(self._by_id.values())

    def cheapest(self, capability: Capability | None = None) -> ModelSpec | None:
        """Return the cheapest model (optionally filtered by capability)."""
        candidates = self.list_by_capability(capability) if capability else self.all()
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.cost_per_1k_input + s.cost_per_1k_output)

    def fastest(self, capability: Capability | None = None) -> ModelSpec | None:
        """Return the fastest model (optionally filtered by capability)."""
        candidates = self.list_by_capability(capability) if capability else self.all()
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.speed)


# Singleton — importable everywhere
MODELS = ModelRegistry()
