"""Dynamic model registry with auto-discovery from provider APIs.

Models are loaded from three sources (in priority order):
1. Local cache file (~/.krolik/models.json) — persisted discoveries
2. OpenRouter /api/v1/models API — live model list with pricing
3. Minimal hardcoded defaults — only used when nothing else is available

The registry auto-classifies models into tiers by cost:
- free:     $0 per 1M tokens
- cheap:    <$0.05 per 1M tokens
- standard: <$0.15 per 1M tokens
- premium:  >=$0.15 per 1M tokens
- research: models with search capabilities (Perplexity, etc.)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class Tier(str, Enum):
    FREE = "free"
    CHEAP = "cheap"
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


# ── Cost thresholds (per 1M input tokens, USD) ───────────────────
COST_TIER_THRESHOLDS = {
    "free": 0.0,
    "cheap": 0.05,
    "standard": 0.15,
    # anything above 0.15 → premium
}

# Models that should never be auto-discovered (too expensive, deprecated, etc.)
_SKIP_PATTERNS = frozenset({
    "openai/o1", "openai/gpt-4-32k", "mistralai/codestral-mamba",
})


@dataclass
class ModelSpec:
    """Specification for a single model."""

    id: str
    provider: str
    tier: Tier
    context_window: int = 128_000
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    priority: int = 50
    speed: int = 3
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    aliases: tuple[str, ...] = ()
    max_output_tokens: int = 8192
    status: str = "active"
    added_at: str = ""
    source: str = "default"

    @property
    def is_free(self) -> bool:
        return self.cost_per_1m_input == 0 and self.cost_per_1m_output == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "tier": self.tier.value,
            "context_window": self.context_window,
            "cost_per_1m_input": self.cost_per_1m_input,
            "cost_per_1m_output": self.cost_per_1m_output,
            "priority": self.priority,
            "speed": self.speed,
            "capabilities": [c.value for c in self.capabilities],
            "aliases": list(self.aliases),
            "max_output_tokens": self.max_output_tokens,
            "status": self.status,
            "added_at": self.added_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            id=d["id"],
            provider=d.get("provider", _infer_provider(d["id"])),
            tier=Tier(d.get("tier", "standard")),
            context_window=d.get("context_window", 128_000),
            cost_per_1m_input=d.get("cost_per_1m_input", 0.0),
            cost_per_1m_output=d.get("cost_per_1m_output", 0.0),
            priority=d.get("priority", 50),
            speed=d.get("speed", 3),
            capabilities=frozenset(Capability(c) for c in d.get("capabilities", ["chat"])),
            aliases=tuple(d.get("aliases", [])),
            max_output_tokens=d.get("max_output_tokens", 8192),
            status=d.get("status", "active"),
            added_at=d.get("added_at", ""),
            source=d.get("source", "config"),
        )


def _infer_provider(model_id: str) -> str:
    """Infer provider from model id prefix."""
    if model_id.startswith("cliproxy/"):
        return "cliproxy"
    if model_id.startswith("openrouter/"):
        return "openrouter"
    first = model_id.split("/")[0] if "/" in model_id else model_id
    provider_map = {
        "anthropic": "anthropic", "google": "gemini", "gemini": "gemini",
        "openai": "openai", "meta-llama": "openrouter", "mistralai": "openrouter",
        "deepseek": "openrouter", "qwen": "openrouter",
    }
    return provider_map.get(first, "openrouter")


def classify_tier_by_cost(cost_per_1m: float) -> Tier:
    """Classify a model into a tier based on its cost per 1M input tokens."""
    if cost_per_1m <= 0:
        return Tier.FREE
    if cost_per_1m < COST_TIER_THRESHOLDS["cheap"]:
        return Tier.CHEAP
    if cost_per_1m < COST_TIER_THRESHOLDS["standard"]:
        return Tier.STANDARD
    return Tier.PREMIUM


def calculate_priority(
    model_id: str,
    cost: float,
    context_length: int,
    created_ts: float | None = None,
) -> int:
    """Calculate priority score (0–100) for model ranking within a tier."""
    priority = 50

    # Cost (lower = better for budget tiers)
    if cost == 0:
        priority += 20
    elif cost < 0.02:
        priority += 15
    elif cost < 0.05:
        priority += 10

    # Context length bonus
    if context_length >= 128_000:
        priority += 15
    elif context_length >= 32_000:
        priority += 10
    elif context_length >= 16_000:
        priority += 5

    # Recency bonus
    if created_ts:
        days_old = (time.time() - created_ts) / 86400
        if days_old < 30:
            priority += 10
        elif days_old < 90:
            priority += 5

    # Brand bonus
    if any(name in model_id.lower() for name in ("claude", "gemini", "gpt-4", "gpt-5")):
        priority += 5

    return max(0, min(100, priority))


def _detect_capabilities(model_id: str, description: str = "") -> frozenset[Capability]:
    """Detect capabilities from model id and description."""
    caps = {Capability.CHAT}
    lower = (model_id + " " + description).lower()

    if any(kw in lower for kw in ("code", "codex", "codestral", "starcoder", "deepseek-coder")):
        caps.add(Capability.CODE)
    if any(kw in lower for kw in ("vision", "image", "multimodal", "4o", "gemini")):
        caps.add(Capability.VISION)
    if any(kw in lower for kw in ("reason", "thinking", "o1", "o3", "r1")):
        caps.add(Capability.REASONING)
    if any(kw in lower for kw in ("sonar", "perplexity", "search")):
        caps.add(Capability.SEARCH)
    # Most modern large models can code
    if any(kw in lower for kw in ("claude", "gpt-4", "gpt-5", "gemini", "llama-3")):
        caps.add(Capability.CODE)
        caps.add(Capability.REASONING)

    return frozenset(caps)


# ── Minimal hardcoded defaults (used ONLY when no cache/API) ─────

_DEFAULTS: list[dict[str, Any]] = [
    # CLIProxy (free via local OAuth)
    {"id": "cliproxy/gemini-2.5-flash", "provider": "cliproxy", "tier": "free",
     "context_window": 1_000_000, "speed": 5,
     "capabilities": ["chat", "code", "vision"], "aliases": ["cliproxy-flash"],
     "source": "default"},
    {"id": "cliproxy/claude-sonnet-4-5", "provider": "cliproxy", "tier": "free",
     "context_window": 200_000, "speed": 3,
     "capabilities": ["chat", "code", "reasoning"], "aliases": ["cliproxy-sonnet"],
     "source": "default"},
    # Research
    {"id": "openrouter/perplexity/sonar-pro", "provider": "openrouter", "tier": "research",
     "context_window": 128_000, "cost_per_1m_input": 3.0, "cost_per_1m_output": 15.0,
     "capabilities": ["chat", "search"], "aliases": ["perplexity", "sonar"],
     "source": "default"},
]


class ModelRegistry:
    """Dynamic model registry with multi-source loading.

    Loading order:
    1. Hardcoded defaults (always loaded, lowest priority)
    2. Cache file (~/.krolik/models.json) — overrides defaults
    3. discover() — fetches from OpenRouter API and merges into cache
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._by_id: dict[str, ModelSpec] = {}
        self._by_alias: dict[str, ModelSpec] = {}
        self._cache_path = cache_path or Path.home() / ".krolik" / "models.json"
        self._last_discovery: float = 0.0

        # Load defaults first
        for d in _DEFAULTS:
            self.register(ModelSpec.from_dict(d))

        # Load cached models (overrides defaults)
        self._load_cache()

    def register(self, spec: ModelSpec) -> None:
        """Register a model (overwrites existing with same id)."""
        if spec.status != "active":
            self._by_id.pop(spec.id, None)
            return
        self._by_id[spec.id] = spec
        for alias in spec.aliases:
            self._by_alias[alias.lower()] = spec

    def unregister(self, model_id: str) -> None:
        self._by_id.pop(model_id, None)

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

    def count(self) -> int:
        return len(self._by_id)

    def cheapest(self, capability: Capability | None = None) -> ModelSpec | None:
        candidates = self.list_by_capability(capability) if capability else self.all()
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.cost_per_1m_input + s.cost_per_1m_output)

    def fastest(self, capability: Capability | None = None) -> ModelSpec | None:
        candidates = self.list_by_capability(capability) if capability else self.all()
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.speed)

    # ── Persistence ───────────────────────────────────────────────

    def _load_cache(self) -> None:
        """Load models from cache file."""
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text())
            models = data.get("models", [])
            self._last_discovery = data.get("last_discovery", 0.0)
            count = 0
            for d in models:
                try:
                    self.register(ModelSpec.from_dict(d))
                    count += 1
                except Exception:
                    continue
            if count:
                logger.debug(f"Loaded {count} models from cache")
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")

    def save_cache(self) -> None:
        """Persist current registry to cache file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "last_discovery": self._last_discovery,
            "models": [s.to_dict() for s in self._by_id.values()],
        }
        self._cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.debug(f"Saved {len(self._by_id)} models to cache")

    @property
    def needs_discovery(self) -> bool:
        """True if no discovery has been done or it's been >24h."""
        if self._last_discovery == 0:
            return True
        return (time.time() - self._last_discovery) > 86400

    # ── Discovery from OpenRouter API ─────────────────────────────

    async def discover(
        self,
        api_key: str,
        max_cost_per_1m: float = 20.0,
    ) -> dict[str, Any]:
        """Fetch models from OpenRouter API and merge into registry.

        Args:
            api_key: OpenRouter API key.
            max_cost_per_1m: Skip models more expensive than this (per 1M input tokens).

        Returns:
            Dict with added/updated/skipped counts.
        """
        import httpx

        result = {"added": 0, "updated": 0, "skipped": 0, "errors": []}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            api_models = data.get("data", [])
            logger.info(f"Discovery: fetched {len(api_models)} models from OpenRouter")

            today = time.strftime("%Y-%m-%d")

            for m in api_models:
                try:
                    model_id = m.get("id", "")
                    if not model_id:
                        continue

                    full_id = f"openrouter/{model_id}"

                    # Skip patterns
                    if any(pat in full_id for pat in _SKIP_PATTERNS):
                        result["skipped"] += 1
                        continue

                    # Parse cost
                    prompt_cost_per_token = float(m.get("pricing", {}).get("prompt", "0"))
                    completion_cost_per_token = float(m.get("pricing", {}).get("completion", "0"))
                    cost_per_1m_input = prompt_cost_per_token * 1_000_000
                    cost_per_1m_output = completion_cost_per_token * 1_000_000

                    if cost_per_1m_input > max_cost_per_1m:
                        result["skipped"] += 1
                        continue

                    ctx = m.get("context_length", 128_000)
                    created = m.get("created", None)
                    description = m.get("description", "")

                    tier = classify_tier_by_cost(cost_per_1m_input)

                    # Research override
                    if any(kw in model_id.lower() for kw in ("perplexity", "sonar")):
                        tier = Tier.RESEARCH

                    priority = calculate_priority(full_id, cost_per_1m_input, ctx, created)
                    capabilities = _detect_capabilities(model_id, description)

                    speed = 3
                    if "flash" in model_id.lower() or cost_per_1m_input == 0:
                        speed = 5
                    elif cost_per_1m_input > 5.0:
                        speed = 2

                    existing = self._by_id.get(full_id)
                    is_new = existing is None

                    spec = ModelSpec(
                        id=full_id,
                        provider="openrouter",
                        tier=tier,
                        context_window=ctx,
                        cost_per_1m_input=cost_per_1m_input,
                        cost_per_1m_output=cost_per_1m_output,
                        priority=priority,
                        speed=speed,
                        capabilities=capabilities,
                        max_output_tokens=m.get("top_provider", {}).get("max_completion_tokens", 8192) or 8192,
                        status="active",
                        added_at=existing.added_at if existing else today,
                        source="openrouter",
                    )
                    self.register(spec)

                    if is_new:
                        result["added"] += 1
                    else:
                        result["updated"] += 1

                except Exception as e:
                    result["errors"].append(f"{m.get('id', '?')}: {e}")
                    continue

            self._last_discovery = time.time()
            self.save_cache()
            logger.info(
                f"Discovery complete: {result['added']} added, "
                f"{result['updated']} updated, {result['skipped']} skipped"
            )

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Discovery failed: {e}")

        return result


# Singleton — importable everywhere.
# Loads defaults + cache on import. Call MODELS.discover() for live data.
MODELS = ModelRegistry()
