"""Tier-based model router for cost optimization and task matching.

Routes tasks to the cheapest model that can handle them:
- Free tier    (score 0-40):  simple tasks, translations, summaries
- Standard tier (score 41-70): code, refactoring, analysis
- Premium tier  (score 71+):   architecture, complex reasoning
- Research tier  (keyword):    web search, fact-checking

The router scores a task description and returns the best model + provider.
Supports learning from outcomes to improve routing over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from krolik.llm.models import MODELS, Capability, ModelSpec, Tier


# ── Keyword sets for scoring ──────────────────────────────────────

_PREMIUM_KEYWORDS = frozenset({
    "architect", "design", "system design", "complex", "critical",
    "production", "security", "audit", "migrate", "refactor entire",
    "scalab", "distributed", "consensus", "tradeoff", "trade-off",
})

_STANDARD_KEYWORDS = frozenset({
    "implement", "code", "function", "class", "test", "debug",
    "fix bug", "refactor", "api", "endpoint", "database", "query",
    "parse", "transform", "convert", "validate", "module",
    "typescript", "python", "rust", "javascript", "golang",
})

_RESEARCH_KEYWORDS = frozenset({
    "search", "find", "research", "look up", "what is",
    "latest", "news", "trend", "compare", "review",
    "who is", "when did", "how does", "explain",
})

_FREE_KEYWORDS = frozenset({
    "translate", "summarize", "summary", "format", "rewrite",
    "grammar", "spell", "list", "hello", "hi", "thanks",
    "simple", "quick", "short", "brief",
})


@dataclass
class RouteResult:
    """Result of task routing."""

    model: ModelSpec
    tier: Tier
    score: int
    reason: str
    fallbacks: list[ModelSpec] = field(default_factory=list)

    @property
    def provider(self) -> str:
        return self.model.provider

    @property
    def model_id(self) -> str:
        return self.model.id


class ModelRouter:
    """Score-based task router with configurable thresholds.

    Usage::

        router = ModelRouter()
        result = router.route("Implement user authentication with OAuth2")
        # → RouteResult(tier=STANDARD, model=sonnet, score=55)

        result = router.route("Design distributed event sourcing architecture")
        # → RouteResult(tier=PREMIUM, model=opus, score=82)
    """

    def __init__(
        self,
        free_threshold: int = 40,
        standard_threshold: int = 70,
        prefer_free_providers: bool = True,
        available_providers: set[str] | None = None,
    ) -> None:
        self._free_threshold = free_threshold
        self._standard_threshold = standard_threshold
        self._prefer_free = prefer_free_providers
        self._available_providers = available_providers
        self._outcome_history: list[_Outcome] = []

    def route(
        self,
        task: str,
        required_capability: Capability | None = None,
    ) -> RouteResult:
        """Route a task to the best model.

        Args:
            task: Natural-language task description.
            required_capability: Hard requirement (e.g. Capability.CODE).

        Returns:
            RouteResult with model, tier, score, and fallbacks.
        """
        score = self._score_task(task)
        tier = self._score_to_tier(score, task)

        # Pick best model for the tier
        model = self._select_model(tier, required_capability)
        if not model:
            # Escalate tier if no model available
            for fallback_tier in [Tier.STANDARD, Tier.PREMIUM, Tier.FREE]:
                if fallback_tier != tier:
                    model = self._select_model(fallback_tier, required_capability)
                    if model:
                        tier = fallback_tier
                        break

        if not model:
            raise RouterError("No suitable model found for any tier")

        # Build fallback chain
        fallbacks = self._build_fallbacks(tier, model, required_capability)

        reason = self._explain_score(score, tier, task)
        logger.debug(f"Router: '{task[:60]}...' → {tier.value} ({model.id}, score={score})")

        return RouteResult(
            model=model,
            tier=tier,
            score=score,
            reason=reason,
            fallbacks=fallbacks,
        )

    def record_outcome(
        self,
        model_id: str,
        task: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record task outcome for learning loop.

        If a model consistently fails, future routing will escalate.
        """
        self._outcome_history.append(
            _Outcome(model_id=model_id, task_hash=hash(task), success=success)
        )
        # Keep bounded
        if len(self._outcome_history) > 500:
            self._outcome_history = self._outcome_history[-300:]

    def get_success_rate(self, model_id: str) -> float:
        """Return success rate for a model (0.0–1.0)."""
        relevant = [o for o in self._outcome_history if o.model_id == model_id]
        if not relevant:
            return 1.0  # Assume good until proven otherwise
        return sum(1 for o in relevant if o.success) / len(relevant)

    # ── Internal scoring ──────────────────────────────────────────

    def _score_task(self, task: str) -> int:
        """Score a task from 0–100 based on complexity signals."""
        task_lower = task.lower()
        score = 50  # Base score

        # Keyword matching
        for kw in _PREMIUM_KEYWORDS:
            if kw in task_lower:
                score += 12

        for kw in _STANDARD_KEYWORDS:
            if kw in task_lower:
                score += 5

        for kw in _FREE_KEYWORDS:
            if kw in task_lower:
                score -= 10

        # Length signals complexity
        word_count = len(task.split())
        if word_count > 50:
            score += 10
        elif word_count < 10:
            score -= 10

        # Code blocks signal technical work
        if "```" in task:
            score += 8

        # Multiple requirements signal complexity
        bullet_count = task.count("- ") + task.count("* ") + task.count("\n")
        if bullet_count > 5:
            score += 10

        # Learning adjustment: if free tier has low success → bump up
        free_models = MODELS.list_by_tier(Tier.FREE)
        for fm in free_models:
            if self.get_success_rate(fm.id) < 0.5 and score < self._free_threshold:
                score = self._free_threshold + 5
                break

        return max(0, min(100, score))

    def _score_to_tier(self, score: int, task: str) -> Tier:
        """Map score to tier, with research override."""
        task_lower = task.lower()

        # Research override
        for kw in _RESEARCH_KEYWORDS:
            if kw in task_lower:
                # Check if it's primarily a search task
                research_count = sum(1 for k in _RESEARCH_KEYWORDS if k in task_lower)
                code_count = sum(1 for k in _STANDARD_KEYWORDS if k in task_lower)
                if research_count > code_count:
                    return Tier.RESEARCH

        if score <= self._free_threshold:
            return Tier.FREE
        if score <= self._standard_threshold:
            return Tier.STANDARD
        return Tier.PREMIUM

    def _select_model(
        self, tier: Tier, capability: Capability | None
    ) -> ModelSpec | None:
        """Select best model for a tier, respecting provider availability."""
        candidates = MODELS.list_by_tier(tier)

        if capability:
            candidates = [c for c in candidates if capability in c.capabilities]

        if self._available_providers:
            candidates = [
                c for c in candidates if c.provider in self._available_providers
            ]

        if not candidates:
            return None

        # Prefer free providers (CLIProxy) when configured
        if self._prefer_free:
            free_first = [c for c in candidates if c.is_free]
            if free_first:
                return max(free_first, key=lambda c: c.speed)

        # Otherwise pick fastest
        return max(candidates, key=lambda c: c.speed)

    def _build_fallbacks(
        self,
        tier: Tier,
        primary: ModelSpec,
        capability: Capability | None,
    ) -> list[ModelSpec]:
        """Build ordered fallback chain (different provider, same or higher tier)."""
        fallbacks: list[ModelSpec] = []
        seen_providers = {primary.provider}

        # Same tier, different provider
        for candidate in MODELS.list_by_tier(tier):
            if candidate.provider not in seen_providers:
                if not capability or capability in candidate.capabilities:
                    if not self._available_providers or candidate.provider in self._available_providers:
                        fallbacks.append(candidate)
                        seen_providers.add(candidate.provider)

        # Escalate: next tier up
        tier_order = [Tier.FREE, Tier.STANDARD, Tier.PREMIUM]
        if tier in tier_order:
            idx = tier_order.index(tier)
            for higher_tier in tier_order[idx + 1 :]:
                for candidate in MODELS.list_by_tier(higher_tier):
                    if candidate.provider not in seen_providers:
                        if not capability or capability in candidate.capabilities:
                            if not self._available_providers or candidate.provider in self._available_providers:
                                fallbacks.append(candidate)
                                seen_providers.add(candidate.provider)
                                break

        return fallbacks[:3]  # Max 3 fallbacks

    @staticmethod
    def _explain_score(score: int, tier: Tier, task: str) -> str:
        """Human-readable explanation of routing decision."""
        task_preview = task[:80] + ("..." if len(task) > 80 else "")
        return f"score={score} → {tier.value} | task: {task_preview}"


# ── Supporting types ──────────────────────────────────────────────


@dataclass
class _Outcome:
    model_id: str
    task_hash: int
    success: bool


class RouterError(Exception):
    """Raised when routing fails."""
