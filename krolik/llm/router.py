"""Task router with scoring, signature tracking, and persistent learning.

Ported from OpenClaw's task-router plugin with full feature parity:
- 5 tiers: free / cheap / standard / premium / research
- Bilingual keywords (EN + RU) with weighted scoring
- Task type detection (trivial, simple, code, content, analysis, architect, research)
- MD5 signature tracking for deduplication
- Composite model scoring (test results + success rate + latency + priority)
- Persistent outcome recording to JSON file
- Cascade fallback with tier escalation
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from krolik.llm.models import MODELS, Capability, ModelSpec, Tier


# ── Tier score thresholds ─────────────────────────────────────────

TIER_THRESHOLDS: dict[str, tuple[int, int]] = {
    "free": (0, 25),
    "cheap": (26, 45),
    "standard": (46, 70),
    "premium": (71, 100),
}

# ── Base scores per task type ─────────────────────────────────────

BASE_SCORES: dict[str, int] = {
    "trivial": 10,
    "simple": 20,
    "content": 35,
    "general": 35,
    "code": 50,
    "research": 50,
    "analysis": 60,
    "architect": 80,
}

# ── Complexity keywords (EN + RU) with score modifiers ────────────

COMPLEXITY_KEYWORDS: dict[str, int] = {
    # EN — high complexity
    "architect": 25, "design system": 25, "security audit": 25,
    "refactor entire": 25, "performance optimization": 20,
    "rewrite": 20, "migrate": 18, "integrate": 15, "complex": 15,
    "analyze": 12, "debug": 12, "implement": 10, "add feature": 10,
    "create": 8, "build": 8, "fix bug": 6, "update": 5,
    # EN — low complexity (negative)
    "format": -10, "rename": -10, "typo": -15, "comment": -8,
    "simple": -15, "quick": -10, "trivial": -15, "lint": -12,
    # RU — high complexity
    "архитектур": 25, "спроектир": 25, "безопасност": 20,
    "рефактор": 20, "оптимизир": 15, "мигрир": 18, "интегрир": 15,
    "сложн": 15, "комплексн": 15, "анализ": 12, "отладк": 12,
    "реализ": 10, "создай подборк": 12, "напиши статью": 10,
    "создай": 8, "напиши": 6, "исправ": 5, "обнов": 5,
    "глубокий анализ": 20, "техническ": 12,
    "делегируй": 10, "оркестрируй": 12, "распараллель": 15,
    "парс": 8, "спарс": 8, "seo": 10,
    # RU — low complexity (negative)
    "опечатк": -15, "простой": -12, "быстро": -10, "мелк": -10,
    "перевед": 5,
}

RESEARCH_KEYWORDS: list[str] = [
    # EN
    "research", "find out", "search for", "compare", "alternatives",
    "what is", "how does", "explore", "investigate", "benchmark",
    "latest", "news", "look up",
    # RU
    "исследуй", "найди информацию", "сравни", "альтернатив",
    "что такое", "как работает", "изучи", "проанализируй рынок",
]


# ── Data types ────────────────────────────────────────────────────


@dataclass
class RouteResult:
    """Result of task routing."""

    model: ModelSpec
    tier: Tier
    score: int
    reason: str
    signature: str
    task_type: str
    fallbacks: list[ModelSpec] = field(default_factory=list)
    composite_score: float = 0.0
    alternatives: list[str] = field(default_factory=list)

    @property
    def provider(self) -> str:
        return self.model.provider

    @property
    def model_id(self) -> str:
        return self.model.id


@dataclass
class OutcomeRecord:
    """Persistent record of a routing outcome."""

    signature: str
    model_id: str
    outcome: str  # "success" | "fail"
    task_preview: str
    timestamp: float
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sig": self.signature,
            "model": self.model_id,
            "outcome": self.outcome,
            "task": self.task_preview,
            "ts": self.timestamp,
            "latency": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OutcomeRecord:
        return cls(
            signature=d["sig"],
            model_id=d["model"],
            outcome=d["outcome"],
            task_preview=d.get("task", ""),
            timestamp=d.get("ts", 0),
            latency_ms=d.get("latency", 0),
        )


class RouterError(Exception):
    """Raised when routing fails."""


# ── Main router ───────────────────────────────────────────────────


class ModelRouter:
    """Score-based task router with persistent learning.

    Usage::

        router = ModelRouter()
        result = router.route("Implement user authentication with OAuth2")
        # → RouteResult(tier=STANDARD, score=55, signature="a1b2c3...")

        result = router.route("Переведи на русский: Hello world")
        # → RouteResult(tier=FREE, score=15)
    """

    def __init__(
        self,
        prefer_free_providers: bool = True,
        available_providers: set[str] | None = None,
        outcomes_path: Path | None = None,
    ) -> None:
        self._prefer_free = prefer_free_providers
        self._available_providers = available_providers
        self._outcomes_path = outcomes_path or Path.home() / ".krolik" / "routing_outcomes.json"
        self._outcomes: list[OutcomeRecord] = []
        self._load_outcomes()

    # ── Public API ────────────────────────────────────────────────

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
            RouteResult with model, tier, score, signature, and fallbacks.
        """
        task_type = self._detect_task_type(task)
        is_research = task_type == "research"
        score = self._score_task(task, task_type)
        tier = self._score_to_tier(score, is_research)
        signature = self._compute_signature(task)

        # Pick best model using composite scoring
        model = self._select_model(tier, required_capability)
        if not model:
            # Escalate tier if no model available
            for fallback_tier in [Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM, Tier.FREE]:
                if fallback_tier != tier:
                    model = self._select_model(fallback_tier, required_capability)
                    if model:
                        tier = fallback_tier
                        break

        if not model:
            raise RouterError("No suitable model found for any tier")

        # Build fallback chain
        fallbacks = self._build_fallbacks(tier, model, required_capability)

        # Composite score for the selected model
        composite = self._composite_score(model)

        reason = self._build_reasoning(task, task_type, score, tier, model, composite)

        logger.debug(f"Router: '{task[:60]}...' → {tier.value} ({model.id}, score={score})")

        return RouteResult(
            model=model,
            tier=tier,
            score=score,
            reason=reason,
            signature=signature,
            task_type=task_type,
            fallbacks=fallbacks,
            composite_score=composite,
            alternatives=[f.id for f in fallbacks[:3]],
        )

    def record_outcome(
        self,
        model_id: str,
        task: str,
        success: bool,
        latency_ms: int = 0,
        error: str | None = None,
    ) -> None:
        """Record task outcome for learning loop. Persisted to disk."""
        sig = self._compute_signature(task)
        record = OutcomeRecord(
            signature=sig,
            model_id=model_id,
            outcome="success" if success else "fail",
            task_preview=task[:100],
            timestamp=time.time(),
            latency_ms=latency_ms,
        )
        self._outcomes.append(record)
        # Keep bounded
        if len(self._outcomes) > 1000:
            self._outcomes = self._outcomes[-500:]
        self._save_outcomes()

    def get_success_rate(self, model_id: str) -> float:
        """Return success rate for a model (0.0–1.0)."""
        relevant = [o for o in self._outcomes if o.model_id == model_id]
        if not relevant:
            return 1.0
        return sum(1 for o in relevant if o.outcome == "success") / len(relevant)

    def get_avg_latency(self, model_id: str) -> int:
        """Return average latency in ms for a model."""
        relevant = [o for o in self._outcomes if o.model_id == model_id and o.latency_ms > 0]
        if not relevant:
            return 2000
        return int(sum(o.latency_ms for o in relevant) / len(relevant))

    def get_stats(self) -> dict[str, Any]:
        """Return router statistics."""
        model_stats: dict[str, dict[str, Any]] = {}
        for o in self._outcomes:
            if o.model_id not in model_stats:
                model_stats[o.model_id] = {"total": 0, "success": 0, "latencies": []}
            s = model_stats[o.model_id]
            s["total"] += 1
            if o.outcome == "success":
                s["success"] += 1
            if o.latency_ms > 0:
                s["latencies"].append(o.latency_ms)

        return {
            mid: {
                "total": s["total"],
                "success_rate": s["success"] / max(s["total"], 1),
                "avg_latency_ms": int(sum(s["latencies"]) / max(len(s["latencies"]), 1)),
            }
            for mid, s in model_stats.items()
        }

    # ── Task type detection ───────────────────────────────────────

    @staticmethod
    def _detect_task_type(task: str) -> str:
        lower = task.lower()
        if any(kw in lower for kw in RESEARCH_KEYWORDS):
            return "research"
        if re.search(r"architect|design.*system|infrastructure", lower):
            return "architect"
        if re.search(r"typo|format|rename|lint", lower):
            return "trivial"
        if re.search(r"simple|quick|trivial", lower):
            return "simple"
        if re.search(r"code|implement|function|class|module|api", lower):
            return "code"
        if re.search(r"write|article|content|blog|documentation", lower):
            return "content"
        if re.search(r"analyze|report|data|metrics", lower):
            return "analysis"
        return "general"

    # ── Scoring engine ────────────────────────────────────────────

    def _score_task(self, task: str, task_type: str) -> int:
        score = BASE_SCORES.get(task_type, 40)
        lower = task.lower()

        for kw, modifier in COMPLEXITY_KEYWORDS.items():
            if kw in lower:
                score += modifier

        # Length signals complexity
        word_count = len(task.split())
        if word_count > 50:
            score += 10
        elif word_count < 10:
            score -= 5

        # Code blocks signal technical work
        if "```" in task:
            score += 8

        # Multiple requirements
        bullet_count = task.count("- ") + task.count("* ") + task.count("\n")
        if bullet_count > 5:
            score += 10

        # Learning: if free tier has low success → bump up
        for fm in MODELS.list_by_tier(Tier.FREE):
            if self.get_success_rate(fm.id) < 0.5 and score < TIER_THRESHOLDS["free"][1]:
                score = TIER_THRESHOLDS["cheap"][0] + 5
                break

        return max(5, min(95, score))

    @staticmethod
    def _score_to_tier(score: int, is_research: bool) -> Tier:
        if is_research:
            return Tier.RESEARCH
        for tier_name, (low, high) in TIER_THRESHOLDS.items():
            if low <= score <= high:
                return Tier(tier_name)
        return Tier.PREMIUM if score > 85 else Tier.FREE

    @staticmethod
    def _compute_signature(task: str) -> str:
        """Normalize task and compute MD5 signature for deduplication."""
        normalized = task.lower()
        normalized = re.sub(r"\b\w+\.(py|js|ts|tsx|sh|md|json|yaml|yml)\b", r"FILE.\1", normalized)
        normalized = re.sub(r"\d+", "N", normalized)
        normalized = " ".join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    # ── Composite scoring ─────────────────────────────────────────

    def _composite_score(self, model: ModelSpec) -> float:
        """Calculate composite score (0–100) using all available data."""
        success_rate = self.get_success_rate(model.id) * 100
        avg_latency = self.get_avg_latency(model.id)
        latency_penalty = min(30, avg_latency / 200)

        composite = (
            (model.priority * 0.4)
            + (success_rate * 0.3)
            + (model.speed * 4)  # speed 1-5 → 4-20 points
            - latency_penalty
        )
        return max(0, min(100, composite))

    # ── Model selection ───────────────────────────────────────────

    def _select_model(
        self, tier: Tier, capability: Capability | None
    ) -> ModelSpec | None:
        candidates = MODELS.list_by_tier(tier)

        if capability:
            candidates = [c for c in candidates if capability in c.capabilities]

        if self._available_providers:
            candidates = [
                c for c in candidates if c.provider in self._available_providers
            ]

        if not candidates:
            return None

        # Rank by composite score
        scored = [(c, self._composite_score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Prefer free providers when configured
        if self._prefer_free:
            free_candidates = [(c, s) for c, s in scored if c.is_free]
            if free_candidates:
                return free_candidates[0][0]

        return scored[0][0]

    def _build_fallbacks(
        self,
        tier: Tier,
        primary: ModelSpec,
        capability: Capability | None,
    ) -> list[ModelSpec]:
        fallbacks: list[ModelSpec] = []
        seen = {primary.id}

        def _add_candidates(candidates: list[ModelSpec]) -> None:
            for c in candidates:
                if c.id not in seen:
                    if not capability or capability in c.capabilities:
                        if not self._available_providers or c.provider in self._available_providers:
                            fallbacks.append(c)
                            seen.add(c.id)

        # Same tier, different model
        _add_candidates(MODELS.list_by_tier(tier))

        # Escalate: cheap → standard → premium
        tier_order = [Tier.FREE, Tier.CHEAP, Tier.STANDARD, Tier.PREMIUM]
        if tier in tier_order:
            idx = tier_order.index(tier)
            for higher in tier_order[idx + 1:]:
                _add_candidates(MODELS.list_by_tier(higher))

        return fallbacks[:4]

    # ── Reasoning ─────────────────────────────────────────────────

    @staticmethod
    def _build_reasoning(
        task: str,
        task_type: str,
        score: int,
        tier: Tier,
        model: ModelSpec,
        composite: float,
    ) -> str:
        parts = [
            f"type={task_type}",
            f"base={BASE_SCORES.get(task_type, 40)}",
            f"→{tier.value}(score={score})",
            f"model={model.id}",
        ]
        if composite != 50:
            parts.append(f"rating={composite:.0f}")
        return " | ".join(parts)

    # ── Persistence ───────────────────────────────────────────────

    def _load_outcomes(self) -> None:
        if not self._outcomes_path.exists():
            return
        try:
            data = json.loads(self._outcomes_path.read_text())
            self._outcomes = [OutcomeRecord.from_dict(d) for d in data.get("outcomes", [])]
            logger.debug(f"Loaded {len(self._outcomes)} routing outcomes")
        except Exception as e:
            logger.warning(f"Failed to load routing outcomes: {e}")

    def _save_outcomes(self) -> None:
        self._outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "outcomes": [o.to_dict() for o in self._outcomes],
        }
        self._outcomes_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.debug(f"Saved {len(self._outcomes)} routing outcomes")
