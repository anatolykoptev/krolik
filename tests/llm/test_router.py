"""Tests for the task router."""

import json
import pytest
from pathlib import Path

from krolik.llm.models import Capability, Tier
from krolik.llm.router import ModelRouter, RouteResult, RouterError, OutcomeRecord


@pytest.fixture
def router(tmp_path):
    return ModelRouter(outcomes_path=tmp_path / "outcomes.json")


# ── Tier routing ──────────────────────────────────────────────────


def test_simple_task_routes_low_score(router):
    result = router.route("Переведи 'hello' на русский")
    assert result.score <= 45  # general base=35, no complexity keywords


def test_code_task_scores_above_cheap(router):
    result = router.route("Implement a REST API endpoint for user authentication with JWT tokens")
    assert result.score > 45  # code base=50, + implement + api keywords


def test_complex_task_scores_premium_range(router):
    result = router.route(
        "Design a distributed event sourcing architecture with CQRS pattern, "
        "handle eventual consistency, design the consensus protocol, "
        "and plan the migration strategy from the monolith"
    )
    assert result.score > 70  # architect base=80, + design system + migrate keywords


def test_research_task_routes_to_research(router):
    result = router.route("Search for the latest news about OpenAI and explain what happened")
    assert result.tier == Tier.RESEARCH


def test_trivial_task_routes_low(router):
    result = router.route("Fix this typo")
    assert result.score < 30


# ── Task type detection ───────────────────────────────────────────


def test_task_type_code(router):
    result = router.route("Implement a function to parse JSON")
    assert result.task_type == "code"


def test_task_type_architect(router):
    result = router.route("Design system architecture for microservices")
    assert result.task_type == "architect"


def test_task_type_research(router):
    result = router.route("Research the latest trends in AI")
    assert result.task_type == "research"


def test_task_type_trivial(router):
    result = router.route("Fix the typo in the comment")
    assert result.task_type == "trivial"


def test_task_type_content(router):
    result = router.route("Write a blog article about Rust")
    assert result.task_type == "content"


# ── Route result ──────────────────────────────────────────────────


def test_route_result_has_model(router):
    result = router.route("Hello world")
    assert result.model is not None
    assert result.model_id
    assert result.provider


def test_route_result_has_signature(router):
    result = router.route("Write a function")
    assert result.signature
    assert len(result.signature) == 12  # MD5 truncated to 12


def test_route_result_has_reason(router):
    result = router.route("Write a function")
    assert result.reason
    assert "score=" in result.reason
    assert "type=" in result.reason


def test_route_with_required_capability(router):
    result = router.route("Do something", required_capability=Capability.CODE)
    assert Capability.CODE in result.model.capabilities


def test_fallbacks_are_populated(router):
    result = router.route("Implement user authentication")
    assert isinstance(result.fallbacks, list)


# ── Signature tracking ────────────────────────────────────────────


def test_same_task_same_signature(router):
    r1 = router.route("Write a function")
    r2 = router.route("Write a function")
    assert r1.signature == r2.signature


def test_numbers_normalized_in_signature(router):
    r1 = router.route("Fix bug #123 in file.py")
    r2 = router.route("Fix bug #456 in file.py")
    assert r1.signature == r2.signature


def test_different_tasks_different_signature(router):
    r1 = router.route("Write a function")
    r2 = router.route("Design an architecture")
    assert r1.signature != r2.signature


# ── Outcome recording ────────────────────────────────────────────


def test_record_outcome_and_success_rate(router):
    router.record_outcome("cliproxy/gemini-2.5-flash", "test task 1", success=True)
    router.record_outcome("cliproxy/gemini-2.5-flash", "test task 2", success=True)
    router.record_outcome("cliproxy/gemini-2.5-flash", "test task 3", success=False)

    rate = router.get_success_rate("cliproxy/gemini-2.5-flash")
    assert abs(rate - 2 / 3) < 0.01


def test_unknown_model_defaults_to_100_percent(router):
    rate = router.get_success_rate("nonexistent/model")
    assert rate == 1.0


def test_avg_latency_default(router):
    assert router.get_avg_latency("nonexistent/model") == 2000


def test_avg_latency_recorded(router):
    router.record_outcome("m1", "t1", success=True, latency_ms=100)
    router.record_outcome("m1", "t2", success=True, latency_ms=200)
    assert router.get_avg_latency("m1") == 150


def test_outcomes_persisted(tmp_path):
    path = tmp_path / "outcomes.json"
    r1 = ModelRouter(outcomes_path=path)
    r1.record_outcome("test/model", "test task", success=True, latency_ms=500)

    # New router loads from same file
    r2 = ModelRouter(outcomes_path=path)
    assert r2.get_success_rate("test/model") == 1.0


def test_get_stats(router):
    router.record_outcome("m1", "t1", success=True, latency_ms=100)
    router.record_outcome("m1", "t2", success=False, latency_ms=200)
    stats = router.get_stats()
    assert "m1" in stats
    assert stats["m1"]["total"] == 2
    assert stats["m1"]["success_rate"] == 0.5


# ── Learning ──────────────────────────────────────────────────────


def test_learning_escalation(router):
    """If free tier has low success rate, router should escalate."""
    for _ in range(10):
        router.record_outcome("cliproxy/gemini-2.5-flash", "fail task", success=False)

    result = router.route("Простой перевод текста")
    assert result.score > 25 or result.tier != Tier.FREE


# ── Score bounds ──────────────────────────────────────────────────


def test_score_clamped(router):
    result = router.route("hi")
    assert 5 <= result.score <= 95

    result = router.route(
        "Design architect system design complex critical production security "
        "audit migrate refactor entire " * 3
    )
    assert 5 <= result.score <= 95


# ── RU keywords ───────────────────────────────────────────────────


def test_ru_complex_keywords(router):
    result = router.route("Спроектируй архитектуру распределённой системы с безопасностью")
    assert result.score > 60


def test_ru_simple_keywords(router):
    result = router.route("Простой мелкий фикс опечатки")
    assert result.score < 30


# ── Filters ───────────────────────────────────────────────────────


def test_available_providers_filter(tmp_path):
    router = ModelRouter(
        available_providers={"cliproxy"},
        outcomes_path=tmp_path / "outcomes.json",
    )
    result = router.route("Write some code")
    assert result.model.provider == "cliproxy"


def test_prefer_free_providers(router):
    result = router.route("Переведи текст")
    assert result.model.is_free


def test_code_bullets_increase_score(router):
    simple = router.route("Fix a bug")
    complex_task = router.route(
        "Fix the following bugs:\n"
        "- Authentication fails on timeout\n"
        "- Database connection pool leaks\n"
        "- API returns 500 on edge case\n"
        "- Memory leak in worker process\n"
        "- Race condition in queue consumer\n"
        "- Missing error handling in payment flow\n"
        "```python\ndef broken():\n    pass\n```"
    )
    assert complex_task.score > simple.score


# ── Outcome serialization ─────────────────────────────────────────


def test_outcome_record_roundtrip():
    o = OutcomeRecord(
        signature="abc123",
        model_id="test/model",
        outcome="success",
        task_preview="test",
        timestamp=12345.0,
        latency_ms=100,
    )
    d = o.to_dict()
    restored = OutcomeRecord.from_dict(d)
    assert restored.signature == o.signature
    assert restored.outcome == o.outcome
    assert restored.latency_ms == o.latency_ms
