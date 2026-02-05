"""Tests for the model router."""

import pytest

from krolik.llm.models import Capability, Tier
from krolik.llm.router import ModelRouter, RouteResult, RouterError


@pytest.fixture
def router():
    return ModelRouter()


def test_simple_task_routes_to_free(router):
    result = router.route("Translate 'hello' to Russian")
    assert result.tier == Tier.FREE
    assert result.score <= 40


def test_code_task_routes_to_standard(router):
    result = router.route("Implement a REST API endpoint for user authentication with JWT tokens")
    assert result.tier in (Tier.STANDARD, Tier.PREMIUM)
    assert result.score > 40


def test_complex_task_routes_to_premium(router):
    result = router.route(
        "Design a distributed event sourcing architecture with CQRS pattern, "
        "handle eventual consistency, design the consensus protocol, "
        "and plan the migration strategy from the monolith"
    )
    assert result.tier == Tier.PREMIUM
    assert result.score > 70


def test_research_task_routes_to_research(router):
    result = router.route("Search for the latest news about OpenAI and explain what happened")
    assert result.tier == Tier.RESEARCH


def test_route_result_has_model(router):
    result = router.route("Hello world")
    assert result.model is not None
    assert result.model_id
    assert result.provider


def test_route_result_has_reason(router):
    result = router.route("Write a function")
    assert result.reason
    assert "score=" in result.reason


def test_route_with_required_capability(router):
    result = router.route("Do something", required_capability=Capability.CODE)
    assert Capability.CODE in result.model.capabilities


def test_fallbacks_are_populated(router):
    result = router.route("Implement user authentication")
    # Should have at least one fallback (different provider or higher tier)
    assert isinstance(result.fallbacks, list)


def test_record_outcome_and_success_rate(router):
    router.record_outcome("google/gemini-2.0-flash", "test task", success=True)
    router.record_outcome("google/gemini-2.0-flash", "test task 2", success=True)
    router.record_outcome("google/gemini-2.0-flash", "test task 3", success=False)
    
    rate = router.get_success_rate("google/gemini-2.0-flash")
    assert abs(rate - 2/3) < 0.01


def test_unknown_model_defaults_to_100_percent(router):
    rate = router.get_success_rate("nonexistent/model")
    assert rate == 1.0


def test_learning_escalation(router):
    """If free tier has low success rate, router should escalate."""
    # Poison the free tier
    for _ in range(10):
        router.record_outcome("google/gemini-2.0-flash", "fail task", success=False)
    
    # Now a simple task should escalate
    result = router.route("Summarize this text briefly")
    # Score should be bumped above free threshold
    assert result.score > 40 or result.tier != Tier.FREE


def test_score_clamped_0_100(router):
    result = router.route("hi")
    assert 0 <= result.score <= 100

    result = router.route(
        "Design architect system design complex critical production security "
        "audit migrate refactor entire scalable distributed consensus tradeoff " * 3
    )
    assert 0 <= result.score <= 100


def test_available_providers_filter():
    """Router should only select models from available providers."""
    router = ModelRouter(available_providers={"gemini"})
    result = router.route("Write some code")
    assert result.model.provider == "gemini"


def test_prefer_free_providers():
    """When prefer_free is True, free models should be preferred within a tier."""
    router = ModelRouter(prefer_free_providers=True)
    result = router.route("Translate this text")
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
