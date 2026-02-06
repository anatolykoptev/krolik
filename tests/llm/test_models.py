"""Tests for the dynamic model registry."""

import json
import pytest
from pathlib import Path

from krolik.llm.models import (
    MODELS,
    Capability,
    ModelRegistry,
    ModelSpec,
    Tier,
    classify_tier_by_cost,
    calculate_priority,
    _detect_capabilities,
)


def test_registry_has_default_models():
    """Registry should have at least the hardcoded defaults."""
    assert MODELS.count() >= 3  # cliproxy-flash, cliproxy-sonnet, perplexity
    assert len(MODELS.list_by_tier(Tier.FREE)) > 0
    assert len(MODELS.list_by_tier(Tier.RESEARCH)) > 0


def test_lookup_by_id():
    spec = MODELS.get("cliproxy/gemini-2.5-flash")
    assert spec is not None
    assert spec.provider == "cliproxy"
    assert spec.tier == Tier.FREE
    assert Capability.CHAT in spec.capabilities


def test_lookup_by_alias():
    spec = MODELS.get("cliproxy-flash")
    assert spec is not None
    assert spec.id == "cliproxy/gemini-2.5-flash"


def test_lookup_case_insensitive_alias():
    spec = MODELS.get("CLIPROXY-FLASH")
    assert spec is not None


def test_lookup_missing():
    assert MODELS.get("nonexistent/model-xyz") is None


def test_list_by_capability():
    code_models = MODELS.list_by_capability(Capability.CODE)
    assert len(code_models) > 0
    for m in code_models:
        assert Capability.CODE in m.capabilities


def test_list_by_provider():
    cliproxy_models = MODELS.list_by_provider("cliproxy")
    assert len(cliproxy_models) > 0
    for m in cliproxy_models:
        assert m.provider == "cliproxy"


def test_cheapest():
    cheapest = MODELS.cheapest()
    assert cheapest is not None
    assert cheapest.is_free


def test_cheapest_with_capability():
    cheapest_code = MODELS.cheapest(Capability.CODE)
    assert cheapest_code is not None
    assert Capability.CODE in cheapest_code.capabilities


def test_fastest():
    fastest = MODELS.fastest()
    assert fastest is not None
    assert fastest.speed >= 3


def test_free_models_have_zero_cost():
    for m in MODELS.list_by_tier(Tier.FREE):
        assert m.is_free, f"{m.id} should be free"


# ── Tier classification ──────────────────────────────────────────


def test_classify_tier_free():
    assert classify_tier_by_cost(0) == Tier.FREE


def test_classify_tier_cheap():
    assert classify_tier_by_cost(0.03) == Tier.CHEAP


def test_classify_tier_standard():
    assert classify_tier_by_cost(0.10) == Tier.STANDARD


def test_classify_tier_premium():
    assert classify_tier_by_cost(0.20) == Tier.PREMIUM


# ── Priority calculation ─────────────────────────────────────────


def test_priority_free_model_gets_bonus():
    p = calculate_priority("test/model", cost=0, context_length=128_000)
    assert p > 50  # Should get bonuses for free + long context


def test_priority_expensive_model():
    p = calculate_priority("test/model", cost=10.0, context_length=4_096)
    assert p == 50  # No bonuses


def test_priority_clamped():
    p = calculate_priority("claude/model", cost=0, context_length=1_000_000, created_ts=9999999999.0)
    assert 0 <= p <= 100


# ── Capability detection ─────────────────────────────────────────


def test_detect_code_capability():
    caps = _detect_capabilities("deepseek/deepseek-coder-v2")
    assert Capability.CODE in caps


def test_detect_search_capability():
    caps = _detect_capabilities("perplexity/sonar-pro")
    assert Capability.SEARCH in caps


def test_detect_claude_capabilities():
    caps = _detect_capabilities("anthropic/claude-sonnet-4")
    assert Capability.CODE in caps
    assert Capability.REASONING in caps


# ── Serialization ────────────────────────────────────────────────


def test_model_spec_roundtrip():
    spec = ModelSpec(
        id="test/model",
        provider="test",
        tier=Tier.CHEAP,
        context_window=32_000,
        cost_per_1m_input=0.03,
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        aliases=("myalias",),
        source="test",
    )
    d = spec.to_dict()
    restored = ModelSpec.from_dict(d)
    assert restored.id == spec.id
    assert restored.tier == spec.tier
    assert restored.cost_per_1m_input == spec.cost_per_1m_input
    assert restored.capabilities == spec.capabilities


# ── Cache persistence ────────────────────────────────────────────


def test_cache_save_load(tmp_path):
    cache_path = tmp_path / "models.json"
    reg = ModelRegistry(cache_path=cache_path)
    reg.register(ModelSpec(
        id="test/cached-model",
        provider="test",
        tier=Tier.CHEAP,
        capabilities=frozenset({Capability.CHAT}),
        source="test",
    ))
    reg.save_cache()

    assert cache_path.exists()
    data = json.loads(cache_path.read_text())
    assert data["version"] == 2
    assert any(m["id"] == "test/cached-model" for m in data["models"])

    # Load into new registry
    reg2 = ModelRegistry(cache_path=cache_path)
    assert reg2.get("test/cached-model") is not None


def test_cache_handles_missing_file(tmp_path):
    cache_path = tmp_path / "nonexistent" / "models.json"
    reg = ModelRegistry(cache_path=cache_path)
    assert reg.count() >= 3  # Still has defaults


def test_needs_discovery_initially(tmp_path):
    reg = ModelRegistry(cache_path=tmp_path / "models.json")
    assert reg.needs_discovery


def test_unregister(tmp_path):
    reg = ModelRegistry(cache_path=tmp_path / "models.json")
    reg.register(ModelSpec(
        id="test/to-remove", provider="test", tier=Tier.FREE,
        capabilities=frozenset({Capability.CHAT}),
    ))
    assert reg.get("test/to-remove") is not None
    reg.unregister("test/to-remove")
    assert reg.get("test/to-remove") is None


def test_inactive_model_not_registered(tmp_path):
    reg = ModelRegistry(cache_path=tmp_path / "models.json")
    reg.register(ModelSpec(
        id="test/inactive", provider="test", tier=Tier.FREE,
        capabilities=frozenset({Capability.CHAT}),
        status="disabled",
    ))
    assert reg.get("test/inactive") is None
