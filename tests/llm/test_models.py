"""Tests for the model registry."""

import pytest

from krolik.llm.models import MODELS, Capability, ModelRegistry, ModelSpec, Tier


def test_registry_has_models():
    """Registry should have models from all tiers."""
    assert len(MODELS.all()) > 0
    assert len(MODELS.list_by_tier(Tier.FREE)) > 0
    assert len(MODELS.list_by_tier(Tier.STANDARD)) > 0
    assert len(MODELS.list_by_tier(Tier.PREMIUM)) > 0
    assert len(MODELS.list_by_tier(Tier.RESEARCH)) > 0


def test_lookup_by_id():
    spec = MODELS.get("google/gemini-2.0-flash")
    assert spec is not None
    assert spec.provider == "gemini"
    assert spec.tier == Tier.FREE
    assert Capability.CHAT in spec.capabilities


def test_lookup_by_alias():
    spec = MODELS.get("flash")
    assert spec is not None
    assert spec.id == "google/gemini-2.0-flash"


def test_lookup_case_insensitive_alias():
    spec = MODELS.get("FLASH")
    assert spec is not None
    assert spec.id == "google/gemini-2.0-flash"


def test_lookup_missing():
    assert MODELS.get("nonexistent/model-xyz") is None


def test_list_by_capability():
    code_models = MODELS.list_by_capability(Capability.CODE)
    assert len(code_models) > 0
    for m in code_models:
        assert Capability.CODE in m.capabilities


def test_list_by_provider():
    gemini_models = MODELS.list_by_provider("gemini")
    assert len(gemini_models) > 0
    for m in gemini_models:
        assert m.provider == "gemini"


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


def test_custom_registry():
    """Users can create a custom registry."""
    reg = ModelRegistry()
    custom = ModelSpec(
        id="custom/my-model",
        provider="custom",
        tier=Tier.FREE,
        capabilities=frozenset({Capability.CHAT}),
        aliases=("my-model",),
    )
    reg.register(custom)
    assert reg.get("custom/my-model") is not None
    assert reg.get("my-model") is not None


def test_model_spec_is_frozen():
    spec = MODELS.get("flash")
    assert spec is not None
    with pytest.raises(AttributeError):
        spec.id = "hacked"


def test_free_models_have_zero_cost():
    for m in MODELS.list_by_tier(Tier.FREE):
        assert m.is_free, f"{m.id} should be free"


def test_premium_models_have_cost():
    for m in MODELS.list_by_tier(Tier.PREMIUM):
        assert m.cost_per_1k_input > 0, f"{m.id} should have cost"
