import pytest
from pydantic import ValidationError

from core.router import ModelProfile, RoutingContext, route


def profile(key="small", **updates):
    data = dict(id=key, backend="transformers-local", model_name=key, model_path="models/local",
                tier="small_model", max_context_chars=30000, estimated_memory_mb=1000,
                task_types=["classification", "hypothesis_generation"])
    return ModelProfile(**(data | updates))


def context(**updates):
    data = dict(task_type="hypothesis_generation", context_chars=1000, evidence_count=25,
                complexity_score=0.8, available_memory_mb=8000, execution_mode="local")
    return RoutingContext(**(data | updates))


def test_task_and_complexity_choose_reasoning_tier():
    profiles = [profile(), profile("reasoning", tier="reasoning_model", estimated_memory_mb=4000)]
    assert route(context(), profiles).selected_profile.id == "reasoning"
    assert route(context(task_type="classification", complexity_score=0.1), profiles).selected_profile.id == "small"
    assert route(context(task_type="classification", complexity_score=0.9), profiles).selected_profile.id == "reasoning"


def test_long_context_and_memory_filter_profiles():
    profiles = [profile(max_context_chars=1000), profile("reasoning", tier="reasoning_model", estimated_memory_mb=4000)]
    decision = route(context(context_chars=2000), profiles)
    assert decision.selected_profile.id == "reasoning"
    assert "context limit" in decision.rejected_profiles["small"]
    decision = route(context(available_memory_mb=2000), [profile(), profiles[1]])
    assert decision.selected_profile.id == "small"
    assert "memory estimate" in decision.rejected_profiles["reasoning"]
    assert "Preferred tier unavailable" in decision.reason


def test_latency_and_quality_preferences_are_inspectable():
    profiles = [profile("quick", latency_rank=0), profile("quality", tier="reasoning_model", quality_rank=5, latency_rank=5)]
    assert route(context(latency_preference="fast"), profiles).selected_profile.id == "quick"
    assert route(context(latency_preference="quality"), profiles).selected_profile.id == "quality"
    assert "estimates, not benchmarks" in route(context(), profiles).reason


def test_unknown_memory_never_becomes_fake_budget():
    decision = route(context(available_memory_mb=None), [profile()])
    assert decision.context.available_memory_mb is None
    assert "memory budget unavailable" in decision.reason


def test_host_ram_is_not_used_as_gpu_memory():
    decision = route(context(available_memory_mb=1, memory_scope="host"), [profile(device="xpu")])
    assert "xpu memory budget unavailable" in decision.reason


def test_mode_task_disabled_and_no_implicit_mock_fallback():
    with pytest.raises(ValueError, match="No eligible"):
        route(context(), [profile(execution_mode="mock")])
    with pytest.raises(ValueError, match="No eligible"):
        route(context(task_type="extraction"), [profile()])
    with pytest.raises(ValueError, match="disabled"):
        route(context(), [profile(enabled=False)])


def test_decisions_and_ties_are_deterministic():
    profiles = [profile("b"), profile("a")]
    first = route(context(), profiles)
    assert first == route(context(), list(reversed(profiles)))
    assert first.selected_profile.id == "a"
    assert first == type(first).model_validate_json(first.model_dump_json())
    with pytest.raises(ValueError, match="unique"):
        route(context(), [profile(), profile()])


@pytest.mark.parametrize("updates", [{"context_chars": -1}, {"complexity_score": 1.1}, {"available_memory_mb": -1}])
def test_invalid_context_rejected(updates):
    with pytest.raises(ValidationError):
        context(**updates)
