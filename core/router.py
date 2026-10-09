"""Deterministic local routing; profile memory/latency ranks are estimates."""

from typing import Literal

from pydantic import BaseModel, Field

TaskType = Literal["extraction", "classification", "hypothesis_generation", "verification", "final_diagnosis"]
MemoryScope = Literal["host", "cuda", "xpu"]


class ModelProfile(BaseModel):
    id: str = Field(min_length=1)
    backend: Literal["deterministic-mock", "transformers-local"]
    model_name: str
    model_path: str | None = None
    revision: str | None = None
    tier: Literal["small_model", "reasoning_model"]
    task_types: list[TaskType] = Field(default_factory=lambda: ["hypothesis_generation"])
    max_context_chars: int = Field(gt=0)
    estimated_memory_mb: int = Field(ge=0)
    latency_rank: int = Field(default=1, ge=0)
    quality_rank: int = Field(default=1, ge=0)
    device: Literal["cpu", "cuda", "xpu"] = "cpu"
    execution_mode: Literal["mock", "local", "stub"] = "local"
    enabled: bool = True
    max_new_tokens: int = Field(default=2048, gt=0)


class RoutingContext(BaseModel):
    task_type: TaskType
    context_chars: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    complexity_score: float = Field(ge=0, le=1)
    available_memory_mb: int | None = Field(default=None, ge=0)
    memory_scope: MemoryScope = "host"
    latency_preference: Literal["fast", "balanced", "quality"] = "balanced"
    execution_mode: Literal["mock", "local", "stub"] = "mock"


class RoutingDecision(BaseModel):
    context: RoutingContext
    preferred_tier: Literal["small_model", "reasoning_model"]
    selected_profile: ModelProfile
    reason: str
    rejected_profiles: dict[str, str] = Field(default_factory=dict)


def route(context: RoutingContext, profiles: list[ModelProfile]) -> RoutingDecision:
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("Model profile IDs must be unique")
    preferred = "reasoning_model" if (
        context.task_type in {"hypothesis_generation", "verification", "final_diagnosis"}
        or context.complexity_score >= 0.6 or context.context_chars > 20000
    ) else "small_model"
    if context.latency_preference == "fast" and context.complexity_score < 0.6:
        preferred = "small_model"
    eligible, rejected = [], {}
    for profile in profiles:
        scope = "host" if profile.device == "cpu" else profile.device
        if not profile.enabled:
            rejected[profile.id] = "Profile disabled (e.g. unvalidated target hardware)."
        elif profile.execution_mode != context.execution_mode:
            rejected[profile.id] = "Execution mode differs; no implicit mock/local substitution."
        elif context.task_type not in profile.task_types:
            rejected[profile.id] = "Task type unsupported."
        elif context.context_chars > profile.max_context_chars:
            rejected[profile.id] = "Input exceeds configured character context limit."
        elif (scope == context.memory_scope and context.available_memory_mb is not None
              and profile.estimated_memory_mb > context.available_memory_mb):
            rejected[profile.id] = "Configured memory estimate exceeds available memory in the same memory scope."
        else:
            eligible.append(profile)
    if not eligible:
        raise ValueError(f"No eligible model profile: {rejected}")

    def priority(profile):
        tier_penalty = int(profile.tier != preferred)
        if context.latency_preference == "fast":
            return (profile.latency_rank, tier_penalty, profile.estimated_memory_mb, profile.id)
        if context.latency_preference == "quality":
            return (tier_penalty, -profile.quality_rank, profile.latency_rank, profile.id)
        return (tier_penalty, profile.latency_rank, -profile.quality_rank, profile.id)

    selected = min(eligible, key=priority)
    scope = "host" if selected.device == "cpu" else selected.device
    budget = (f"{context.available_memory_mb} MiB available {scope} memory"
              if scope == context.memory_scope and context.available_memory_mb is not None
              else f"{scope} memory budget unavailable; runtime allocation remains authoritative")
    reason = (f"{context.task_type}: {context.context_chars} input characters, complexity {context.complexity_score:.2f}, "
              f"{context.latency_preference} preference. Preferred {preferred}; selected {selected.id} "
              f"({selected.tier}) on {selected.device}; {budget}. Configured footprint estimate "
              f"{selected.estimated_memory_mb} MiB and latency rank {selected.latency_rank} are profile estimates, not benchmarks.")
    if selected.tier != preferred:
        reason += " Preferred tier unavailable or outranked by latency preference; using the eligible actual model."
    return RoutingDecision(context=context, preferred_tier=preferred, selected_profile=selected,
                           reason=reason, rejected_profiles=rejected)
