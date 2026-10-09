"""Route and measure calls while preserving the Phase 2 generate protocol."""

from collections.abc import Callable
from time import perf_counter

from core.router import ModelProfile, RoutingContext, route
from inference.base import InferenceBackend
from inference.mock_backend import MockBackend
from inference.telemetry import InferenceCallTelemetry, process_rss_mb


def create_backend(profile: ModelProfile) -> InferenceBackend:
    if profile.backend == "deterministic-mock" and profile.execution_mode == "mock":
        return MockBackend()
    if profile.backend == "transformers-local" and profile.execution_mode == "local" and profile.model_path:
        from inference.transformers_backend import TransformersBackend

        return TransformersBackend(profile.model_path, device=profile.device, cpu_threads=4)
    raise ValueError("Profile has no executable backend binding; no stub or implicit fallback is allowed")


class RoutedBackend:
    def __init__(self, profiles: list[ModelProfile], context: RoutingContext,
                 factory: Callable[[ModelProfile], InferenceBackend] = create_backend):
        self.profiles, self.context, self.factory = profiles, context, factory
        self.calls: list[InferenceCallTelemetry] = []
        self.call_io: list[dict] = []
        self._backends: dict[str, InferenceBackend] = {}
        self._last_backend = None
        self.candidate_format = "per_candidate_ids" if context.execution_mode == "local" else "quoted_claims"

    @property
    def name(self):
        return self._last_backend.name if self._last_backend else "unselected"

    @property
    def model_name(self):
        return self._last_backend.model_name if self._last_backend else "unselected"

    @property
    def is_mock(self):
        return self._last_backend.is_mock if self._last_backend else self.context.execution_mode == "mock"

    def note_candidate_validation(self, valid: bool):
        status = "valid" if valid else "invalid"
        self.calls[-1].candidate_validation = status
        self.call_io[-1]["candidate_validation"] = status

    def generate(self, prompt: str, *, max_new_tokens: int = 2048, temperature: float = 0.0) -> str:
        context = self.context.model_copy(update={"context_chars": len(prompt)})
        route_start = perf_counter()
        decision = route(context, self.profiles)
        routing_ms = (perf_counter() - route_start) * 1000
        profile = decision.selected_profile
        max_new_tokens = min(max_new_tokens, profile.max_new_tokens)
        started = perf_counter()
        output, error_type = "", None
        backend = None
        try:
            if profile.id not in self._backends:
                self._backends[profile.id] = self.factory(profile)
            backend = self._backends[profile.id]
            if bool(backend.is_mock) != (profile.execution_mode == "mock"):
                raise ValueError("Actual backend execution mode does not match the routed profile")
            if backend.name != profile.backend:
                raise ValueError("Actual backend identity does not match the routed profile")
            self._last_backend = backend
            output = backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            return output
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            elapsed = (perf_counter() - started) * 1000
            measurements = getattr(backend, "last_generation_metrics", None) or {}
            if profile.execution_mode == "mock" or error_type:
                measurements = {}
            generation_ms = measurements.get("generation_latency_ms")
            output_tokens = measurements.get("output_tokens")
            rate = output_tokens / (generation_ms / 1000) if output_tokens is not None and generation_ms else None
            self.calls.append(InferenceCallTelemetry(
                call_index=len(self.calls) + 1, task_type=context.task_type,
                backend=backend.name if backend else profile.backend,
                model=backend.model_name if backend else profile.model_name,
                execution_mode=profile.execution_mode,
                metric_label=f"Measured {profile.execution_mode} call wall time; token counts unavailable" if not measurements
                             else f"Measured {profile.execution_mode} call with tokenizer counts",
                input_chars=len(prompt), output_chars=len(output),
                input_tokens=measurements.get("input_tokens"), output_tokens=output_tokens,
                inference_latency_ms=elapsed, generation_latency_ms=generation_ms,
                output_tokens_per_second=rate, process_rss_mb=process_rss_mb(),
                device_allocated_memory_mb=measurements.get("device_allocated_memory_mb"),
                cold_start=measurements.get("cold_start"), routing_overhead_ms=routing_ms,
                status="failed" if error_type else "completed", error_type=error_type, routing=decision,
            ))
            self.call_io.append(dict(profile_id=profile.id, prompt=prompt, output=output,
                                     max_new_tokens=max_new_tokens, temperature=temperature, error_type=error_type))
