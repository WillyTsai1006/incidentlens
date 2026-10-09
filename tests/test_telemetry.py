import pytest

from app.phase3 import default_profiles
from core.router import RoutingContext
from inference.mock_backend import MockBackend
from inference.routed_backend import RoutedBackend
from inference.telemetry import hardware_snapshot, process_rss_mb


def wrapper(factory=None, mode="mock"):
    profiles = default_profiles()
    profiles[0].execution_mode = mode
    context = RoutingContext(task_type="hypothesis_generation", context_chars=0,
                             evidence_count=25, complexity_score=0.8, execution_mode=mode)
    return RoutedBackend(profiles, context, factory) if factory else RoutedBackend(profiles, context)


def prompt():
    return '{"evidence": []}'


def test_mock_metrics_are_measured_and_explicitly_labeled():
    backend = wrapper()
    backend.generate(prompt())
    call = backend.calls[0]
    assert call.execution_mode == "mock" and "mock" in call.metric_label
    assert call.input_chars == len(prompt()) and call.output_chars > 0
    assert call.inference_latency_ms >= 0 and call.routing_overhead_ms >= 0
    assert call.input_tokens is call.output_tokens is call.output_tokens_per_second is None
    assert call.device_allocated_memory_mb is None
    assert backend.call_io[0]["prompt"] == prompt()


def test_backend_reused_and_every_call_including_retries_is_recorded():
    made = []

    def factory(profile):
        made.append(profile.id)
        return MockBackend()

    backend = wrapper(factory)
    backend.generate(prompt())
    backend.generate(prompt())
    assert made == ["mock-demo"]
    assert [c.call_index for c in backend.calls] == [1, 2]


def test_failed_calls_record_status_without_stale_metrics():
    class Broken(MockBackend):
        last_generation_metrics = dict(input_tokens=999, output_tokens=999)

        def generate(self, prompt, **kwargs):
            raise RuntimeError("unavailable")

    backend = wrapper(lambda p: Broken())
    with pytest.raises(RuntimeError):
        backend.generate(prompt())
    call = backend.calls[0]
    assert call.status == "failed" and call.error_type == "RuntimeError"
    assert call.output_chars == 0 and call.input_tokens is None


def test_stub_token_metrics_are_explicitly_labeled():
    class Stub(MockBackend):
        is_mock = False
        last_generation_metrics = dict(input_tokens=10, output_tokens=5, generation_latency_ms=250,
                                       device_allocated_memory_mb=None, cold_start=True)

    backend = wrapper(lambda p: Stub(), mode="stub")
    backend.generate(prompt())
    call = backend.calls[0]
    assert call.execution_mode == "stub" and "stub" in call.metric_label
    assert call.output_tokens_per_second == 20
    assert call.device_allocated_memory_mb is None


def test_unavailable_hardware_metrics_are_null(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    snapshot = hardware_snapshot("xpu")
    assert snapshot.host_available_memory_mb is None
    assert snapshot.device_available_memory_mb is None and snapshot.device_name is None
    assert process_rss_mb() is None


def test_actual_backend_mode_mismatch_rejected():
    class FalseLocal(MockBackend):
        is_mock = False

    backend = wrapper(lambda p: FalseLocal())
    with pytest.raises(ValueError, match="does not match"):
        backend.generate(prompt())
    assert backend.calls[0].status == "failed"
