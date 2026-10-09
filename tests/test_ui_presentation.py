import copy
import sys
from types import SimpleNamespace

import pytest

from app.phase3 import run_phase3
from app.ui_presenter import incident_overview, runtime_summary, supporting_summary, verification_status
from app.ui_presenter import display_path, presentation
from app.ui_runtime import DEMO_MODEL, detect_cuda, ui_profiles
from test_loader import DEMO


@pytest.fixture
def run():
    return run_phase3(DEMO, persist=False)


def test_overview_values_and_support_are_grounded_without_mutation(run):
    before = copy.deepcopy(run)
    overview = incident_overview(run["evidence"])
    assert [v["value"] for v in overview.values()] == ["32 GiB", "4", "9.5 GiB", "38 GiB"]
    summary = supporting_summary(run)
    assert {"derived:projected_memory_gib", "config.yaml:runtime.gpu_memory_limit_gb"} <= {e["id"] for e in summary}
    assert all(e["id"] in {n["id"] for n in run["evidence"]} for e in summary)
    assert before == run
    assert all(v["value"] == "N/A" for v in incident_overview([]).values())


@pytest.mark.parametrize("statuses,expected", [([], "INSUFFICIENT EVIDENCE"),
    (["missing"], "INSUFFICIENT EVIDENCE"), (["supported", "missing"], "PARTIALLY VERIFIED"),
    (["supported", "mixed"], "MIXED EVIDENCE"), (["contradicted"], "MIXED EVIDENCE"),
    (["supported"], "SUPPORTED BY AVAILABLE CHECKS")])
def test_verification_labels_do_not_claim_causality(statuses, expected):
    assert verification_status([dict(status=s) for s in statuses]) == expected


def test_mock_runtime_missing_metrics_stay_unavailable(run):
    summary = runtime_summary(run)
    assert summary["generation_s"] is summary["tokens_per_second"] is summary["gpu_peak_mb"] is None
    assert summary["gpu_snapshot_mb"] is None
    assert summary["end_to_end_s"] == run["telemetry"]["end_to_end_latency_ms"] / 1000


def test_runtime_aggregate_uses_current_calls_and_never_mislabels_peak(run):
    calls = run["telemetry"]["inference_calls"]
    calls[0].update(generation_latency_ms=1000, output_tokens=20, device_allocated_memory_mb=42)
    calls.append(copy.deepcopy(calls[0]))
    calls[1].update(call_index=2, generation_latency_ms=3000, output_tokens=80, device_allocated_memory_mb=45)
    summary = runtime_summary(run)
    assert summary["generation_s"] == 4 and summary["tokens_per_second"] == 25
    assert summary["gpu_snapshot_mb"] == 45 and summary["gpu_peak_mb"] is None


def test_cuda_defaults_detection_and_explicit_cpu_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda i: "GPU test identity")))
    assert detect_cuda()["name"] == "GPU test identity"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    result = detect_cuda()
    assert not result["available"] and "CPU" in result["warning"]


def test_cuda_demo_profile_without_changing_router_or_other_models(tmp_path):
    profile = ui_profiles("transformers", str(DEMO_MODEL), "cuda")[0]
    assert profile.device == "cuda" and profile.estimated_memory_mb == 4096
    assert profile.max_new_tokens == 512
    assert ui_profiles("transformers", str(tmp_path), "cuda")[0].estimated_memory_mb == 8192
    assert ui_profiles("mock", None, "cpu")[0].execution_mode == "mock"


@pytest.mark.parametrize("value, expected", [
    ("C:\\Users\\presenter\\private\\incident.log", "incident.log"),
    ("/home/presenter/private/incident.log", "incident.log"),
    ("runs\\demo-id", "runs/demo-id"),
    ("examples/incident.log", "examples/incident.log"),
])
def test_display_paths_hide_external_roots(value, expected):
    assert display_path(value) == expected


def test_presentation_paths_preserve_original_provenance(run):
    before = copy.deepcopy(run)
    view = presentation(run)
    assert all(str(DEMO.parent.parent) not in row["source"] for row in view["evidence"])
    assert display_path(str(DEMO_MODEL)) == "models/Qwen2.5-0.5B-Instruct"
    assert run == before


def test_runtime_summary_uses_selected_run_not_previous_measurements(run):
    previous = copy.deepcopy(run)
    previous["telemetry"]["end_to_end_latency_ms"] = 123456
    current = copy.deepcopy(run)
    current["telemetry"]["end_to_end_latency_ms"] = 2345
    assert runtime_summary(previous)["end_to_end_s"] == 123.456
    assert runtime_summary(current)["end_to_end_s"] == 2.345
