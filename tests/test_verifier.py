from pathlib import Path

import pytest

from core.hypothesis import generate_hypotheses
from core.incident_loader import IncidentFile
from core.schemas import GroundedClaim
from core.verifier import verify_hypothesis
from inference.mock_backend import MockBackend
from test_correlation import modified, run


def verify(graph, kind):
    candidates, _ = generate_hypotheses(*graph, MockBackend())
    return verify_hypothesis(next(h for h in candidates if h.kind == kind), *graph)


def results(hypothesis):
    return {result.criterion_id: result for result in hypothesis.verification_results}


def test_primary_verification_has_explicit_criteria_and_missing_process_confirmation(demo_graph):
    hypothesis = verify(demo_graph, "duplicate_models")
    assert len(hypothesis.verification_plan) == len(hypothesis.verification_results) == 6
    assert [r.status for r in hypothesis.verification_results] == ["supported"] * 5 + ["missing"]
    assert "Process-level" in hypothesis.missing_evidence[0]
    assert len({Path(demo_graph[0].get(key).source.file_path).suffix
                for key in hypothesis.supporting_evidence_ids}) >= 3
    assert all(c.supporting_observation and c.contradicting_observation for c in hypothesis.verification_plan)
    assert hypothesis.confidence == 0  # Scoring is a separate stage.


def test_stepwise_growth_weakens_leak_but_no_steady_state_is_not_negative_evidence(demo_graph):
    hypothesis = verify(demo_graph, "memory_leak")
    by_id = results(hypothesis)
    assert by_id["steady_growth"].status == "missing"
    assert by_id["leak_pattern"].status == "contradicted"
    assert "competing" in by_id["leak_pattern"].explanation
    assert hypothesis.contradicting_evidence_ids == ["derived:per_worker_memory_gib"]
    assert not any("no leak" in r.explanation.lower() for r in hypothesis.verification_results)


def test_batch_hypothesis_is_weakened_by_initialization_failure(demo_graph):
    hypothesis = verify(demo_graph, "runtime_allocation")
    by_id = results(hypothesis)
    assert by_id["batch_pressure"].status == "missing"
    assert by_id["runtime_timing"].status == "contradicted"
    assert by_id["runtime_timing"].contradicting_evidence_ids == ["app.log:L9:oom"]


def test_capacity_counterevidence_and_disabled_worker_local_loading():
    hypothesis = verify(run(modified("config.yaml", "gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: 48")), "duplicate_models")
    assert results(hypothesis)["capacity"].status == "contradicted"
    hypothesis = verify(run(modified("config.yaml", "load_per_worker: true", "load_per_worker: false")), "duplicate_models")
    assert results(hypothesis)["worker_model"].status == "contradicted"
    assert results(hypothesis)["capacity"].status == "missing"


def test_temporal_verification_uses_precedes_not_causal_edges(demo_graph):
    repository, relations = demo_graph
    causal_only = [r.model_copy(update={"relation": "causes"}) for r in relations if r.relation == "precedes"]
    hypothesis = verify((repository, causal_only), "duplicate_models")
    assert results(hypothesis)["startup_alignment"].status == "missing"


def test_verification_rejects_unknown_claims_and_kinds(demo_graph):
    candidates, _ = generate_hypotheses(*demo_graph, MockBackend())
    unknown = GroundedClaim(statement="event", evidence_ids=["unknown"])
    with pytest.raises(ValueError, match="unknown"):
        verify_hypothesis(candidates[0].model_copy(update={"claims": [unknown]}), *demo_graph)
    with pytest.raises(ValueError, match="criteria"):
        verify_hypothesis(candidates[0].model_copy(update={"kind": None}), *demo_graph)


def test_verifier_does_not_trust_candidate_support_or_confidence(demo_graph):
    candidates, _ = generate_hypotheses(*demo_graph, MockBackend())
    forged = candidates[0].model_copy(update={"supporting_evidence_ids": ["unknown"], "confidence": 1.0})
    checked = verify_hypothesis(forged, *demo_graph)
    assert "unknown" not in checked.supporting_evidence_ids
    assert checked.confidence == 0


@pytest.mark.parametrize("values,expected", [([10, 11, 12], "supported"), ([10, 10, 10], "contradicted")])
def test_steady_state_observations_evaluate_leak_criteria(values, expected):
    files = [IncidentFile(file_path="system.log", content="\n".join(
        f"2026-10-08T09:01:0{i}Z INFO phase=steady_state active_workers=3 gpu_memory_used_gib={value}"
        for i, value in enumerate(values)
    ))]
    hypothesis = verify(run(files), "memory_leak")
    assert results(hypothesis)["steady_growth"].status == expected
