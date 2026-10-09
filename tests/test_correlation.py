import re

import pytest

from core.correlation import correlate
from core.evidence_extractor import extract_evidence
from core.evidence_repository import EvidenceRepository
from core.incident_loader import load_incident
from test_loader import DEMO


def run(files):
    repository = EvidenceRepository(extract_evidence(files))
    derived, relations = correlate(repository)
    repository.extend(derived)
    repository.validate_relations(relations)
    return repository, relations


def modified(filename, old, new):
    return [file.model_copy(update={"content": file.content.replace(old, new)})
            if file.file_path.endswith(filename) else file for file in load_incident(DEMO)]


def test_expected_derived_values_and_relation_chain():
    repository, relations = run(load_incident(DEMO))
    assert repository.get("derived:model_replicas").value == 4
    assert repository.get("derived:per_worker_memory_gib").value == 9.5
    assert repository.get("derived:projected_memory_gib").value == 38.0
    assert repository.get("derived:projected_memory_gib").value > 32
    assert repository.get("system.log:L4:memory").value <= 32
    assert repository.get("config.yaml:runtime.gpu_memory_limit_gb").value == 32
    failure = repository.get("system.log:L5:oom")
    requested = float(re.search(r"requested_gib=([\d.]+)", failure.source.raw_excerpt).group(1))
    free = float(re.search(r"free_gib=([\d.]+)", failure.source.raw_excerpt).group(1))
    assert requested == repository.get("derived:per_worker_memory_gib").value
    assert free == 32 - repository.get("system.log:L4:memory").value
    assert requested > free
    assert repository.get("derived:exceeds_capacity").value is True
    edges = {(r.source_evidence_id, r.relation, r.target_evidence_id) for r in relations}
    assert ("config.yaml:service.workers", "supports", "derived:model_replicas") in edges
    assert ("inference.py:L7:worker_model_load", "supports", "derived:model_replicas") in edges
    assert ("derived:model_replicas", "correlates_with", "derived:per_worker_memory_gib") in edges
    assert ("system.log:L4:memory", "precedes", "app.log:L9:oom") in edges
    assert ("app.log:L9:oom", "precedes", "app.log:L10:service_failure") in edges
    assert all(r.relation != "causes" for r in relations)
    for item in repository.all():
        if item.category == "derived_fact":
            assert any(r.source_evidence_id == item.id and r.relation == "derived_from" for r in relations)


@pytest.mark.parametrize("filename,old,new", [
    ("config.yaml", "workers: 4", "workers: 1"),
    ("config.yaml", "workers: 4", "workers: true"),
    ("config.yaml", "workers: 4", "workers: 0"),
    ("config.yaml", "load_per_worker: true", "load_per_worker: false"),
    ("inference.py", "load_model(MODEL_NAME)", "get_shared_model(MODEL_NAME)"),
])
def test_no_replica_claim_without_required_inputs(filename, old, new):
    repository, _ = run(modified(filename, old, new))
    assert not any(e.id in {"derived:model_replicas", "derived:projected_memory_gib"} for e in repository.all())


@pytest.mark.parametrize("old,new", [
    ("gpu_memory_unit: GiB", "gpu_memory_unit: GB"),
    ("  gpu_memory_unit: GiB", ""),
    ("gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: 0"),
    ("gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: true"),
    ("gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: .inf"),
])
def test_missing_or_incompatible_capacity_does_not_fabricate_comparison(old, new):
    repository, _ = run(modified("config.yaml", old, new))
    assert not any(e.id == "derived:exceeds_capacity" for e in repository.all())


def test_higher_capacity_changes_comparison():
    repository, relations = run(modified("config.yaml", "gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: 48"))
    assert repository.get("derived:exceeds_capacity").value is False
    assert not any(r.source_evidence_id == "derived:exceeds_capacity" and r.relation == "supports" for r in relations)


def test_configured_worker_count_must_cover_observed_startups():
    repository, _ = run(modified("config.yaml", "workers: 4", "workers: 2"))
    assert not any(e.id == "derived:projected_memory_gib" for e in repository.all())


def test_worker_identity_match_does_not_confuse_worker_1_and_10():
    _, relations = run(modified("app.log", "worker-4 CUDA", "worker-10 CUDA"))
    assert not any(r.source_evidence_id == "app.log:L2:worker_start"
                   and r.target_evidence_id == "app.log:L9:oom" for r in relations)


@pytest.mark.parametrize("old,new", [
    ("2026-10-08T09:00:00Z INFO gpu_memory_used_gib=0.0\n", ""),
    ("gpu_memory_used_gib=19.0", "gpu_memory_used_gib=12.0"),
    ("gpu_memory_used_gib=28.5", "gpu_memory_used_gib=19.0"),
    ("worker-2 started", "worker-1 started"),
    ("worker-2 started", "sampling"),
    ("2026-10-08T09:00:04Z", "2026-10-08T09:00:01Z"),
    ("2026-10-08T09:00:04Z", ""),
])
def test_incomplete_or_non_stepwise_metrics_do_not_imply_per_worker_growth(old, new):
    repository, _ = run(modified("system.log", old, new))
    assert not any(e.id == "derived:per_worker_memory_gib" for e in repository.all())


def test_no_memory_data_no_memory_derived_facts():
    files = [file for file in load_incident(DEMO) if not file.file_path.endswith("system.log")]
    repository, _ = run(files)
    assert [e.id for e in repository.all() if e.category == "derived_fact"] == ["derived:model_replicas"]


def test_temporal_relationships_require_ordering():
    _, relations = run(modified("app.log", "2026-10-08T09:00:09Z", "2026-10-08T09:00:01Z"))
    assert not any(r.target_evidence_id == "app.log:L10:service_failure" for r in relations)


def test_timezone_offsets_compared_as_instants():
    _, relations = run(modified("app.log", "2026-10-08T09:00:09Z", "2026-10-08T17:00:09+08:00"))
    assert any(r.target_evidence_id == "app.log:L10:service_failure" for r in relations)


def test_empty_evidence_has_no_relations_or_derived_facts():
    assert correlate(EvidenceRepository()) == ([], [])


def test_input_order_does_not_change_correlations():
    files = load_incident(DEMO)
    first, second = run(files), run(list(reversed(files)))
    assert {e.id: e.value for e in first[0].all()} == {e.id: e.value for e in second[0].all()}
    assert {r.model_dump_json() for r in first[1]} == {r.model_dump_json() for r in second[1]}
