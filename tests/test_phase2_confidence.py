import pytest

from core.confidence import score_verified_hypothesis
from core.hypothesis import generate_hypotheses
from core.schemas import EvidenceRelation
from core.verifier import verify_hypothesis
from inference.mock_backend import MockBackend
from test_correlation import modified, run


def scored(graph, kind="duplicate_models"):
    candidates, _ = generate_hypotheses(*graph, MockBackend())
    verified = verify_hypothesis(next(h for h in candidates if h.kind == kind), *graph)
    return verified, score_verified_hypothesis(verified, *graph)


def test_verified_score_components_and_missing_evidence_coverage(demo_graph):
    _, score = scored(demo_graph)
    assert score.support_ratio == score.verification_strength == score.evidence_coverage == pytest.approx(5 / 6)
    assert score.source_diversity == 1
    assert score.contradiction_penalty == 0
    assert score.score == 0.866667
    assert "not a probability" in score.caveat


def test_contradictions_and_missing_data_reduce_authoritative_score(demo_graph):
    _, original = scored(demo_graph)
    _, contradiction = scored(run(modified("config.yaml", "gpu_memory_limit_gb: 32", "gpu_memory_limit_gb: 48")))
    assert contradiction.score < original.score
    assert contradiction.contradiction_penalty == 0.2
    _, missing = scored(run(modified("system.log", "gpu_memory_used_gib=19.0", "gpu_memory_used_gib=18.0")))
    assert missing.score < original.score


def test_repeated_evidence_does_not_increase_score(demo_graph):
    verified, original = scored(demo_graph)
    duplicated_results = [r.model_copy(update={"supporting_evidence_ids": r.supporting_evidence_ids * 5})
                          for r in verified.verification_results]
    duplicated = verified.model_copy(update={"verification_results": duplicated_results, "confidence": 1})
    assert score_verified_hypothesis(duplicated, *demo_graph) == original


def test_no_results_or_inconsistent_status_cannot_be_scored(demo_graph):
    verified, _ = scored(demo_graph)
    with pytest.raises(ValueError, match="one verification"):
        score_verified_hypothesis(verified.model_copy(update={"verification_results": []}), *demo_graph)
    results = [r.model_copy(deep=True) for r in verified.verification_results]
    results[-1].status = "supported"
    with pytest.raises(ValueError, match="inconsistent"):
        score_verified_hypothesis(verified.model_copy(update={"verification_results": results}), *demo_graph)


def test_derived_facts_require_acyclic_complete_provenance(demo_graph):
    verified, _ = scored(demo_graph)
    repository, relations = demo_graph
    with pytest.raises(ValueError, match="provenance"):
        score_verified_hypothesis(verified, repository, [r for r in relations if r.relation != "derived_from"])
    loop = EvidenceRelation(source_evidence_id="derived:exceeds_capacity", relation="derived_from",
                            target_evidence_id="derived:exceeds_capacity", explanation="invalid loop")
    with pytest.raises(ValueError, match="Cyclic"):
        score_verified_hypothesis(verified, repository, [*relations, loop])
