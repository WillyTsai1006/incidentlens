import json

import pytest

from core.hypothesis import HypothesisGenerationError, generate_hypotheses
from core.schemas import EvidenceRelation
from inference.mock_backend import MockBackend


class ScriptedBackend(MockBackend):
    def __init__(self, transform, *, first_only=False):
        self.transform = transform
        self.first_only = first_only
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(json.loads(prompt))
        assert kwargs["temperature"] == 0
        valid = super().generate(prompt, **kwargs)
        return valid if self.first_only and len(self.prompts) > 1 else self.transform(json.loads(valid))


def test_generator_consumes_graph_and_returns_candidates_without_final_answer(demo_graph):
    repository, relations = demo_graph
    backend = ScriptedBackend(json.dumps)
    hypotheses, attempts = generate_hypotheses(repository, relations, backend)
    assert attempts == 1 and len(hypotheses) == 3
    assert {h.kind for h in hypotheses} == {"duplicate_models", "runtime_allocation", "memory_leak"}
    assert backend.prompts[0]["relations"]
    assert len(backend.prompts[0]["evidence"]) == 25
    for hypothesis in hypotheses:
        assert hypothesis.title and hypothesis.description and hypothesis.verification_action
        assert hypothesis.confidence == 0 and hypothesis.confidence_metadata.score == 0
        assert hypothesis.supporting_evidence_ids == []  # Only the verifier assigns classifications.
        for claim in hypothesis.claims:
            assert claim.statement == repository.get(claim.evidence_ids[0]).statement


def changed(payload, field, value):
    payload["hypotheses"][0][field] = value
    return json.dumps(payload)


@pytest.mark.parametrize("transform", [
    lambda p: "not JSON",
    lambda p: json.dumps({**p, "final_answer": "definitely duplicate models"}),
    lambda p: changed(p, "confidence", 0.99),
    lambda p: changed(p, "supporting_evidence_ids", ["invented"]),
    lambda p: changed(p, "claims", [{"statement": "event", "evidence_ids": ["invented"]}]),
    lambda p: changed(p, "claims", [{"statement": "GPU hardware failed", "evidence_ids": ["config.yaml:service.workers"]}]),
    lambda p: changed(p, "claims", [{"statement": "uncited factual claim", "evidence_ids": []}]),
    lambda p: changed(p, "description", "The GPU definitely failed"),
    lambda p: changed(p, "kind", "unsupported_domain"),
    lambda p: changed(p, "id", p["hypotheses"][1]["id"]),
    lambda p: changed(p, "kind", "memory_leak"),
    lambda p: json.dumps({"hypotheses": p["hypotheses"][:1]}),
])
def test_invalid_or_ungrounded_generation_retries_then_fails(demo_graph, transform):
    backend = ScriptedBackend(transform)
    with pytest.raises(HypothesisGenerationError, match="after 2 attempts"):
        generate_hypotheses(*demo_graph, backend)
    assert len(backend.prompts) == 2
    assert "retry_instruction" in backend.prompts[1]


def test_retry_accepts_valid_grounded_response(demo_graph):
    backend = ScriptedBackend(lambda p: "broken JSON", first_only=True)
    hypotheses, attempts = generate_hypotheses(*demo_graph, backend)
    assert attempts == 2 and len(hypotheses) == 3


def test_json_fence_is_accepted_but_trailing_prose_is_not(demo_graph):
    backend = ScriptedBackend(lambda p: "```json\n" + json.dumps(p) + "\n```")
    assert len(generate_hypotheses(*demo_graph, backend)[0]) == 3
    backend = ScriptedBackend(lambda p: json.dumps(p) + "\nDefinitely proven.")
    with pytest.raises(HypothesisGenerationError):
        generate_hypotheses(*demo_graph, backend)


def test_unknown_graph_endpoint_is_rejected_before_inference(demo_graph):
    repository, relations = demo_graph
    backend = ScriptedBackend(json.dumps)
    bad = EvidenceRelation(source_evidence_id="invented", relation="supports",
                           target_evidence_id="derived:exceeds_capacity", explanation="bad")
    with pytest.raises(ValueError, match="Unknown"):
        generate_hypotheses(repository, [*relations, bad], backend)
    assert backend.prompts == []


def test_zero_attempt_budget_rejected(demo_graph):
    with pytest.raises(ValueError, match="positive"):
        generate_hypotheses(*demo_graph, MockBackend(), max_attempts=0)
