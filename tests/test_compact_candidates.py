import json

import pytest

from core.hypothesis import HypothesisGenerationError, generate_hypotheses
from inference.mock_backend import MockBackend


class CompactBackend(MockBackend):
    candidate_format = "per_candidate_ids"

    def __init__(self, mutation=None):
        self.mutation = mutation

    def generate(self, prompt, **kwargs):
        payload = json.loads(prompt)
        assert "relations" in payload and "candidate" in payload
        assert all("source" not in e for e in payload["evidence"])
        choices = dict(duplicate_models="derived:model_replicas", runtime_allocation="app.log:L9:oom", memory_leak="system.log:L4:memory")
        response = {"evidence_ids": [choices[payload["candidate"]["kind"]]]}
        if self.mutation:
            self.mutation(response)
        return json.dumps(response)


def test_compact_ids_expand_to_exact_grounded_claims(demo_graph):
    hypotheses, attempts = generate_hypotheses(*demo_graph, CompactBackend())
    assert len(hypotheses) == 3 and attempts == 3
    for h in hypotheses:
        assert h.claims and h.confidence == 0
        for claim in h.claims:
            assert claim.statement == demo_graph[0].get(claim.evidence_ids[0]).statement


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(evidence_ids=["unknown"]),
    lambda r: r.update(confidence=0.99),
    lambda r: r.update(description="Unsupported factual text"),
    lambda r: r.update(kind="derived_fact"),
    lambda r: r.update(evidence_ids="uncited text"),
    lambda r: r.update(evidence_ids=[{"statement": "fabricated"}]),
])
def test_compact_output_keeps_strict_schema_and_id_validation(demo_graph, mutation):
    with pytest.raises(HypothesisGenerationError):
        generate_hypotheses(*demo_graph, CompactBackend(mutation))
