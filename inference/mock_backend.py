"""Deterministic candidate proposals, explicitly labeled as mock inference.

Reads only the evidence in the prompt, not expected diagnosis fixtures. This
adapter proposes competitors; verification and ranking are application work.
"""

import json


class MockBackend:
    name = "deterministic-mock"
    model_name = "mock-candidate-catalog-v1"
    is_mock = True

    def generate(self, prompt: str, *, max_new_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        payload = json.loads(prompt)
        evidence = payload["evidence"]
        hypotheses = []
        for kind in ("duplicate_models", "runtime_allocation", "memory_leak"):
            relevant = [item for item in evidence if (
                (kind == "duplicate_models" and item["category"] in {"code_behavior", "derived_fact"})
                or (kind != "duplicate_models" and item["id"].endswith(":oom"))
            )]
            hypotheses.append({
                "id": f"h-{kind}", "kind": kind,
                "claims": [{"statement": item["statement"], "evidence_ids": [item["id"]]}
                           for item in relevant],
            })
        return json.dumps({"hypotheses": hypotheses})
