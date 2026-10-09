import pytest
from pydantic import ValidationError

from core.confidence import ConfidenceInputs, score_confidence


def inputs(**changes):
    fields = dict(supporting_slots=3, expected_support_slots=4,
                  supporting_source_types=3, causal_consistency=1)
    return ConfidenceInputs(**(fields | changes))


def test_formula_and_auditable_components():
    result = score_confidence(inputs(contradiction_weights=[0.05]))
    assert result.support_ratio == 0.75
    assert result.source_diversity == 0.75
    assert result.contradiction_penalty == 0.05
    assert result.score == 0.75  # .55*.75 + .25*.75 + .20 - .05


def test_clamping_and_count_normalization():
    assert score_confidence(inputs(supporting_slots=100, supporting_source_types=100)).score == 1
    assert score_confidence(inputs(contradiction_weights=[1, 1])).score == 0
    assert score_confidence(inputs(supporting_slots=0, supporting_source_types=0, causal_consistency=0)).score == 0


def test_more_support_increases_score_and_contradictions_lower_it():
    baseline = score_confidence(inputs()).score
    assert score_confidence(inputs(supporting_slots=4)).score > baseline
    assert score_confidence(inputs(contradiction_weights=[0.1])).score < baseline


@pytest.mark.parametrize("changes", [
    {"expected_support_slots": 0}, {"expected_source_types": 0},
    {"supporting_slots": -1}, {"supporting_source_types": -1},
    {"causal_consistency": 1.1}, {"causal_consistency": float("nan")},
    {"contradiction_weights": [-0.1]}, {"contradiction_weights": [float("nan")]},
])
def test_invalid_scoring_inputs_rejected(changes):
    with pytest.raises(ValidationError):
        inputs(**changes)
