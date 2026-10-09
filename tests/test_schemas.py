import pytest
from pydantic import ValidationError

from core.schemas import Evidence, Hypothesis, SourceReference


def test_source_line_validation():
    for fields in ({"line_start": 0}, {"line_start": 3, "line_end": 2}, {"line_end": 2}):
        with pytest.raises(ValidationError):
            SourceReference(file_path="app.log", raw_excerpt="event", **fields)


def test_evidence_roundtrip_and_confidence_bounds():
    item = Evidence(id="e1", category="log_event", statement="event",
                    source=SourceReference(file_path="app.log", raw_excerpt="event"))
    assert Evidence.model_validate_json(item.model_dump_json()) == item
    with pytest.raises(ValidationError):
        Evidence.model_validate({**item.model_dump(), "confidence": 1.1})


def test_hypothesis_lists_are_independent():
    fields = dict(id="h1", title="title", description="description", verification_action="inspect")
    first, second = Hypothesis(**fields), Hypothesis(**fields)
    first.supporting_evidence_ids.append("e1")
    assert second.supporting_evidence_ids == []
