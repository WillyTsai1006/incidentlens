import json

import pytest

from core.evidence_repository import EvidenceRepository
from core.schemas import Evidence, EvidenceRelation, SourceReference


def item(key):
    return Evidence(id=key, category="log_event", statement="event",
                    source=SourceReference(file_path="app.log", raw_excerpt="event"))


def test_repository_lookup_and_json_roundtrip():
    first, second = item("e1"), item("e2")
    repository = EvidenceRepository([first])
    repository.extend([second])
    assert repository.get("e1") == first
    assert [Evidence.model_validate(data) for data in json.loads(repository.to_json())] == [first, second]
    with pytest.raises(KeyError):
        repository.get("missing")


def test_duplicate_rejection_is_atomic():
    repository = EvidenceRepository([item("e1")])
    with pytest.raises(ValueError, match="Duplicate"):
        repository.extend([item("e2"), item("e1")])
    assert len(repository.all()) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        EvidenceRepository([item("e1"), item("e1")])


def test_relation_endpoints_must_exist():
    repository = EvidenceRepository([item("e1"), item("e2")])
    fields = dict(source_evidence_id="e1", relation="precedes", explanation="earlier")
    repository.validate_relations([EvidenceRelation(**fields, target_evidence_id="e2")])
    with pytest.raises(ValueError, match="Unknown"):
        repository.validate_relations([EvidenceRelation(**fields, target_evidence_id="missing")])
