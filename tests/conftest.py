from pathlib import Path

import pytest

from core.correlation import correlate
from core.evidence_extractor import extract_evidence
from core.evidence_repository import EvidenceRepository
from core.incident_loader import load_incident


@pytest.fixture
def demo_graph():
    directory = Path(__file__).resolve().parents[1] / "examples/gpu_oom_duplicate_models"
    repository = EvidenceRepository(extract_evidence(load_incident(directory)))
    derived, relations = correlate(repository)
    repository.extend(derived)
    return repository, relations
