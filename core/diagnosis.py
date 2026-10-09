"""Evidence -> candidates -> criteria verification -> ranking -> final result."""

from collections.abc import Callable

from core.confidence import score_verified_hypothesis
from core.evidence_repository import EvidenceRepository
from core.hypothesis import generate_hypotheses, validate_claims
from core.schemas import DiagnosisResult, EvidenceRelation, GroundedClaim
from core.verifier import verify_hypothesis
from inference.base import InferenceBackend


def diagnose(repository: EvidenceRepository, relations: list[EvidenceRelation],
             backend: InferenceBackend, *, progress: Callable[[str], None] | None = None) -> DiagnosisResult:
    if progress:
        progress("Generating hypotheses")
    candidates, attempts = generate_hypotheses(repository, relations, backend)
    if progress:
        progress("Verifying and scoring hypotheses")
    verified = []
    for candidate in candidates:
        hypothesis = verify_hypothesis(candidate, repository, relations)
        metadata = score_verified_hypothesis(hypothesis, repository, relations)
        verified.append(hypothesis.model_copy(update={"confidence": metadata.score, "confidence_metadata": metadata}))
    if progress:
        progress("Ranking root causes")
    ranked = sorted(verified, key=lambda h: (-h.confidence, h.kind, h.id))
    top = ranked[0]
    support = [repository.get(key) for key in top.supporting_evidence_ids]
    contradiction = [repository.get(key) for key in top.contradicting_evidence_ids]
    factual_claims = [GroundedClaim(statement=e.statement, evidence_ids=[e.id]) for e in [*support, *contradiction]]
    validate_claims(factual_claims, repository)
    discriminatory_ids = {c.id for c in top.verification_plan if c.discriminating}
    has_support = top.confidence > 0 and any(r.status in {"supported", "mixed"}
                                           and r.criterion_id in discriminatory_ids for r in top.verification_results)
    conclusion = ("Best-supported explanation; causality not proven." if has_support
                  else "Insufficient evidence to identify a root cause; ordering is a deterministic tie-break only.")
    return DiagnosisResult(
        incident_summary=conclusion, hypotheses=ranked, top_root_cause_id=top.id if has_support else "",
        primary_root_cause=top if has_support else None, alternative_hypotheses=ranked[1:] if has_support else ranked,
        confidence=top.confidence if has_support else 0,
        supporting_evidence=support, contradicting_evidence=contradiction,
        verification_results=top.verification_results,
        recommended_action=top.recommended_action if has_support else "Collect incident evidence before selecting a remediation.",
        selected_model=backend.model_name, inference_backend=backend.name,
        execution_mode="mock" if backend.is_mock else "local", generation_attempts=attempts,
        factual_claims=factual_claims, evidence_grounding_rate=1.0 if factual_claims else 0.0,
        conclusion_status=conclusion,
    )
