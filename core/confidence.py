"""Transparent heuristic scoring, not a calibrated probability or LLM score."""

from typing import Annotated

from pydantic import BaseModel, Field

from core.evidence_repository import EvidenceRepository
from core.schemas import ConfidenceMetadata, EvidenceRelation, Hypothesis


class ConfidenceInputs(BaseModel):
    supporting_slots: int = Field(ge=0)
    expected_support_slots: int = Field(gt=0)
    supporting_source_types: int = Field(ge=0)
    expected_source_types: int = Field(default=4, gt=0)
    causal_consistency: float = Field(ge=0, le=1)
    contradiction_weights: list[Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=list)


class ConfidenceScore(BaseModel):
    support_ratio: float
    source_diversity: float
    causal_consistency: float
    contradiction_penalty: float
    score: float


def score_confidence(inputs: ConfidenceInputs) -> ConfidenceScore:
    support = min(inputs.supporting_slots / inputs.expected_support_slots, 1.0)
    diversity = min(inputs.supporting_source_types / inputs.expected_source_types, 1.0)
    penalty = sum(inputs.contradiction_weights)
    raw = 0.55 * support + 0.25 * diversity + 0.20 * inputs.causal_consistency - penalty
    return ConfidenceScore(
        support_ratio=support, source_diversity=diversity,
        causal_consistency=inputs.causal_consistency, contradiction_penalty=penalty,
        score=round(max(0.0, min(1.0, raw)), 6),
    )


def score_verified_hypothesis(hypothesis: Hypothesis, repository: EvidenceRepository,
                              relations: list[EvidenceRelation]) -> ConfidenceMetadata:
    """Score evaluated criteria; derived facts add no independent source diversity."""
    repository.validate_relations(relations)
    criteria = {criterion.id: criterion for criterion in hypothesis.verification_plan}
    results = {result.criterion_id: result for result in hypothesis.verification_results}
    if (not criteria or len(criteria) != len(hypothesis.verification_plan)
            or len(results) != len(hypothesis.verification_results) or set(criteria) != set(results)):
        raise ValueError("Confidence requires one verification result per unique criterion")
    support_weight = verified_weight = covered_weight = penalty = 0.0
    support_ids = set()
    for key, criterion in criteria.items():
        result = results[key]
        for evidence_id in [*result.supporting_evidence_ids, *result.contradicting_evidence_ids]:
            repository.get(evidence_id)
        expected_status = ("mixed" if result.supporting_evidence_ids and result.contradicting_evidence_ids
                           else "supported" if result.supporting_evidence_ids
                           else "contradicted" if result.contradicting_evidence_ids else "missing")
        if result.status != expected_status:
            raise ValueError("Verification status is inconsistent with evidence references")
        factor = 0.5 if result.status == "mixed" else 1.0 if result.status == "supported" else 0.0
        support_weight += criterion.weight * factor
        if criterion.discriminating:
            verified_weight += criterion.weight * factor
        if result.status != "missing":
            covered_weight += criterion.weight
        if result.contradicting_evidence_ids:
            penalty += criterion.contradiction_penalty
        support_ids.update(result.supporting_evidence_ids)

    parents = {}
    for relation in relations:
        if relation.relation == "derived_from":
            parents.setdefault(relation.source_evidence_id, set()).add(relation.target_evidence_id)

    def originals(key, visiting):
        item = repository.get(key)
        if key in visiting:
            raise ValueError("Cyclic derivation provenance")
        if item.category != "derived_fact":
            return {key}
        if key not in parents:
            raise ValueError("Derived evidence is missing provenance")
        return set().union(*(originals(parent, visiting | {key}) for parent in parents[key]))

    leaves = set().union(*(originals(key, set()) for key in support_ids))
    metric_files = {e.source.file_path for e in repository.all() if e.category == "system_metric"}
    types = set()
    for key in leaves:
        item = repository.get(key)
        source_type = ("yaml" if item.category == "config_value" else "code" if item.category == "code_behavior"
                       else "system_log" if item.source.file_path in metric_files else "app_log")
        types.add(source_type)
    total = sum(c.weight for c in criteria.values())
    discriminating_total = sum(c.weight for c in criteria.values() if c.discriminating)
    support_ratio = support_weight / total
    diversity = min(len(types) / 4, 1)
    strength = verified_weight / discriminating_total if discriminating_total else 0
    coverage = covered_weight / total
    raw = 0.45 * support_ratio + 0.20 * diversity + 0.20 * strength + 0.15 * coverage - penalty
    return ConfidenceMetadata(support_ratio=support_ratio, source_diversity=diversity,
                              verification_strength=strength, evidence_coverage=coverage,
                              contradiction_penalty=penalty, score=round(max(0, min(1, raw)), 6))
