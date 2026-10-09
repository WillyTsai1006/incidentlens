"""Shared artifacts, with observations kept separate from later hypotheses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceReference(BaseModel):
    file_path: str = Field(min_length=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    raw_excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_lines(self) -> "SourceReference":
        if self.line_end is not None:
            if self.line_start is None or self.line_end < self.line_start:
                raise ValueError("line_end requires line_start and must not precede it")
        return self


class Evidence(BaseModel):
    id: str = Field(min_length=1)
    category: Literal[
        "log_event", "config_value", "code_behavior", "system_metric", "derived_fact"
    ]
    statement: str = Field(min_length=1)
    value: str | int | float | bool | None = None
    unit: str | None = None
    timestamp: str | None = None
    source: SourceReference
    confidence: float = Field(default=1.0, ge=0, le=1)


class EvidenceRelation(BaseModel):
    source_evidence_id: str
    relation: Literal[
        "causes", "supports", "contradicts", "precedes", "correlates_with", "derived_from"
    ]
    target_evidence_id: str
    explanation: str


HypothesisKind = Literal["duplicate_models", "runtime_allocation", "memory_leak"]


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(min_length=1)
    # Model factual claims are exact quotations of one normalized evidence item.
    evidence_ids: list[str] = Field(min_length=1, max_length=1)


class CandidateHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    kind: HypothesisKind
    claims: list[GroundedClaim] = Field(default_factory=list)


class CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[CandidateHypothesis] = Field(min_length=3, max_length=3)


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(default_factory=list)


class VerificationCriterion(BaseModel):
    id: str
    supporting_observation: str
    contradicting_observation: str
    weight: float = Field(default=1, gt=0, le=1)
    contradiction_penalty: float = Field(default=0.1, ge=0, le=1)
    discriminating: bool = True


class VerificationResult(BaseModel):
    criterion_id: str
    status: Literal["supported", "contradicted", "mixed", "missing"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str


class ConfidenceMetadata(BaseModel):
    method: str = "verified-evidence-heuristic-v1"
    support_ratio: float = Field(default=0, ge=0, le=1)
    source_diversity: float = Field(default=0, ge=0, le=1)
    verification_strength: float = Field(default=0, ge=0, le=1)
    evidence_coverage: float = Field(default=0, ge=0, le=1)
    contradiction_penalty: float = Field(default=0, ge=0)
    score: float = Field(default=0, ge=0, le=1)
    caveat: str = "Heuristic evidence score, not a probability or proof of causality."


class Hypothesis(BaseModel):
    id: str
    title: str
    description: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    verification_action: str
    confidence: float = Field(default=0, ge=0, le=1)
    kind: HypothesisKind | None = None
    claims: list[GroundedClaim] = Field(default_factory=list)
    verification_plan: list[VerificationCriterion] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    confidence_metadata: ConfidenceMetadata = Field(default_factory=ConfidenceMetadata)
    recommended_action: str = ""


class DiagnosisResult(BaseModel):
    incident_summary: str
    hypotheses: list[Hypothesis]
    top_root_cause_id: str
    recommended_action: str
    selected_model: str
    inference_backend: str
    latency_ms: float | None = Field(default=None, ge=0)
    memory_mb: float | None = Field(default=None, ge=0)
    primary_root_cause: Hypothesis | None = None
    alternative_hypotheses: list[Hypothesis] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    factual_claims: list[GroundedClaim] = Field(default_factory=list)
    evidence_grounding_rate: float = Field(default=0, ge=0, le=1)
    generation_attempts: int = Field(default=1, ge=1)
    execution_mode: Literal["mock", "local"] = "mock"
    conclusion_status: str = "Best-supported explanation; causality not proven."
