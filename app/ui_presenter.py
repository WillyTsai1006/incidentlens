"""UI-facing serialization independent of Streamlit and model packages."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.evidence_repository import EvidenceRepository
from core.hypothesis import validate_claims

from core.schemas import DiagnosisResult, Evidence, EvidenceRelation
from core.router import RoutingDecision
from inference.telemetry import RunTelemetry


def display_path(value: str) -> str:
    """Shorten paths for display only; original run provenance remains intact."""
    path = PureWindowsPath(value) if "\\" in value or PureWindowsPath(value).drive else PurePosixPath(value)
    if path.is_absolute():
        try:
            return path.relative_to(str(Path(__file__).resolve().parents[1])).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def presentation(run: dict) -> dict:
    diagnosis = DiagnosisResult.model_validate(run["diagnosis"])
    evidence = [Evidence.model_validate(e) for e in run["evidence"]]
    relations = [EvidenceRelation.model_validate(r) for r in run["relations"]]
    routing = [RoutingDecision.model_validate(r) for r in run["routing"]]
    telemetry = RunTelemetry.model_validate(run["telemetry"])
    known = {e.id for e in evidence}
    repository = EvidenceRepository(evidence)
    repository.validate_relations(relations)
    validate_claims(diagnosis.factual_claims, repository)
    for h in diagnosis.hypotheses:
        validate_claims(h.claims, repository)
        if not set(h.supporting_evidence_ids + h.contradicting_evidence_ids) <= known:
            raise ValueError("UI diagnosis references unknown evidence")
    primary = diagnosis.primary_root_cause
    return dict(
        mode=diagnosis.execution_mode, title=primary.title if primary else "Insufficient evidence",
        confidence=diagnosis.confidence, conclusion=diagnosis.conclusion_status,
        action=diagnosis.recommended_action, verification=primary.verification_action if primary else "Collect discriminating observations.",
        missing=primary.missing_evidence if primary else [],
        supporting=[e.model_dump(mode="json") for e in diagnosis.supporting_evidence],
        contradicting=[e.model_dump(mode="json") for e in diagnosis.contradicting_evidence],
        hypotheses=[dict(id=h.id, title=h.title, confidence=h.confidence,
                         coverage=h.confidence_metadata.evidence_coverage) for h in diagnosis.hypotheses],
        evidence=[dict(id=e.id, kind="Derived fact" if e.category == "derived_fact" else "Observed evidence",
                       category=e.category, statement=e.statement, value=json.dumps(e.value), unit=e.unit,
                       source=f"{display_path(e.source.file_path)}:{e.source.line_start}-{e.source.line_end}", excerpt=e.source.raw_excerpt) for e in evidence],
        relations=[r.model_dump(mode="json") for r in relations],
        routing=[r.model_dump(mode="json") for r in routing], telemetry=telemetry.model_dump(mode="json"),
        verification_results=[r.model_dump(mode="json") for r in diagnosis.verification_results],
    )


def format_value(value, unit=None) -> str:
    text = f"{value:g}" if isinstance(value, (float, int)) and not isinstance(value, bool) else str(value)
    return f"{text} {unit}" if unit else text


def incident_overview(evidence: list[dict]) -> dict:
    """Format extracted values; missing fields stay unknown."""
    items = [Evidence.model_validate(e) for e in evidence]
    def find(suffix):
        return next((e for e in items if e.id.endswith(suffix)), None)
    unit = find(":runtime.gpu_memory_unit")
    fields = {
        "GPU Capacity": (find(":runtime.gpu_memory_limit_gb"), unit.value if unit else None),
        "Worker Count": (find(":service.workers"), None),
        "Model Replica Size": (find("derived:per_worker_memory_gib"), "GiB"),
        "Projected Demand": (find("derived:projected_memory_gib"), "GiB"),
    }
    return {label: dict(value=format_value(e.value, units) if e else "N/A", evidence_id=e.id if e else None)
            for label, (e, units) in fields.items()}


def supporting_summary(run: dict) -> list[dict]:
    primary = run["diagnosis"]["primary_root_cause"]
    if primary is None:
        return []
    ids = set(primary["supporting_evidence_ids"])
    while True:
        parents = {r["target_evidence_id"] for r in run["relations"]
                   if r["relation"] == "derived_from" and r["source_evidence_id"] in ids}
        if parents <= ids:
            break
        ids |= parents
    evidence = [e for e in run["evidence"] if e["id"] in ids]
    priority = [":service.workers", ":model.load_per_worker", ":worker_model_load",
                "derived:projected_memory_gib", ":runtime.gpu_memory_limit_gb", ":oom"]
    ordered = []
    for suffix in priority:
        item = next((e for e in evidence if e["id"].endswith(suffix)), None)
        if item and item not in ordered:
            ordered.append(item)
    ordered += [e for e in evidence if e not in ordered]
    return ordered[:6]


def verification_status(results: list[dict]) -> str:
    statuses = [r["status"] for r in results]
    if not statuses or all(s == "missing" for s in statuses):
        return "INSUFFICIENT EVIDENCE"
    if any(s in {"contradicted", "mixed"} for s in statuses):
        return "MIXED EVIDENCE"
    return "PARTIALLY VERIFIED" if "missing" in statuses else "SUPPORTED BY AVAILABLE CHECKS"


def runtime_summary(run: dict) -> dict:
    telemetry = RunTelemetry.model_validate(run["telemetry"])
    calls = telemetry.inference_calls
    generation = sum(c.generation_latency_ms for c in calls) if calls and all(c.generation_latency_ms is not None for c in calls) else None
    tokens = sum(c.output_tokens for c in calls) if calls and all(c.output_tokens is not None for c in calls) else None
    memory = [c.device_allocated_memory_mb for c in calls if c.device_allocated_memory_mb is not None]
    return dict(model=calls[-1].routing.selected_profile.model_name if calls else "N/A",
                device=telemetry.hardware.device_name or telemetry.hardware.device,
                end_to_end_s=telemetry.end_to_end_latency_ms / 1000,
                generation_s=generation / 1000 if generation is not None else None,
                tokens_per_second=tokens / (generation / 1000) if tokens is not None and generation else None,
                gpu_snapshot_mb=max(memory) if memory else None, gpu_peak_mb=None,
                routing_ms=telemetry.routing_overhead_ms)
