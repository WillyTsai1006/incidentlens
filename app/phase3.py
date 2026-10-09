"""Measured, routed pipeline; legacy Phase 1/2 entry points remain unchanged."""

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
from time import perf_counter

from core.correlation import correlate
from core.diagnosis import diagnose
from core.evidence_extractor import extract_evidence
from core.evidence_repository import EvidenceRepository
from core.incident_loader import IncidentFile, load_incident
from core.router import ModelProfile, RoutingContext
from core.run_artifacts import new_run_id, save_run
from inference.routed_backend import RoutedBackend, create_backend
from inference.telemetry import RunTelemetry, hardware_snapshot


class Phase3RunError(ValueError):
    pass


def default_profiles(backend: str = "mock", model_path: str | Path | None = None, device: str = "cpu") -> list[ModelProfile]:
    if backend == "mock":
        return [ModelProfile(id="mock-demo", backend="deterministic-mock", model_name="mock-candidate-catalog-v1",
                             tier="small_model", max_context_chars=200000, estimated_memory_mb=0, execution_mode="mock")]
    if backend != "transformers" or model_path is None:
        raise ValueError("A local model path is required for Transformers")
    path = Path(model_path).resolve()
    revision_file = path / ".cache/huggingface/download/model.safetensors.metadata"
    revision = revision_file.read_text(encoding="utf-8").splitlines()[0] if revision_file.is_file() else None
    return [ModelProfile(id="local-small", backend="transformers-local", model_name=path.name,
                         model_path=str(path), revision=revision, tier="small_model", device=device,
                         max_context_chars=100000, estimated_memory_mb=8192, execution_mode="local", max_new_tokens=512)]


def package_versions() -> dict:
    versions = {}
    for name in ("incident-diagnosis-agent", "pydantic", "PyYAML", "psutil", "torch", "transformers", "streamlit"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def run_phase3(directory: str | Path | list[IncidentFile], *, profiles: list[ModelProfile] | None = None,
               execution_mode: str = "mock", device: str = "cpu", latency_preference: str = "balanced",
               available_memory_mb: int | None = None, complexity_score: float | None = None,
               save_root: str | Path = "runs", persist: bool = True,
               factory: Callable = create_backend, progress: Callable[[str], None] | None = None) -> dict:
    started = perf_counter()
    run_id = new_run_id()
    created_at = datetime.now(timezone.utc).isoformat()
    project_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(project_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                     for folder in ("app", "core", "inference") for path in sorted((project_root/folder).glob("*.py"))}
    profiles = profiles if profiles is not None else default_profiles()
    hardware = hardware_snapshot(device)
    measured_budget = hardware.host_available_memory_mb if device == "cpu" else hardware.device_available_memory_mb
    budget = available_memory_mb if available_memory_mb is not None else int(measured_budget) if measured_budget is not None else None
    files, evidence, relations, result, routed, error = [], [], [], None, None, None
    context = None
    try:
        if progress:
            progress("Parsing sources and extracting evidence")
        files = load_incident(directory) if not isinstance(directory, list) else directory
        repository = EvidenceRepository(extract_evidence(files))
        if progress:
            progress("Correlating evidence")
        derived, relations = correlate(repository)
        repository.extend(derived)
        repository.validate_relations(relations)
        evidence = repository.all()
        context = RoutingContext(task_type="hypothesis_generation", context_chars=0, evidence_count=len(evidence),
                                 complexity_score=complexity_score if complexity_score is not None else min(1, len(evidence)/50 + len(relations)/100),
                                 available_memory_mb=budget, memory_scope="host" if device == "cpu" else device,
                                 latency_preference=latency_preference, execution_mode=execution_mode)
        routed = RoutedBackend(profiles, context, factory)
        result = diagnose(repository, relations, routed, progress=progress)
    except Exception as exc:
        error = exc
    calls = routed.calls if routed else []
    telemetry = RunTelemetry(end_to_end_latency_ms=(perf_counter() - started) * 1000,
                             routing_overhead_ms=sum(call.routing_overhead_ms for call in calls),
                             inference_calls=calls, hardware=hardware)
    request = dict(execution_mode=execution_mode, device=device, latency_preference=latency_preference,
                   available_memory_mb=budget, memory_budget_source="user override" if available_memory_mb is not None else "runtime snapshot or unavailable",
                   complexity_score=context.complexity_score if context else complexity_score,
                   generation=dict(temperature=0, requested_max_new_tokens=2048, max_attempts=2,
                                   attempts_scope="per candidate" if execution_mode == "local" else "candidate bundle",
                                   candidate_format="per_candidate_ids" if execution_mode == "local" else "quoted_claims", cpu_threads=4))
    metadata = dict(run_id=run_id, created_at=created_at, phase=3, schema_version=1,
                    status="failed" if error else "completed", error_type=type(error).__name__ if error else None,
                    python=platform.python_version(), packages=package_versions(), request=request,
                    source_sha256=source_hashes,
                    inputs=[dict(original_path=f.file_path, snapshot=f"inputs/{Path(f.file_path).name}",
                                 sha256=hashlib.sha256(f.content.encode("utf-8")).hexdigest()) for f in files])
    output = dict(phase=3, run_id=run_id, evidence=[e.model_dump(mode="json") for e in evidence],
                  relations=[r.model_dump(mode="json") for r in relations],
                  diagnosis=result.model_dump(mode="json") if result else None,
                  routing=[call.routing.model_dump(mode="json") for call in calls],
                  telemetry=telemetry.model_dump(mode="json"), metadata=metadata)
    if persist:
        artifacts = {
            "evidence.json": output["evidence"], "relations.json": output["relations"],
            "hypotheses.json": output["diagnosis"]["hypotheses"] if result else [],
            "verification.json": {h.id: [r.model_dump(mode="json") for r in h.verification_results] for h in result.hypotheses} if result else {},
            "diagnosis.json": output["diagnosis"], "routing.json": output["routing"],
            "telemetry.json": output["telemetry"], "metadata.json": metadata,
            "profiles.json": [p.model_dump(mode="json") for p in profiles],
            "inference_io.json": routed.call_io if routed else [],
        }
        saved = save_run(save_root, run_id, artifacts, files)
        output["artifact_directory"] = str(saved)
    if error:
        location = f"; artifacts: {output['artifact_directory']}" if persist else ""
        raise Phase3RunError(f"Phase 3 run failed ({type(error).__name__}): {error}{location}") from error
    return output
