"""Application-owned verification criteria for three competing explanations.

Results describe evidence compatibility; none establishes a proven causal link.
Missing observations are explicit, not treated as evidence of absence.
"""

from datetime import datetime
import re

from core.evidence_repository import EvidenceRepository
from core.hypothesis import validate_claims
from core.schemas import ConfidenceMetadata, Evidence, EvidenceRelation, Hypothesis, VerificationCriterion, VerificationResult


def criterion(key, support, contradiction, *, weight=1, penalty=0.1, discriminating=True):
    return VerificationCriterion(id=key, supporting_observation=support,
                                 contradicting_observation=contradiction, weight=weight,
                                 contradiction_penalty=penalty, discriminating=discriminating)


OOM = criterion("oom", "An allocation/OOM event is observed.",
                "An explicitly verified successful allocation would weaken allocation failure.",
                weight=0.25, discriminating=False)
PLANS = {
    "duplicate_models": [
        criterion("multiple_workers", "Configuration specifies more than one worker.",
                  "Configuration specifies a single worker."),
        criterion("worker_model", "Worker-local loading is enabled and load_model occurs in worker_main.",
                  "Worker-local model loading is explicitly disabled."),
        criterion("startup_steps", "Uniform GPU memory increments align with distinct worker starts.",
                  "Measured growth occurs independently of worker starts; investigate another mechanism."),
        criterion("capacity", "Projected worker-local demand exceeds the configured capacity.",
                  "Projected demand fits within configured capacity.", penalty=0.2),
        criterion("startup_alignment", "Worker startup and memory growth precede OOM, followed by service failure.",
                  "Failure timing instead aligns with request execution."),
        criterion("process_replicas", "Process-level allocations confirm a distinct model replica per worker.",
                  "Process-level allocations confirm a single shared replica."),
    ],
    "runtime_allocation": [
        OOM,
        criterion("batch_pressure", "Measured batch allocation exceeds measured pre-batch headroom.",
                  "Measured batch allocation fits within pre-batch headroom.", penalty=0.2),
        criterion("runtime_timing", "Allocation failure is explicitly observed during inference/request execution.",
                  "Observed failure occurs during initialization; this weakens a request-time explanation.", penalty=0.08),
    ],
    "memory_leak": [
        OOM,
        criterion("steady_growth", "Memory grows over multiple steady-state samples at fixed active worker count.",
                  "Memory is stable over the sampled steady-state window; this weakens but does not exclude a leak."),
        criterion("leak_pattern", "Steady-state growth occurs without additional worker starts.",
                  "Uniform memory steps align with worker starts, providing a competing initialization explanation.", penalty=0.12),
    ],
}


def _ordered(left: Evidence, right: Evidence) -> bool:
    try:
        return bool(left.timestamp and right.timestamp
                    and datetime.fromisoformat(left.timestamp) < datetime.fromisoformat(right.timestamp))
    except (TypeError, ValueError):
        return False


def _steady_state(items: list[Evidence]) -> tuple[list[str], list[str]]:
    samples = [e for e in items if e.id.endswith(":memory") and "phase=steady_state" in e.source.raw_excerpt]
    samples.sort(key=lambda e: (e.source.file_path, e.source.line_start or 0))
    counts = [re.search(r"\bactive_workers=(\d+)\b", e.source.raw_excerpt) for e in samples]
    if (len(samples) < 3 or len({e.source.file_path for e in samples}) != 1
            or not all(counts) or len({c.group(1) for c in counts if c}) != 1
            or not all(_ordered(a, b) for a, b in zip(samples, samples[1:]))):
        return [], []
    starts = [e for e in items if e.id.endswith(":worker_start")]
    if any(_ordered(samples[0], e) and _ordered(e, samples[-1]) for e in starts):
        return [], []
    ids = [e.id for e in samples]
    if all(b.value > a.value for a, b in zip(samples, samples[1:])):
        return ids, []
    if len({e.value for e in samples}) == 1:
        return [], ids
    return [], []


def verify_hypothesis(hypothesis: Hypothesis, repository: EvidenceRepository,
                      relations: list[EvidenceRelation]) -> Hypothesis:
    repository.validate_relations(relations)
    validate_claims(hypothesis.claims, repository)
    if hypothesis.kind not in PLANS:
        raise ValueError("No verification criteria for hypothesis kind")
    items = repository.all()
    by_id = {e.id: e for e in items}
    plan = [item.model_copy(deep=True) for item in PLANS[hypothesis.kind]]

    def config(suffix):
        matches = [e for e in items if e.id.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    steady_support, steady_contradiction = _steady_state(items)
    ooms = [e for e in items if e.id.endswith(":oom")]
    startup_ooms = [e for e in ooms if "during initialization" in e.source.raw_excerpt]
    runtime_ooms = [e for e in ooms if "phase=inference" in e.source.raw_excerpt or "during inference" in e.source.raw_excerpt]
    steps = by_id.get("derived:per_worker_memory_gib")

    def evaluate(key):
        if key == "oom":
            return [e.id for e in ooms], []
        if key == "multiple_workers":
            workers = config(":service.workers")
            if workers and type(workers.value) is int and workers.value > 0:
                return ([workers.id], []) if workers.value > 1 else ([], [workers.id])
        elif key == "worker_model":
            flag = config(":model.load_per_worker")
            code = [e.id for e in items if e.value == "worker_model_load"]
            if flag and flag.value is False:
                return [], [flag.id]
            if flag and flag.value is True and code:
                return [flag.id, *code], []
        elif key == "startup_steps":
            return ([steps.id] if steps else []), steady_support
        elif key == "capacity":
            pressure = by_id.get("derived:exceeds_capacity")
            if pressure:
                return ([pressure.id], []) if pressure.value is True else ([], [pressure.id])
        elif key == "startup_alignment":
            support = set()
            for oom in ooms:
                incoming = [r.source_evidence_id for r in relations
                            if r.relation == "precedes" and r.target_evidence_id == oom.id]
                failures = [r.target_evidence_id for r in relations if r.relation == "precedes"
                            and r.source_evidence_id == oom.id and r.target_evidence_id.endswith(":service_failure")]
                if (any(key.endswith(":worker_start") for key in incoming)
                        and any(key.endswith(":memory") for key in incoming) and failures):
                    support.update([*incoming, oom.id, *failures])
            return sorted(support), [e.id for e in runtime_ooms]
        elif key == "runtime_timing":
            return [e.id for e in runtime_ooms], [e.id for e in startup_ooms]
        elif key == "batch_pressure":
            support, contradiction = [], []
            for allocation in [e for e in items if e.id.endswith(":batch_allocation")]:
                headroom = [e for e in items if e.id.endswith(":batch_headroom") and e.source == allocation.source]
                if len(headroom) == 1:
                    ids = [allocation.id, headroom[0].id]
                    (support if allocation.value > headroom[0].value else contradiction).extend(ids)
            return support, contradiction
        elif key == "steady_growth":
            return steady_support, steady_contradiction
        elif key == "leak_pattern":
            return steady_support, ([steps.id] if steps else [])
        # No process-level replica observation is available in the MVP inputs.
        return [], []

    results = []
    for rule in plan:
        support, contradiction = evaluate(rule.id)
        status = "mixed" if support and contradiction else "supported" if support else "contradicted" if contradiction else "missing"
        results.append(VerificationResult(
            criterion_id=rule.id, status=status,
            supporting_evidence_ids=sorted(set(support)), contradicting_evidence_ids=sorted(set(contradiction)),
            explanation=("Both observations occur; the evidence is mixed." if status == "mixed"
                         else rule.supporting_observation if support
                         else rule.contradicting_observation if contradiction
                         else f"Not observed: {rule.supporting_observation}"),
        ))
    return hypothesis.model_copy(update={
        "verification_plan": plan, "verification_results": results,
        "supporting_evidence_ids": sorted({key for r in results for key in r.supporting_evidence_ids}),
        "contradicting_evidence_ids": sorted({key for r in results for key in r.contradicting_evidence_ids}),
        "missing_evidence": [r.explanation for r in results if r.status == "missing"],
        "confidence": 0.0,
        "confidence_metadata": ConfidenceMetadata(),
    })
