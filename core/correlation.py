"""Deterministic correlations for the bundled worker-startup incident format.

Returns new derived evidence and relations; callers explicitly add those facts
to the repository. Relationships express observations/inferences, not proof of
causality from co-occurrence alone.
"""

from datetime import datetime
import math
import re

from core.evidence_repository import EvidenceRepository
from core.schemas import Evidence, EvidenceRelation


def _before(left: Evidence, right: Evidence, *, allow_equal: bool = False) -> bool:
    if not left.timestamp or not right.timestamp:
        return False
    try:
        first, second = datetime.fromisoformat(left.timestamp), datetime.fromisoformat(right.timestamp)
        return first <= second if allow_equal else first < second
    except (ValueError, TypeError):
        return False


def correlate(repository: EvidenceRepository) -> tuple[list[Evidence], list[EvidenceRelation]]:
    observed = [e for e in repository.all() if e.category != "derived_fact"]
    derived: list[Evidence] = []
    relations: list[EvidenceRelation] = []

    def unique(suffix: str) -> Evidence | None:
        matches = [e for e in observed if e.id.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    def relate(left: Evidence, kind: str, right: Evidence, explanation: str):
        relations.append(EvidenceRelation(
            source_evidence_id=left.id, relation=kind,
            target_evidence_id=right.id, explanation=explanation,
        ))

    def fact(key: str, statement: str, value, inputs: list[Evidence], unit=None) -> Evidence:
        item = Evidence(
            id=f"derived:{key}", category="derived_fact", statement=statement,
            value=value, unit=unit, source=inputs[0].source,
        )
        derived.append(item)
        for parent in inputs:
            relate(item, "derived_from", parent, "Deterministic calculation uses this observed input.")
        return item

    workers = unique(":service.workers")
    per_worker = unique(":model.load_per_worker")
    code = [e for e in observed if e.value == "worker_model_load"]
    replicas = None
    if (workers and type(workers.value) is int and workers.value > 1
            and per_worker and per_worker.value is True and code):
        replicas = fact(
            "model_replicas", "Configured workers imply independent worker-local model replicas.",
            workers.value, [workers, per_worker, *code], "replicas",
        )
        relate(workers, "supports", replicas, "Multiple workers are configured.")
        for item in code:
            relate(item, "supports", replicas, "Model loading occurs inside worker_main.")

    starts = [e for e in observed if e.id.endswith(":worker_start")]
    memory = [e for e in observed if e.id.endswith(":memory") and e.category == "system_metric" and e.unit == "GiB"]
    memory.sort(key=lambda e: (e.source.file_path, e.source.line_start or 0))
    memory_starts = []
    for reading in memory[1:]:
        aligned = [e for e in starts if e.source == reading.source and e.timestamp == reading.timestamp]
        if len(aligned) == 1:
            memory_starts.append(aligned[0])
    increments = [float(b.value) - float(a.value) for a, b in zip(memory, memory[1:])]
    has_baseline = bool(memory) and not any(e.source == memory[0].source for e in starts)
    coherent_growth = (
        len(memory) >= 3 and len({e.source.file_path for e in memory}) == 1
        and has_baseline and len(memory_starts) == len(memory) - 1
        and len({e.value for e in memory_starts}) == len(memory_starts)
        and all(_before(a, b) for a, b in zip(memory, memory[1:]))
        and all(delta > 0 and math.isclose(delta, increments[0], rel_tol=0.01) for delta in increments)
    )
    growth = None
    if coherent_growth:
        increment = round(sum(increments) / len(increments), 6)
        growth = fact(
            "per_worker_memory_gib", "Approximate GPU memory increment per observed worker startup.",
            increment, [*memory, *memory_starts], "GiB",
        )
        for start, reading in zip(memory_starts, memory[1:]):
            relate(start, "correlates_with", reading, "Worker start and GPU reading share a log location and timestamp.")
        if replicas:
            relate(replicas, "correlates_with", growth, "Worker-local model configuration is consistent with startup memory steps.")

    capacity, capacity_unit = unique(":runtime.gpu_memory_limit_gb"), unique(":runtime.gpu_memory_unit")
    over_capacity = None
    if (growth and replicas and replicas.value >= len(memory_starts)
            and capacity and type(capacity.value) in (int, float)
            and math.isfinite(capacity.value) and capacity.value > 0
            and capacity_unit and capacity_unit.value == "GiB"):
        demand = fact(
            "projected_memory_gib", "Baseline + configured replicas × observed startup increment.",
            round(float(memory[0].value) + replicas.value * growth.value, 6),
            [memory[0], replicas, growth], "GiB",
        )
        over_capacity = fact(
            "exceeds_capacity", "Projected worker-local memory demand exceeds configured GPU capacity.",
            demand.value > capacity.value, [demand, capacity, capacity_unit],
        )

    ooms = [e for e in observed if e.id.endswith(":oom")]
    failures = [e for e in observed if e.id.endswith(":service_failure")]
    for oom in ooms:
        if growth and _before(memory[-1], oom):
            relate(memory[-1], "precedes", oom, "Last observed memory increase precedes the allocation failure.")
            if over_capacity and over_capacity.value is True:
                relate(over_capacity, "supports", oom, "Projected demand exceeds capacity before startup allocation fails.")
        for start in starts:
            if (_before(start, oom) and isinstance(start.value, str)
                    and re.search(rf"\b{re.escape(start.value)}\b", oom.source.raw_excerpt)):
                relate(start, "precedes", oom, "The named worker starts before its allocation fails.")
        for failure in failures:
            if _before(oom, failure):
                relate(oom, "precedes", failure, "Allocation failure occurs before service startup failure; causality is inferred later.")
    return derived, relations
