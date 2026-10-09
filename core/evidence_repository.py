"""In-memory evidence indexed by stable IDs, exportable as JSON."""

from core.schemas import Evidence, EvidenceRelation


class EvidenceRepository:
    def __init__(self, evidence: list[Evidence] | None = None):
        self._items: dict[str, Evidence] = {}
        self.extend(evidence or [])

    def extend(self, evidence: list[Evidence]) -> None:
        ids = [item.id for item in evidence]
        if len(ids) != len(set(ids)) or any(key in self._items for key in ids):
            raise ValueError("Duplicate evidence ID")
        self._items.update((item.id, item) for item in evidence)

    def get(self, evidence_id: str) -> Evidence:
        return self._items[evidence_id]

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def validate_relations(self, relations: list[EvidenceRelation]) -> None:
        for relation in relations:
            for key in (relation.source_evidence_id, relation.target_evidence_id):
                if key not in self._items:
                    raise ValueError(f"Unknown evidence ID in relation: {key}")

    def to_json(self) -> str:
        from pydantic import TypeAdapter

        return TypeAdapter(list[Evidence]).dump_json(self.all(), indent=2).decode()
