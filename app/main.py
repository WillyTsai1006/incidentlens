"""Evidence-grounded Phase 2 diagnosis CLI; mock by default, local opt-in."""

import argparse
import json
import logging
from pathlib import Path

from core.correlation import correlate
from core.diagnosis import diagnose
from core.evidence_extractor import extract_evidence
from core.evidence_repository import EvidenceRepository
from core.incident_loader import load_incident
from core.schemas import Evidence, EvidenceRelation
from inference.base import InferenceBackend, InferenceUnavailable
from inference.mock_backend import MockBackend

DEMO_DIRECTORY = Path(__file__).resolve().parents[1] / "examples/gpu_oom_duplicate_models"


def analyze_core(directory: str | Path) -> dict:
    repository = EvidenceRepository(extract_evidence(load_incident(directory)))
    derived, relations = correlate(repository)
    repository.extend(derived)
    repository.validate_relations(relations)
    return {
        "phase": 1,
        "evidence": [item.model_dump(mode="json") for item in repository.all()],
        "relations": [item.model_dump(mode="json") for item in relations],
    }


def analyze_incident(directory: str | Path, backend: InferenceBackend | None = None) -> dict:
    artifacts = analyze_core(directory)
    repository = EvidenceRepository([Evidence.model_validate(e) for e in artifacts["evidence"]])
    relations = [EvidenceRelation.model_validate(r) for r in artifacts["relations"]]
    result = diagnose(repository, relations, backend if backend is not None else MockBackend())
    return {**artifacts, "phase": 2, "diagnosis": result.model_dump(mode="json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true", help="Parse the bundled synthetic incident")
    inputs.add_argument("--incident", type=Path, help="Directory of local incident files")
    parser.add_argument("--core-only", action="store_true", help="Export Phase 1 artifacts without diagnosis")
    parser.add_argument("--backend", choices=["mock", "transformers"], default="mock")
    parser.add_argument("--model-path", type=Path, help="Existing local model directory; no downloads")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--phase3", action="store_true", help="Run adaptive routing and telemetry; persist this run")
    parser.add_argument("--save-run", action="store_true", help="Produce a complete Phase 3 competition run artifact")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--profiles", type=Path, help="JSON list of structured model/backend profiles")
    parser.add_argument("--latency-preference", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--available-memory-mb", type=int, help="Explicit memory budget override in MiB, not a measurement")
    parser.add_argument("--complexity-score", type=float, help="Override the deterministic [0,1] evidence/graph complexity estimate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        directory = DEMO_DIRECTORY if args.demo else args.incident
        if args.core_only:
            if args.phase3 or args.save_run:
                raise ValueError("--core-only cannot be combined with Phase 3")
            artifacts = analyze_core(directory)
        elif args.phase3 or args.save_run:
            from pydantic import TypeAdapter
            from app.phase3 import default_profiles, run_phase3
            from core.router import ModelProfile

            profiles = (TypeAdapter(list[ModelProfile]).validate_json(args.profiles.read_text(encoding="utf-8"))
                        if args.profiles else default_profiles(args.backend, args.model_path, args.device))
            artifacts = run_phase3(directory, profiles=profiles,
                                   execution_mode="mock" if args.backend == "mock" else "local",
                                   device=args.device, latency_preference=args.latency_preference,
                                   available_memory_mb=args.available_memory_mb, complexity_score=args.complexity_score,
                                   save_root=args.runs_dir)
        else:
            backend = MockBackend()
            if args.backend == "transformers":
                if args.model_path is None:
                    raise ValueError("--model-path is required for the Transformers backend")
                from inference.transformers_backend import TransformersBackend

                backend = TransformersBackend(args.model_path, device=args.device)
            artifacts = analyze_incident(directory, backend)
    except (OSError, ValueError, InferenceUnavailable) as exc:
        logging.getLogger(__name__).error("Cannot analyze incident: %s", exc)
        return 2
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
