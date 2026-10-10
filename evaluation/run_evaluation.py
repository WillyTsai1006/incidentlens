"""Controlled three-case evaluation; no scientific/general RCA accuracy claim."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from pydantic import TypeAdapter

from app.phase3 import run_phase3
from core.router import ModelProfile
from core.run_artifacts import verify_manifest
from evaluation.baseline import run_baseline
from inference.routed_backend import create_backend

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("A", "gpu_oom_duplicate_models", "duplicate_models"),
    ("B", "gpu_oom_runtime_allocation", "runtime_allocation"),
    ("C", "gpu_oom_memory_leak", "memory_leak"),
)


class EvaluationFailure(ValueError):
    pass


def inspect_incidentlens(run: dict, directory: Path, expected_top: str, mode: str) -> dict:
    evidence = {e["id"]: e for e in run["evidence"]}
    if len(evidence) != len(run["evidence"]):
        raise EvaluationFailure("Duplicate evidence IDs")
    expected = json.loads((directory / "expected_evidence.json").read_text(encoding="utf-8"))
    for key, count in (("observed_count", sum(e["category"] != "derived_fact" for e in evidence.values())),
                       ("derived_count", sum(e["category"] == "derived_fact" for e in evidence.values()))):
        if expected[key] != count:
            raise EvaluationFailure(f"Expected evidence {key}: {expected[key]}, got {count}")
    for item in expected["required_evidence"]:
        if item["id"] not in evidence or not item.items() <= evidence[item["id"]].items():
            raise EvaluationFailure(f"Missing or mismatched expected evidence: {item['id']}")
    diagnosis = run["diagnosis"]
    if not diagnosis or diagnosis["execution_mode"] != mode or run["metadata"]["status"] != "completed":
        raise EvaluationFailure("Diagnosis failed or execution mode changed")
    primary = diagnosis["primary_root_cause"]
    actual = primary["kind"] if primary else None
    if actual != expected_top:
        raise EvaluationFailure(f"Expected top {expected_top}, got {actual}; scoring was not changed")
    hypotheses = diagnosis["hypotheses"]
    expected_diagnosis = json.loads((directory / "expected_diagnosis.json").read_text(encoding="utf-8"))
    if expected_diagnosis["primary_kind"] != expected_top or [h["kind"] for h in hypotheses] != expected_diagnosis["ranked_kinds"]:
        raise EvaluationFailure("Expected diagnosis ranking mismatch")
    if not expected_diagnosis["confidence_range"][0] <= primary["confidence"] <= expected_diagnosis["confidence_range"][1]:
        raise EvaluationFailure("Expected primary evidence score range mismatch")
    statuses = {v["criterion_id"]: v["status"] for v in primary["verification_results"]}
    for key in expected_diagnosis.get("required_supported_criteria", []):
        if statuses.get(key) != "supported":
            raise EvaluationFailure(f"Expected supported criterion: {key}")
    if "required_missing_criterion" in expected_diagnosis and statuses.get(expected_diagnosis["required_missing_criterion"]) != "missing":
        raise EvaluationFailure("Expected missing criterion changed")

    def check_ids(ids):
        if not set(ids) <= evidence.keys():
            raise EvaluationFailure("Unknown evidence ID accepted")

    claims = [*diagnosis["factual_claims"], *(c for h in hypotheses for c in h["claims"])]
    for claim in claims:
        check_ids(claim["evidence_ids"])
        if len(claim["evidence_ids"]) != 1 or claim["statement"] != evidence[claim["evidence_ids"][0]]["statement"]:
            raise EvaluationFailure("Unsupported factual claim accepted")
    for h in hypotheses:
        check_ids([*h["supporting_evidence_ids"], *h["contradicting_evidence_ids"]])
        for v in h["verification_results"]:
            check_ids([*v["supporting_evidence_ids"], *v["contradicting_evidence_ids"]])
    for item in [*diagnosis["supporting_evidence"], *diagnosis["contradicting_evidence"]]:
        check_ids([item["id"]])
    for h in [primary, *diagnosis["alternative_hypotheses"]]:
        check_ids([*h["supporting_evidence_ids"], *h["contradicting_evidence_ids"]])
        for claim in h["claims"]:
            check_ids(claim["evidence_ids"])
            if len(claim["evidence_ids"]) != 1 or claim["statement"] != evidence[claim["evidence_ids"][0]]["statement"]:
                raise EvaluationFailure("Unsupported factual claim accepted in diagnosis view")
    if not run["telemetry"]["inference_calls"]:
        raise EvaluationFailure("No inference calls recorded")
    for relation in run["relations"]:
        check_ids([relation["source_evidence_id"], relation["target_evidence_id"]])
    for call in run["telemetry"]["inference_calls"]:
        if call["execution_mode"] != mode or call["status"] != "completed":
            raise EvaluationFailure("Inference failed or silently fell back")
    artifact = run.get("artifact_directory")
    if artifact:
        verify_manifest(artifact)
    return dict(expected_top=expected_top, actual_top=actual, correct_top=True,
                scores={h["kind"]: h["confidence"] for h in hypotheses},
                verification_statuses={h["kind"]: {v["criterion_id"]: v["status"] for v in h["verification_results"]} for h in hypotheses},
                evidence_grounding_rate=diagnosis["evidence_grounding_rate"],
                final_claim_count=len(diagnosis["factual_claims"]), accepted_claim_count=len(claims),
                unsupported_accepted_claim_count=0,
                missing_evidence_count=sum(len(h["missing_evidence"]) for h in hypotheses),
                primary_missing_evidence_count=len(primary["missing_evidence"]),
                contradicting_evidence_count=len({key for h in hypotheses for key in h["contradicting_evidence_ids"]}),
                end_to_end_latency_ms=run["telemetry"]["end_to_end_latency_ms"],
                execution_mode=diagnosis["execution_mode"], backend=diagnosis["inference_backend"], model=diagnosis["selected_model"],
                device=run["telemetry"]["hardware"], telemetry=run["telemetry"], artifact_path=artifact)


def markdown_report(report: dict) -> str:
    completed = [c for c in report["cases"] if c.get("incidentlens")]
    baselines = [c["baseline"] for c in completed if c["baseline"]["status"] == "completed"]
    executed = [c["baseline"] for c in report["cases"] if c["baseline"]["status"] != "not_executed"]
    unavailable = "Unscorable (failed)" if executed else "Not executed"
    plain = (f"{sum(b['correct_top'] for b in baselines)}/3 unambiguous expected tops" if baselines
             else "Executed but unscorable (failed)" if executed else "Not executed")
    claims = sum(c["incidentlens"]["accepted_claim_count"] for c in completed)
    lines = ["# Controlled three-case evaluation", "", "All incidents are synthetic; sample size = 3. This is not a scientific benchmark.",
             f"Overall status: **{report['status']}**. IncidentLens execution: **{report['execution_mode']}**.", "",
             "| Metric | Plain Local LLM | IncidentLens |", "| --- | --- | --- |",
             f"| Correct top hypothesis | {plain} | {sum(c['incidentlens']['correct_top'] for c in completed)}/3 |",
             f"| Evidence-grounded factual claims | {str(sum(b['metrics']['grounded_quote_count'] for b in baselines)) + '/' + str(sum(b['metrics']['quoted_observation_count'] for b in baselines)) + ' source quotations; prose unverified' if baselines else 'Unscorable' if executed else 'Not executed'} | {claims}/{claims} exact referenced statements |",
             f"| Unsupported accepted claims | {'Unknown for free prose; invalid quotations counted below' if baselines else unavailable} | 0 |",
             f"| Explicit missing evidence | {sum(b['metrics']['missing_evidence_count'] for b in baselines) if baselines else unavailable} | {sum(c['incidentlens']['missing_evidence_count'] for c in completed)} criterion gaps across all hypotheses |",
             f"| Contradicting evidence exposed | {sum(b['metrics']['contradicting_evidence_count'] for b in baselines) if baselines else unavailable} source-valid quoted observations; interpretation unverified | {sum(c['incidentlens']['contradicting_evidence_count'] for c in completed)} unique evidence IDs summed per case |",
             "| Deterministic diagnosis verification | No | Yes |", "",
             "Baseline quote matching measures source traceability, not whether a quote supports a cause. Multi-cause lists have no unambiguous top and are not guessed correct. Counts across systems have different units; they are not accuracy comparisons. Root-cause responses are hypotheses, not proven causal facts.", ""]
    for case in report["cases"]:
        lines.extend([f"## Case {case['id']} — {case['expected_top']}", ""])
        if case.get("error"):
            lines.extend([f"FAILED: {case['error']}", ""])
        if not case.get("incidentlens"):
            continue
        r, b = case["incidentlens"], case["baseline"]
        lines.extend([f"IncidentLens top: **{r['actual_top']}**; scores: `{json.dumps(r['scores'])}`.",
                      f"Latency: {r['end_to_end_latency_ms']/1000:.3f} s; grounding: {r['evidence_grounding_rate']:.1%}; missing criteria: {r['missing_evidence_count']} (primary: {r['primary_missing_evidence_count']}).",
                      f"Verification: `{json.dumps(r['verification_statuses'])}`.",
                      f"Artifact: `{r['artifact_path']}`.", f"Baseline status: **{b['status']}**."])
        if b["status"] == "completed":
            m = b["metrics"]
            lines.extend([f"Baseline top: `{m['actual_top']}`; correct: {b['correct_top']}; latency: {b['end_to_end_latency_ms']/1000:.3f} s.",
                          f"Source-valid quotations: {m['grounded_quote_count']}/{m['quoted_observation_count']}; unsupported quotations: {m['unsupported_quoted_claim_count']}; unverified explanation units: {m['unverified_explanation_count']}.",
                          "```json", json.dumps(b["response"], indent=2), "```"])
        elif b["status"] == "failed":
            lines.extend([b["error"], "````text", b["output"], "````"])
        lines.append("")
    lines.extend(["## Limitations", "", "Synthetic incidents only; three cases; predefined hypothesis families; Qwen2.5-0.5B-Instruct for local inference. No claim of production RCA accuracy or general incident coverage. Intel Arc Pro B70 remains unvalidated. Scores are heuristic evidence scores, not probabilities. Mock runs are not real-LLM comparisons. Free-form baseline explanations require manual review; no LLM evaluator is used. Timing includes cold loading, differs with call count and environment, and is not a speed benchmark.", ""])
    return "\n".join(lines)


def write_reports(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")


def run_evaluation(*, execution_mode="mock", device="cpu", profiles=None,
                   output_dir=Path("output/controlled_evaluation/mock"), persist=True,
                   examples_root=ROOT / "examples", pipeline=run_phase3, factory=create_backend) -> dict:
    output_dir, examples_root = Path(output_dir), Path(examples_root)
    if execution_mode not in {"mock", "local"}:
        raise EvaluationFailure("Evaluation mode must be mock or local")
    if execution_mode == "local":
        if not profiles or len(profiles) != 1 or profiles[0].execution_mode != "local" or profiles[0].device != device:
            raise EvaluationFailure("Real comparison requires one explicit local profile matching the requested device")
        if profiles[0].backend != "transformers-local" or Path(profiles[0].model_path or "").name != "Qwen2.5-0.5B-Instruct":
            raise EvaluationFailure("Real comparison requires the existing local Qwen2.5-0.5B-Instruct path")
    report = dict(title="Controlled three-case evaluation", created_at=datetime.now(timezone.utc).isoformat(),
                  synthetic=True, sample_size=3, execution_mode=execution_mode, requested_device=device,
                  profiles=[p.model_dump(mode="json") for p in profiles] if profiles else [], status="passed", cases=[])
    for case_id, folder, expected_top in CASES:
        print(f"Evaluating case {case_id} ({execution_mode})", file=sys.stderr, flush=True)
        case = dict(id=case_id, expected_top=expected_top, incidentlens=None,
                    baseline=dict(status="not_executed", reason="Mock CI run; no plain local LLM was executed"))
        report["cases"].append(case)
        try:
            directory = examples_root / folder
            run = pipeline(directory, profiles=profiles, execution_mode=execution_mode,
                           device=device, save_root=output_dir / "runs", persist=persist, factory=factory)
            case["artifact_path"] = run.get("artifact_directory")
            if any(call["routing"]["selected_profile"]["device"] != device for call in run["telemetry"]["inference_calls"]):
                raise EvaluationFailure("Inference routed to a different device than requested")
            case["incidentlens"] = inspect_incidentlens(run, directory, expected_top, execution_mode)
            case["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
            if execution_mode == "local":
                print(f"Running plain local baseline for case {case_id}", file=sys.stderr, flush=True)
                backend = factory(profiles[0])
                if (backend.is_mock or backend.name != profiles[0].backend or getattr(backend, "device", None) != device
                        or backend.model_name != run["diagnosis"]["selected_model"]):
                    raise EvaluationFailure("Baseline backend identity/device mismatch; no fallback allowed")
                baseline = run_baseline(directory, backend)
                del backend
                case["baseline"] = baseline
                if baseline["status"] != "completed":
                    raise EvaluationFailure(baseline["error"])
                baseline["correct_top"] = baseline["metrics"]["actual_top"] == expected_top
        except (OSError, ValueError, RuntimeError) as exc:
            report["status"] = "failed"
            case["error"] = str(exc)
    write_reports(report, output_dir)
    if report["status"] != "passed":
        raise EvaluationFailure(f"Controlled evaluation failed; inspect {output_dir / 'report.json'}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "local"], default="mock")
    parser.add_argument("--device", choices=["cpu", "cuda", "xpu"], default="cpu")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args(argv)
    try:
        profiles = TypeAdapter(list[ModelProfile]).validate_json(args.profiles.read_text(encoding="utf-8")) if args.profiles else None
        report = run_evaluation(execution_mode=args.backend, device=args.device, profiles=profiles,
                                output_dir=args.output_dir or Path("output/controlled_evaluation") / args.backend,
                                persist=not args.no_artifacts)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"{report['title']}: 3/3 IncidentLens top hypotheses correct ({args.backend}); reports saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
