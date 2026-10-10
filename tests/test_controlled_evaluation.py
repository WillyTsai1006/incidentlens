import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.phase3 import run_phase3
from core.run_artifacts import verify_manifest
from core.router import ModelProfile
from evaluation.baseline import (baseline_prompt, evaluate_baseline, hypothesis_family,
                                 parse_baseline, raw_sources, run_baseline)
from evaluation.run_evaluation import (CASES, ROOT, EvaluationFailure,
                                       inspect_incidentlens, main, run_evaluation)
from inference.mock_backend import MockBackend


@pytest.fixture
def runs():
    return {case_id: run_phase3(ROOT / "examples" / folder, persist=False)
            for case_id, folder, _ in CASES}


@pytest.mark.parametrize("case_id,folder,expected_top", CASES)
def test_controlled_fixture_and_ranking(runs, case_id, folder, expected_top):
    directory = ROOT / "examples" / folder
    metrics = inspect_incidentlens(runs[case_id], directory, expected_top, "mock")
    assert metrics["actual_top"] == expected_top
    assert metrics["evidence_grounding_rate"] == 1
    assert metrics["unsupported_accepted_claim_count"] == 0
    assert metrics["accepted_claim_count"] > 0
    for item in runs[case_id]["evidence"]:
        if item["category"] == "derived_fact":
            continue
        source = item["source"]
        lines = Path(source["file_path"]).read_text(encoding="utf-8").splitlines()
        assert source["raw_excerpt"] == "\n".join(lines[source["line_start"] - 1:source["line_end"]])
    assert all(r["relation"] != "causes" for r in runs[case_id]["relations"])


def test_case_b_measured_headroom_and_stable_samples(runs):
    items = runs["B"]["evidence"]
    by_id = {e["id"]: e for e in items}
    assert by_id["app.log:L6:batch_allocation"]["value"] > by_id["app.log:L6:batch_headroom"]["value"]
    samples = [e for e in items if e["id"].endswith(":memory") and "phase=steady_state" in e["source"]["raw_excerpt"]]
    assert len(samples) == 3 and {e["value"] for e in samples} == {16.0}
    assert all("active_workers=2" in e["source"]["raw_excerpt"] for e in samples)
    assert "during inference" in by_id["app.log:L7:oom"]["source"]["raw_excerpt"]
    assert by_id["config.yaml:service.workers"]["value"] * 8 == 16 < 32


def test_case_c_growth_at_fixed_workers(runs):
    items = runs["C"]["evidence"]
    samples = [e for e in items if e["id"].endswith(":memory") and "phase=steady_state" in e["source"]["raw_excerpt"]]
    assert [e["value"] for e in samples] == [16, 20, 24, 28, 31]
    assert all("active_workers=2" in e["source"]["raw_excerpt"] for e in samples)
    assert not any(samples[0]["timestamp"] < e["timestamp"] < samples[-1]["timestamp"]
                   for e in items if e["id"].endswith(":worker_start"))


def test_case_a_fixture_content_and_scores_unchanged(runs):
    # Frozen source content; normalize checkout line endings for portability.
    expected = {
        "app.log": "162801e4866254a6ba7429452f73be379b661887c8e12bb2faa21fa0950c761f",
        "config.yaml": "8029953570b12da8fb40a3b903ca6e512a930d6717b7d8b15f7ec0335777812d",
        "expected_diagnosis.json": "fd28a90555a2b6afd0e430fcca6f458467c5e34fbdcb5f2aac99d62b2e044ecf",
        "expected_evidence.json": "00f33ff6678881f3da100e34e9d3a32cf4ed8a202f7000fa423a32ac74e9bcfa",
        "inference.py": "12b769c0964a1b5073f0877a3d074be4b766acb32a460d4ef21b08576ebef4a0",
        "system.log": "16dc023ae8b07207ab2c2bb8a64f197a93ffb39b9be63930f05a56076bd9c8c8",
    }
    directory = ROOT / "examples/gpu_oom_duplicate_models"
    for name, digest in expected.items():
        assert hashlib.sha256((directory / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() == digest
    assert [h["confidence"] for h in runs["A"]["diagnosis"]["hypotheses"]] == [0.866667, 0.153333, 0.113333]


def test_three_case_runner_and_reports(tmp_path):
    report = run_evaluation(output_dir=tmp_path)
    assert report["status"] == "passed" and report["sample_size"] == 3
    assert all(c["baseline"]["status"] == "not_executed" for c in report["cases"])
    assert all(c["incidentlens"]["correct_top"] for c in report["cases"])
    assert json.loads((tmp_path / "report.json").read_text()) == report
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "3/3" in md and "Not executed" in md and "not a scientific benchmark" in md
    for case in report["cases"]:
        verify_manifest(case["incidentlens"]["artifact_path"])


def test_no_artifacts_option(tmp_path):
    report = run_evaluation(output_dir=tmp_path, persist=False)
    assert all(c["incidentlens"]["artifact_path"] is None for c in report["cases"])
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["evidence"].pop(), "Expected evidence"),
    (lambda r: r["diagnosis"]["factual_claims"][0].update(evidence_ids=["unknown"]), "Unknown evidence"),
    (lambda r: r["diagnosis"]["hypotheses"][0]["claims"][0].update(statement="invented fact"), "Unsupported factual"),
    (lambda r: r["diagnosis"]["hypotheses"][0]["supporting_evidence_ids"].append("unknown"), "Unknown evidence"),
    (lambda r: r["diagnosis"]["primary_root_cause"].update(kind="memory_leak"), "Expected top"),
    (lambda r: r["telemetry"]["inference_calls"][0].update(execution_mode="local"), "silently fell back"),
])
def test_evaluation_fails_on_invalid_pipeline_result(runs, mutation, message):
    run = copy.deepcopy(runs["A"])
    mutation(run)
    with pytest.raises(EvaluationFailure, match=message):
        inspect_incidentlens(run, ROOT / "examples/gpu_oom_duplicate_models", "duplicate_models", "mock")


def test_failed_case_report_and_exit_status(tmp_path):
    def failed(*args, **kwargs):
        raise ValueError("controlled pipeline failure")
    with pytest.raises(EvaluationFailure):
        run_evaluation(output_dir=tmp_path, pipeline=failed)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed" and all(c["error"] for c in report["cases"])
    assert main(["--backend", "local", "--output-dir", str(tmp_path / "invalid")]) == 2


def test_report_write_failure_propagates(tmp_path):
    target = tmp_path / "file"
    target.write_text("not a directory")
    with pytest.raises(OSError):
        run_evaluation(output_dir=target, persist=False)


def baseline_json(**changes):
    return json.dumps(dict(root_cause="runtime_allocation", evidence=[dict(file="app.log", line=1, quote="allocation observed")],
                           explanation="Runtime allocation may explain this incident.", uncertainty=["Allocation attribution is not confirmed."],
                           contradicting_evidence=[], **changes))


@pytest.mark.parametrize("text", ["not JSON", "{}", '{"root_cause": 123}'])
def test_baseline_parse_rejects_unscorable_structure(text):
    with pytest.raises(ValueError):
        parse_baseline(text)


def test_baseline_parser_and_traceability_are_conservative():
    response = parse_baseline("```json\n" + baseline_json() + "\n```")
    metrics = evaluate_baseline(response, {"app.log": "allocation observed\n"})
    assert metrics["actual_top"] == "runtime_allocation"
    assert metrics["evidence_grounding_rate"] == 1
    assert metrics["unverified_explanation_count"] == 1
    assert metrics["unsupported_accepted_claim_count"] is None
    assert metrics["missing_evidence_count"] == 1


@pytest.mark.parametrize("change", [dict(file="missing.log"), dict(line=99), dict(quote="invented"), dict(file="../app.log")])
def test_baseline_unknown_or_fabricated_source_not_grounded(change):
    response = parse_baseline(baseline_json())
    response.evidence[0] = response.evidence[0].model_copy(update=change)
    metrics = evaluate_baseline(response, {"app.log": "allocation observed"})
    assert metrics["unsupported_quoted_claim_count"] == 1
    assert metrics["evidence_grounding_rate"] == 0


@pytest.mark.parametrize("root,expected", [("memory_leak", "memory_leak"), (" Runtime_Allocation ", "runtime_allocation"),
                                          ("duplicate_models or memory_leak", None), ("unknown", None),
                                          (["Duplicate models"], "duplicate_models"), (["Duplicate models", "Memory leak detected"], None)])
def test_baseline_family_mapping_does_not_guess(root, expected):
    assert hypothesis_family(root) == expected


class PlainBackend:
    name, model_name, is_mock = "test-only-local-stub", "test model", False

    def __init__(self, output=None):
        self.output = baseline_json() if output is None else output
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.output


def test_baseline_one_call_no_extraction_or_verification(tmp_path, monkeypatch):
    from core import evidence_extractor, verifier, confidence
    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline shared diagnosis logic")
    monkeypatch.setattr(evidence_extractor, "extract_evidence", forbidden)
    monkeypatch.setattr(verifier, "verify_hypothesis", forbidden)
    monkeypatch.setattr(confidence, "score_verified_hypothesis", forbidden)
    (tmp_path / "app.log").write_text("allocation observed")
    (tmp_path / "expected_diagnosis.json").write_text('{"secret_expected_answer": "ignored"}')
    backend = PlainBackend()
    result = run_baseline(tmp_path, backend)
    assert len(backend.calls) == 1
    assert backend.calls[0][1] == dict(temperature=0, max_new_tokens=512)
    assert "secret_expected_answer" not in backend.calls[0][0]
    assert result["status"] == "completed"
    tree = ast.parse((ROOT / "evaluation/baseline.py").read_text())
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("core", "app")) for n in ast.walk(tree))


def test_baseline_bad_json_is_explicit_failure(tmp_path):
    (tmp_path / "app.log").write_text("allocation observed")
    result = run_baseline(tmp_path, PlainBackend("incomplete JSON"))
    assert result["status"] == "failed" and result["metrics"] is None
    assert result["output"] == "incomplete JSON"


def test_baseline_refuses_mock(tmp_path):
    with pytest.raises(ValueError, match="mock"):
        run_baseline(tmp_path, MockBackend())


def test_baseline_raw_sources_and_prompt_exclude_expected_answers():
    sources = raw_sources(ROOT / "examples/gpu_oom_memory_leak")
    assert set(sources) == {"app.log", "system.log", "config.yaml", "inference.py"}
    prompt = baseline_prompt(sources)
    assert "evidence_ids" not in prompt and "verification_plan" not in prompt
    assert "phase=steady_state active_workers=2" in prompt


def test_baseline_partial_source_quote_and_wrong_line():
    response = parse_baseline(baseline_json())
    response.evidence[0].quote = "allocation observed\n"
    assert evaluate_baseline(response, {"app.log": "2026-10-08T10:00:21Z INFO allocation observed"})["grounded_quote_count"] == 1
    assert evaluate_baseline(response, {"app.log": "unrelated\nallocation observed"})["grounded_quote_count"] == 0


def test_local_runner_comparison_uses_separate_plain_calls(tmp_path):
    calls = []

    class LocalStub(PlainBackend):
        name, model_name, device = "transformers-local", "test-only Qwen stub", "cpu"

        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            if prompt.startswith("{"):
                payload = json.loads(prompt)
                return json.dumps({"evidence_ids": [payload["evidence"][0]["id"]]})
            # Deliberately wrong baseline family; this is a comparison outcome,
            # not an IncidentLens evaluation failure. No real inference claimed.
            sources = raw_sources(ROOT / "examples/gpu_oom_duplicate_models")
            return json.dumps(dict(root_cause="duplicate_models", evidence=[dict(file="app.log", line=1, quote=sources["app.log"].splitlines()[0])],
                                   explanation="A candidate explanation.", uncertainty=[], contradicting_evidence=[]))

    profile = ModelProfile(id="local-evaluation-test", backend="transformers-local", model_name="test-only Qwen stub",
                           model_path="Qwen2.5-0.5B-Instruct", tier="small_model", device="cpu", execution_mode="local",
                           estimated_memory_mb=0, max_context_chars=100000)
    report = run_evaluation(execution_mode="local", profiles=[profile], output_dir=tmp_path, persist=False,
                            factory=lambda _: LocalStub())
    assert report["status"] == "passed"
    assert len(calls) == 12  # 3 evidence-selection calls + 1 plain call per case.
    assert sum(c["baseline"]["correct_top"] for c in report["cases"]) == 1
    assert all(c["baseline"]["generation"]["calls"] == 1 for c in report["cases"])
    assert "1/3" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_baseline_empty_quotes_not_perfect_grounding():
    response = parse_baseline(baseline_json())
    response.evidence = []
    assert evaluate_baseline(response, {})["evidence_grounding_rate"] is None


def test_baseline_contradiction_count_only_source_valid_quotes():
    response = parse_baseline(baseline_json())
    response.contradicting_evidence = [response.evidence[0].model_copy(), response.evidence[0].model_copy(update={"quote": "not in source"})]
    metrics = evaluate_baseline(response, {"app.log": "allocation observed"})
    assert metrics["contradicting_evidence_count"] == 1
    assert metrics["unsupported_quoted_claim_count"] == 1


def test_uncertainty_labels_are_not_missing_observations():
    response = parse_baseline(baseline_json())
    response.uncertainty = ["duplicate_models", "runtime_allocation", "Missing process memory measurements."]
    metrics = evaluate_baseline(response, {"app.log": "allocation observed"})
    assert metrics["uncertainty_count"] == 3
    assert metrics["missing_evidence_count"] == 1
