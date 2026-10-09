import json
from pathlib import Path
import subprocess
import sys

from app.main import analyze_incident, main
from core.diagnosis import diagnose
from core.evidence_repository import EvidenceRepository
from core.schemas import DiagnosisResult
from inference.mock_backend import MockBackend
from test_loader import DEMO


def test_full_pipeline_ranked_grounded_diagnosis(demo_graph):
    result = diagnose(*demo_graph, MockBackend())
    assert len(result.hypotheses) == 3
    assert result.primary_root_cause.kind == "duplicate_models"
    assert [h.confidence for h in result.hypotheses] == sorted((h.confidence for h in result.hypotheses), reverse=True)
    assert result.top_root_cause_id == result.primary_root_cause.id
    assert result.alternative_hypotheses == result.hypotheses[1:]
    assert result.primary_root_cause.verification_action and result.recommended_action
    assert result.confidence == result.primary_root_cause.confidence_metadata.score == 0.866667
    assert {Path(e.source.file_path).suffix for e in result.supporting_evidence} >= {".py", ".log", ".yaml"}
    assert result.evidence_grounding_rate == 1
    assert result.execution_mode == "mock" and result.inference_backend == "deterministic-mock"
    assert "not proven" in result.conclusion_status
    assert DiagnosisResult.model_validate_json(result.model_dump_json()) == result
    expected = json.loads((DEMO / "expected_diagnosis.json").read_text(encoding="utf-8"))
    assert result.primary_root_cause.kind == expected["primary_kind"]
    assert [h.kind for h in result.hypotheses] == expected["ranked_kinds"]
    assert expected["confidence_range"][0] <= result.confidence <= expected["confidence_range"][1]
    for h in result.hypotheses:
        for key in [*h.supporting_evidence_ids, *h.contradicting_evidence_ids]:
            demo_graph[0].get(key)
    for claim in result.factual_claims:
        assert claim.statement == demo_graph[0].get(claim.evidence_ids[0]).statement


def test_top_diagnosis_changes_to_leak_when_steady_state_growth_is_observed(tmp_path):
    (tmp_path / "config.yaml").write_text("service:\n  workers: 3\nmodel:\n  load_per_worker: true\n", encoding="utf-8")
    (tmp_path / "inference.py").write_text("def worker_main(i):\n    model = load_model('x')\n", encoding="utf-8")
    (tmp_path / "system.log").write_text("\n".join(
        f"2026-10-08T09:01:0{i}Z INFO phase=steady_state active_workers=3 gpu_memory_used_gib={value}"
        for i, value in enumerate([20, 22, 24])
    ), encoding="utf-8")
    (tmp_path / "app.log").write_text("2026-10-08T09:01:04Z ERROR phase=steady_state CUDA out of memory\n", encoding="utf-8")
    assert analyze_incident(tmp_path)["diagnosis"]["primary_root_cause"]["kind"] == "memory_leak"


def test_top_diagnosis_changes_to_runtime_allocation_from_raw_inputs(tmp_path):
    (tmp_path / "config.yaml").write_text("service:\n  workers: 1\nmodel:\n  load_per_worker: false\n", encoding="utf-8")
    (tmp_path / "system.log").write_text(
        "2026-10-08T09:00:02Z INFO batch_allocation_gib=12 batch_headroom_gib=4\n", encoding="utf-8")
    (tmp_path / "app.log").write_text(
        "2026-10-08T09:00:02Z ERROR phase=inference CUDA out of memory\n", encoding="utf-8")
    artifacts = analyze_incident(tmp_path)
    result = artifacts["diagnosis"]
    assert result["primary_root_cause"]["kind"] == "runtime_allocation"
    assert result["confidence"] > 0.8
    assert not any(e["id"] == "derived:per_worker_memory_gib" for e in artifacts["evidence"])


def test_no_evidence_does_not_claim_a_proven_root_cause():
    result = diagnose(EvidenceRepository(), [], MockBackend())
    assert all(h.confidence == 0 for h in result.hypotheses)
    assert "Insufficient evidence" in result.conclusion_status
    assert "Collect" in result.recommended_action
    assert result.evidence_grounding_rate == 0
    assert result.primary_root_cause is None and result.top_root_cause_id == ""


def test_shared_oom_evidence_alone_is_insufficient_to_select_a_root_cause(tmp_path):
    (tmp_path / "app.log").write_text("2026-10-08T09:00:02Z ERROR CUDA out of memory\n", encoding="utf-8")
    diagnosis = analyze_incident(tmp_path)["diagnosis"]
    assert diagnosis["primary_root_cause"] is None
    assert diagnosis["confidence"] == 0
    assert "Insufficient evidence" in diagnosis["conclusion_status"]
    assert len(diagnosis["alternative_hypotheses"]) == 3


def test_cli_complete_demo_and_determinism():
    completed = subprocess.run([sys.executable, "-m", "app.main", "--demo"],
                               capture_output=True, text=True, check=True)
    artifacts = json.loads(completed.stdout)
    assert artifacts["phase"] == 2 and len(artifacts["evidence"]) == 25
    assert artifacts["diagnosis"]["primary_root_cause"]["kind"] == "duplicate_models"
    assert not any(r["relation"] == "causes" for r in artifacts["relations"])
    assert artifacts == analyze_incident(DEMO)


def test_cli_local_backend_missing_model_has_clear_error(capsys):
    assert main(["--demo", "--backend", "transformers"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_unavailable_local_model_never_falls_back_to_mock(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "app.main", "--demo", "--backend", "transformers",
                                "--model-path", str(tmp_path / "missing")], capture_output=True, text=True)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "no model will be downloaded" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_invalid_model_json_fails_without_emitting_diagnosis(monkeypatch, capsys):
    class BrokenBackend(MockBackend):
        def generate(self, prompt, **kwargs):
            return '{"unsupported": "answer"}'

    monkeypatch.setattr("app.main.MockBackend", BrokenBackend)
    assert main(["--demo"]) == 2
    assert capsys.readouterr().out == ""


def test_mixed_batch_observations_are_reported_and_penalized(tmp_path):
    (tmp_path / "system.log").write_text(
        "2026-10-08T09:00:02Z INFO batch_allocation_gib=12 batch_headroom_gib=4\n"
        "2026-10-08T09:00:03Z INFO batch_allocation_gib=2 batch_headroom_gib=4\n", encoding="utf-8")
    (tmp_path / "app.log").write_text(
        "2026-10-08T09:00:02Z ERROR phase=inference CUDA out of memory\n", encoding="utf-8")
    hypothesis = next(h for h in analyze_incident(tmp_path)["diagnosis"]["hypotheses"]
                      if h["kind"] == "runtime_allocation")
    batch = next(r for r in hypothesis["verification_results"] if r["criterion_id"] == "batch_pressure")
    assert batch["status"] == "mixed"
    assert len(batch["supporting_evidence_ids"]) == len(batch["contradicting_evidence_ids"]) == 2
    assert hypothesis["confidence_metadata"]["contradiction_penalty"] == 0.2
    assert hypothesis["confidence"] == 0.55
