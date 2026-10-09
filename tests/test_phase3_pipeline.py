import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.main import analyze_incident
from app.phase3 import Phase3RunError, run_phase3
from app.ui_presenter import presentation
from core.run_artifacts import verify_manifest
from inference.mock_backend import MockBackend
from test_loader import DEMO


def test_phase3_mock_pipeline_preserves_diagnosis_and_persists_complete_run(tmp_path):
    run = run_phase3(DEMO, save_root=tmp_path)
    assert run["phase"] == 3
    assert run["diagnosis"] == analyze_incident(DEMO)["diagnosis"]
    assert run["metadata"]["status"] == "completed"
    assert run["diagnosis"]["primary_root_cause"]["kind"] == "duplicate_models"
    assert len(run["routing"]) == len(run["telemetry"]["inference_calls"]) == 1
    assert run["telemetry"]["inference_calls"][0]["input_tokens"] is None
    assert run["telemetry"]["inference_calls"][0]["candidate_validation"] == "valid"
    assert all(r["relation"] != "causes" for r in run["relations"])
    directory = Path(run["artifact_directory"])
    expected = {"evidence.json", "relations.json", "hypotheses.json", "verification.json", "diagnosis.json",
                "routing.json", "telemetry.json", "metadata.json", "profiles.json", "inference_io.json", "manifest.json"}
    assert expected <= {path.name for path in directory.iterdir()}
    verify_manifest(directory)
    assert len(list((directory / "inputs").iterdir())) == 4
    replay = run_phase3(directory / "inputs", save_root=tmp_path)
    assert replay["diagnosis"]["primary_root_cause"]["kind"] == run["diagnosis"]["primary_root_cause"]["kind"]
    assert replay["diagnosis"]["confidence"] == run["diagnosis"]["confidence"]


def test_failure_is_saved_and_not_silently_replaced(tmp_path):
    class Invalid(MockBackend):
        def generate(self, prompt, **kwargs):
            return "unsupported output"

    with pytest.raises(Phase3RunError, match="HypothesisGenerationError"):
        run_phase3(DEMO, save_root=tmp_path, factory=lambda profile: Invalid())
    directory = next(tmp_path.iterdir())
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert json.loads((directory / "diagnosis.json").read_text()) is None
    assert len(json.loads((directory / "telemetry.json").read_text())["inference_calls"]) == 2
    assert all(c["candidate_validation"] == "invalid" for c in json.loads((directory / "telemetry.json").read_text())["inference_calls"])
    verify_manifest(directory)


def test_ui_serialization_observations_and_derived_facts_are_distinct(tmp_path):
    run = run_phase3(DEMO, persist=False)
    view = presentation(run)
    assert view["mode"] == "mock" and view["title"].startswith("Duplicate")
    assert {e["kind"] for e in view["evidence"]} == {"Observed evidence", "Derived fact"}
    assert view["verification_results"] and view["routing"] and view["telemetry"]
    assert json.loads(json.dumps(view))["confidence"] == 0.866667
    run["diagnosis"]["hypotheses"][0]["supporting_evidence_ids"].append("unknown")
    with pytest.raises(ValueError, match="unknown"):
        presentation(run)


def test_cli_competition_artifact_command(tmp_path):
    result = subprocess.run([sys.executable, "-m", "app.main", "--demo", "--save-run", "--runs-dir", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    run = json.loads(result.stdout)
    assert run["phase"] == 3 and Path(run["artifact_directory"]).is_dir()
    assert run["routing"][0]["context"]["task_type"] == "hypothesis_generation"


def test_parse_failure_preserves_failed_run(tmp_path):
    incident = tmp_path / "incident"
    incident.mkdir()
    (incident / "config.yaml").write_text("bad: [", encoding="utf-8")
    with pytest.raises(Phase3RunError):
        run_phase3(incident, save_root=tmp_path / "runs")
    directory = next((tmp_path / "runs").iterdir())
    assert (directory / "inputs/config.yaml").is_file()
    assert json.loads((directory / "metadata.json").read_text())["status"] == "failed"
