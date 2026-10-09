import json
from pathlib import Path
import subprocess
import sys

from app.main import analyze_core, main
from test_loader import DEMO


def test_bundled_core_matches_independent_expected_manifest():
    artifacts = analyze_core(DEMO)
    expected = json.loads((DEMO / "expected_evidence.json").read_text(encoding="utf-8"))
    evidence = {item["id"]: item for item in artifacts["evidence"]}
    assert sum(item["category"] != "derived_fact" for item in evidence.values()) == expected["observed_count"]
    assert sum(item["category"] == "derived_fact" for item in evidence.values()) == expected["derived_count"]
    for required in expected["required_evidence"]:
        assert required.items() <= evidence[required["id"]].items()
    for item in evidence.values():
        source = item["source"]
        lines = Path(source["file_path"]).read_text(encoding="utf-8").splitlines()
        assert source["raw_excerpt"] == "\n".join(lines[source["line_start"] - 1:source["line_end"]])
    for relation in artifacts["relations"]:
        assert relation["source_evidence_id"] in evidence
        assert relation["target_evidence_id"] in evidence
    parents = {}
    for relation in artifacts["relations"]:
        if relation["relation"] == "derived_from":
            parents.setdefault(relation["source_evidence_id"], set()).add(relation["target_evidence_id"])

    def original_inputs(key):
        if key not in parents:
            return {key}
        return set().union(*(original_inputs(parent) for parent in parents[key]))

    inputs = original_inputs("derived:exceeds_capacity")
    assert {"config.yaml:service.workers", "config.yaml:model.load_per_worker",
            "inference.py:L7:worker_model_load", "system.log:L1:memory",
            "system.log:L4:memory", "config.yaml:runtime.gpu_memory_limit_gb"} <= inputs
    assert artifacts == analyze_core(DEMO)
    assert "diagnosis" not in artifacts and "hypotheses" not in artifacts


def test_cli_demo_outputs_only_json_to_stdout():
    completed = subprocess.run([sys.executable, "-m", "app.main", "--demo", "--core-only"],
                               capture_output=True, text=True, check=True)
    artifacts = json.loads(completed.stdout)
    assert artifacts["phase"] == 1
    assert len(artifacts["evidence"]) == 25


def test_cli_reports_missing_directory_without_traceback(tmp_path, capsys):
    assert main(["--incident", str(tmp_path / "missing")]) == 2
    assert capsys.readouterr().out == ""


def test_cli_malformed_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("bad: [", encoding="utf-8")
    assert main(["--incident", str(tmp_path)]) == 2
