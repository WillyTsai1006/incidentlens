import pytest

from core.evidence_extractor import extract_code, extract_config, extract_evidence, extract_logs
from core.incident_loader import IncidentFile, load_incident
from test_loader import DEMO


def test_demo_config_values_and_exact_locations():
    files = load_incident(DEMO)
    items = extract_config(next(file for file in files if file.file_path.endswith("config.yaml")))
    by_id = {item.id: item for item in items}
    assert by_id["config.yaml:service.workers"].value == 4
    assert by_id["config.yaml:model.load_per_worker"].value is True
    assert by_id["config.yaml:model.precision"].value == "fp16"
    workers = by_id["config.yaml:service.workers"]
    assert workers.source.line_start == workers.source.line_end == 2
    assert workers.source.raw_excerpt == "  workers: 4"


@pytest.mark.parametrize("content", ["service: [", "- workers", "", "a: 1\na: 2", "a: [1]", "a: !!python/object:bad {}", "1: value", "a: &a {b: *a}"])
def test_invalid_or_unsupported_yaml(content):
    with pytest.raises(ValueError):
        extract_config(IncidentFile(file_path="config.yaml", content=content))


def test_quoted_config_and_multiline_provenance():
    items = extract_config(IncidentFile(file_path="config.yaml", content='a: "true"\nb: |\n  first\n  second\n'))
    assert items[0].value == "true"
    assert items[1].source.raw_excerpt == "b: |\n  first\n  second"


def test_worker_code_pattern_and_source():
    files = load_incident(DEMO)
    items = extract_code(next(file for file in files if file.file_path.endswith("inference.py")))
    assert len(items) == 1
    assert items[0].value == "worker_model_load"
    assert "model = load_model(MODEL_NAME)" in items[0].source.raw_excerpt
    assert items[0].source.line_start == 6


@pytest.mark.parametrize("content", [
    "model = load_model('x')\ndef worker_main(i):\n    serve(model)",
    "# def worker_main(i): load_model('x')",
    "def worker_main(i):\n    def unused():\n        load_model('x')",
])
def test_non_worker_loads_are_not_per_worker_evidence(content):
    assert extract_code(IncidentFile(file_path="inference.py", content=content)) == []


def test_invalid_python_has_clear_error():
    with pytest.raises(ValueError, match="Invalid Python"):
        extract_code(IncidentFile(file_path="inference.py", content="def broken(:"))


def test_demo_log_events_and_memory():
    items = extract_evidence(load_incident(DEMO))
    assert len([item for item in items if item.id.endswith(":worker_start")]) == 7
    assert [item.value for item in items if item.category == "system_metric"] == [0, 9.5, 19.0, 28.5]
    assert len([item for item in items if item.id.endswith(":oom")]) == 2
    assert len([item for item in items if item.id.endswith(":service_failure")]) == 1
    for item in items:
        assert item.source.line_start and item.source.line_end
        assert item.source.raw_excerpt
        assert item.confidence == 1


def test_unknown_lines_do_not_fabricate_events():
    assert extract_logs(IncidentFile(file_path="app.log", content="INFO healthy\nnot a metric")) == []


def test_invalid_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        extract_logs(IncidentFile(file_path="app.log", content="2026-13-08T09:00:00Z INFO worker-1 started"))
