from pathlib import Path
import os

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "app/ui.py"


@pytest.fixture(autouse=True)
def deterministic_cpu_defaults(monkeypatch):
    # Competition defaults are hardware-aware; tests explicitly simulate CPU-only.
    monkeypatch.setattr("app.ui_runtime.detect_cuda", lambda: dict(available=False, name=None,
                                                                 warning="CUDA unavailable; CPU selected explicitly."))
    streamlit.cache_data.clear()


def test_streamlit_demo_load_analyze_and_render(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_RUNS_DIR", str(tmp_path))
    app = AppTest.from_file(str(UI)).run(timeout=30)
    assert not app.exception
    app.selectbox(key="mode").select("Mock (deterministic)").run()
    app.button(key="load_demo").click().run()
    assert len(app.session_state["inputs"]) == 4
    app.button(key="analyze").click().run(timeout=30)
    assert not app.exception
    run = app.session_state["run"]
    assert run["diagnosis"]["primary_root_cause"]["kind"] == "duplicate_models"
    assert any("MOCK" in message.value for message in app.info)
    assert any("**Actual device:** No model executed (mock)" in text.value for text in app.markdown)
    assert len(app.dataframe) >= 4
    assert Path(run["artifact_directory"]).is_dir()
    assert any(metric.value == "86.7%" for metric in app.metric)
    assert app.title[0].value == "IncidentLens"
    metrics = {metric.label: metric.value for metric in app.metric}
    assert {key: metrics[key] for key in ("Supported", "Contradicted", "Mixed", "Missing")} == {
        "Supported": "5", "Contradicted": "0", "Mixed": "0", "Missing": "1"}
    assert metrics["GPU Memory Snapshot"] == "N/A" and "Peak GPU Memory" not in metrics
    before = app.session_state["run"]
    app.button(key="upload_incident").click().run()
    assert app.session_state["run"] == before


def test_home_capabilities_only_before_incident_loading():
    app = AppTest.from_file(str(UI)).run(timeout=30)
    titles = ("Evidence Grounded", "Verification First", "Private Local AI")
    assert all(any(title in text.value for text in app.markdown) for title in titles)
    assert any("Every diagnosis is linked to traceable evidence." == text.value for text in app.caption)
    assert any("Hypotheses are tested before ranking." == text.value for text in app.caption)
    assert any("Engineering data stays on local hardware." == text.value for text in app.caption)
    app.button(key="load_demo").click().run()
    assert not any(title in text.value for title in titles for text in app.markdown)


def test_competition_defaults_cuda_and_settings_are_collapsed(monkeypatch):
    monkeypatch.setattr("app.ui_runtime.detect_cuda", lambda: dict(available=True, name="Simulated CUDA device", warning=None))
    streamlit.cache_data.clear()
    app = AppTest.from_file(str(UI)).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="mode").value == "Local Transformers (real)"
    assert app.selectbox(key="device").value == "cuda"
    assert app.expander[0].label == "Advanced Runtime Settings" and not app.expander[0].proto.expanded
    app.button(key="load_demo").click().run()
    assert len(app.session_state["inputs"]) == 4
    assert "run" not in app.session_state


def test_cpu_fallback_visible_and_upload_action_keeps_controls_available():
    app = AppTest.from_file(str(UI)).run(timeout=30)
    assert app.selectbox(key="device").value == "cpu"
    assert any("CUDA unavailable" in warning.value for warning in app.warning)
    app.button(key="upload_incident").click().run()
    assert not app.exception and app.get("file_uploader")


def test_streamlit_missing_local_model_shows_error_no_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_RUNS_DIR", str(tmp_path / "runs"))
    app = AppTest.from_file(str(UI)).run(timeout=30)
    app.button(key="load_demo").click().run()
    app.selectbox(key="mode").select("Local Transformers (real)")
    app.text_input(key="model_path").set_value(os.environ["INCIDENT_MODEL_PATH"])
    app.text_input(key="model_path").set_value(str(tmp_path / "missing"))
    app.button(key="analyze").click().run(timeout=30)
    assert not app.exception
    assert app.error and "InferenceUnavailable" in app.error[0].value
    assert "run" not in app.session_state


@pytest.mark.real_model
@pytest.mark.skipif(not os.environ.get("INCIDENT_MODEL_PATH"), reason="Set INCIDENT_MODEL_PATH for optional real Streamlit inference")
def test_streamlit_real_local_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_RUNS_DIR", str(tmp_path))
    app = AppTest.from_file(str(UI)).run(timeout=30)
    app.button(key="load_demo").click().run()
    app.selectbox(key="mode").select("Local Transformers (real)")
    app.button(key="analyze").click().run(timeout=240)
    assert not app.exception and not app.error
    run = app.session_state["run"]
    assert run["diagnosis"]["execution_mode"] == "local"
    assert run["diagnosis"]["primary_root_cause"]["kind"] == "duplicate_models"
    assert len(run["telemetry"]["inference_calls"]) >= 3
    assert all(c["candidate_validation"] == "valid" for c in run["telemetry"]["inference_calls"])
    assert any("REAL local" in message.value for message in app.info)
