from pathlib import Path

import pytest

from core.incident_loader import load_files, load_incident

DEMO = Path(__file__).resolve().parents[1] / "examples/gpu_oom_duplicate_models"


def test_load_demo_sources_only():
    files = load_incident(DEMO)
    assert [Path(file.file_path).name for file in files] == [
        "app.log", "config.yaml", "inference.py", "system.log"
    ]
    assert all(file.content for file in files)


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_files([tmp_path / "missing.log"])
    with pytest.raises(FileNotFoundError):
        load_incident(tmp_path / "missing")


def test_unsupported_and_empty_input(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        load_files([tmp_path / "file.exe"])
    with pytest.raises(ValueError, match="No supported"):
        load_incident(tmp_path)


def test_loading_never_executes_python(tmp_path):
    path = tmp_path / "inference.py"
    path.write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    assert load_files([path])[0].content.startswith("raise")
