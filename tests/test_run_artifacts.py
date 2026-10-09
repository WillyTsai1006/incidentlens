import hashlib
import json
from pathlib import Path

import pytest

from core.incident_loader import IncidentFile
from core.run_artifacts import new_run_id, save_run, verify_manifest


def test_timestamped_unique_run_ids():
    first, second = new_run_id(), new_run_id()
    assert first != second and "T" in first and "Z-" in first


def test_artifacts_snapshots_checksums_and_no_overwrite(tmp_path):
    source = IncidentFile(file_path="outside/app.log", content="local event\n")
    directory = save_run(tmp_path, new_run_id(), {"metadata.json": {"status": "completed"}}, [source])
    assert (directory / "inputs/app.log").read_text(encoding="utf-8") == source.content
    assert json.loads((directory / "metadata.json").read_text()) == {"status": "completed"}
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["inputs/app.log"] == hashlib.sha256(source.content.encode()).hexdigest()
    verify_manifest(directory)
    with pytest.raises(FileExistsError):
        save_run(tmp_path, directory.name, {}, [])
    (directory / "inputs/app.log").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(directory)


@pytest.mark.parametrize("run_id", ["../outside", "..", ""])
def test_unsafe_run_ids_rejected(tmp_path, run_id):
    with pytest.raises(ValueError):
        save_run(tmp_path, run_id, {}, [])


def test_duplicate_inputs_and_unsafe_artifact_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        save_run(tmp_path, new_run_id(), {}, [IncidentFile(file_path="a/app.log", content="1"), IncidentFile(file_path="b/app.log", content="2")])
    with pytest.raises(ValueError, match="basenames"):
        save_run(tmp_path, new_run_id(), {"../outside.json": {}}, [])


def test_nonfinite_json_does_not_leave_a_run_directory(tmp_path):
    run_id = new_run_id()
    with pytest.raises(ValueError):
        save_run(tmp_path, run_id, {"bad.json": {"metric": float("nan")}}, [])
    assert not (tmp_path / run_id).exists()
