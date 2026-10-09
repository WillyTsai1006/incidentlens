"""Local, inspectable run snapshots; unique directories never overwrite runs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from core.incident_loader import IncidentFile


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]


def save_run(root: str | Path, run_id: str, artifacts: dict, inputs: list[IncidentFile]) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Invalid run ID")
    names = [Path(item.file_path).name for item in inputs]
    if len(names) != len(set(names)) or any(name in {"", ".", ".."} for name in names):
        raise ValueError("Input snapshot names must be unique safe basenames")
    # Serialize before creating a directory; NaN/Infinity are not JSON artifacts.
    serialized = {name: json.dumps(value, indent=2, allow_nan=False) for name, value in artifacts.items()}
    if any(Path(name).name != name or not name.endswith(".json") for name in serialized):
        raise ValueError("Artifact names must be JSON basenames")
    directory = Path(root).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    input_directory = directory / "inputs"
    input_directory.mkdir()
    for item, name in zip(inputs, names):
        (input_directory / name).write_text(item.content, encoding="utf-8", newline="")
    for name, content in serialized.items():
        (directory / name).write_text(content, encoding="utf-8", newline="")
    manifest = {str(path.relative_to(directory)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.rglob("*")) if path.is_file()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def verify_manifest(directory: str | Path) -> None:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Run artifact checksum mismatch or unsafe manifest path")
