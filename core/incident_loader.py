"""Read local UTF-8 incident sources without executing supplied code."""

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".log", ".yaml", ".yml", ".py", ".txt"}


class IncidentFile(BaseModel):
    file_path: str
    content: str


def load_incident(directory: str | Path) -> list[IncidentFile]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Incident directory not found: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported incident files in {directory}")
    return load_files(paths)


def load_files(paths: list[Path]) -> list[IncidentFile]:
    files = []
    for path in paths:
        path = Path(path)
        if path.suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        files.append(IncidentFile(file_path=str(path), content=path.read_text(encoding="utf-8")))
    logger.info("Loaded %d local incident sources", len(files))
    return files
