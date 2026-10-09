"""Presentation defaults and profiles; domain routing stays unchanged."""
import json
from pathlib import Path
from core.router import ModelProfile
from app.phase3 import default_profiles

ROOT = Path(__file__).resolve().parents[1]
DEMO_MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"


def detect_cuda() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return dict(available=True, name=torch.cuda.get_device_name(0), warning=None)
    except (ImportError, OSError, RuntimeError) as exc:
        return dict(available=False, name=None, warning=f"CUDA unavailable ({type(exc).__name__}); CPU selected explicitly.")
    return dict(available=False, name=None, warning="CUDA unavailable; CPU selected explicitly. No GPU inference is implied.")


def ui_profiles(backend: str, model_path: str | None, device: str) -> list[ModelProfile]:
    if backend == "transformers" and device == "cuda" and Path(model_path).resolve() == DEMO_MODEL.resolve():
        profiles = json.loads((ROOT / "profiles/demo_rtx5060.json").read_text(encoding="utf-8"))
        return [ModelProfile.model_validate(p) for p in profiles]
    return default_profiles(backend, model_path, device)
