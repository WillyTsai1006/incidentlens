"""Typed runtime measurements; unavailable metrics stay null."""

import platform
from typing import Literal

from pydantic import BaseModel, Field

from core.router import RoutingDecision, TaskType


class HardwareSnapshot(BaseModel):
    platform: str
    cpu: str | None = None
    logical_cpus: int | None = None
    host_total_memory_mb: float | None = None
    host_available_memory_mb: float | None = None
    device: str = "cpu"
    device_name: str | None = None
    device_available_memory_mb: float | None = None
    note: str = "Hardware metrics are measured when available; no B70 compatibility or performance is implied."


class InferenceCallTelemetry(BaseModel):
    call_index: int = Field(ge=1)
    task_type: TaskType
    backend: str
    model: str
    execution_mode: Literal["mock", "local", "stub"]
    metric_label: str
    input_chars: int = Field(ge=0)
    output_chars: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    inference_latency_ms: float = Field(ge=0)
    generation_latency_ms: float | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0)
    process_rss_mb: float | None = Field(default=None, ge=0)
    device_allocated_memory_mb: float | None = Field(default=None, ge=0)
    routing_overhead_ms: float = Field(ge=0)
    cold_start: bool | None = None
    status: Literal["completed", "failed"]
    error_type: str | None = None
    candidate_validation: Literal["not_checked", "valid", "invalid"] = "not_checked"
    routing: RoutingDecision


class RunTelemetry(BaseModel):
    end_to_end_latency_ms: float = Field(ge=0)
    routing_overhead_ms: float = Field(ge=0)
    inference_calls: list[InferenceCallTelemetry]
    hardware: HardwareSnapshot
    timing_scope: str = "Parsing through final diagnosis; artifact writing excluded. Inference call timing includes lazy model loading."
    memory_scope: str = "Process RSS and device allocated memory are snapshots, not peaks. GPU memory is null when not measured."


def process_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info().rss / 2**20
    except (psutil.Error, OSError):
        return None


def hardware_snapshot(device: str = "cpu") -> HardwareSnapshot:
    snapshot = HardwareSnapshot(platform=platform.platform(), cpu=platform.processor() or None, device=device)
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot.host_total_memory_mb = memory.total / 2**20
        snapshot.host_available_memory_mb = memory.available / 2**20
        snapshot.logical_cpus = psutil.cpu_count()
    except (ImportError, OSError):
        pass
    if device == "cpu":
        snapshot.device_name = snapshot.cpu
    else:
        try:
            import torch

            adapter = getattr(torch, device)
            if adapter.is_available():
                snapshot.device_name = adapter.get_device_name(0)
                free, _ = adapter.mem_get_info(0)
                snapshot.device_available_memory_mb = free / 2**20
        except (ImportError, AttributeError, OSError, RuntimeError):
            pass
    return snapshot
