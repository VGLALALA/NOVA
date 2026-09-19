"""Shared domain models. Single source of truth for jobs, tasks, nodes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Backend = Literal["cuda", "rocm", "metal", "cpu"]
Vendor = Literal["nvidia", "amd", "apple", "cpu"]
NodeStatus = Literal["online", "suspect", "offline"]
JobState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
TaskState = Literal["QUEUED", "LEASED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]

KERNEL_SD_T2I = "sd.t2i.v1"


class Device(BaseModel):
    device_id: str  # "cuda:0" | "mps:0" | "cpu:0"
    backend: Backend
    vendor: Vendor
    model: str
    memory_total_mb: int
    memory_free_mb: int = 0
    load: float = 0.0


class NodeManifest(BaseModel):
    node_id: str
    hostname: str = ""
    os: str = ""
    architecture: str = ""
    devices: list[Device] = Field(default_factory=list)
    supported_kernels: list[str] = Field(default_factory=lambda: [KERNEL_SD_T2I])
    benchmark_scores: dict[str, float] = Field(default_factory=dict)
    max_concurrency: int = 1
    current_slots_used: int = 0
    status: NodeStatus = "online"
    http_url: str | None = None  # coordinator announces; workers cache this


class PromptSpec(BaseModel):
    text: str
    seed: int


class JobRequirements(BaseModel):
    min_memory_mb: int = 4096
    allowed_backends: list[Backend] = Field(default_factory=lambda: ["cuda", "rocm", "metal"])
    width: int = 512
    height: int = 512
    steps: int = 4
    model_id: str = "stabilityai/sd-turbo"


class Job(BaseModel):
    job_id: str
    name: str = "nova-gallery"
    job_type: str = "sd_gallery"
    kernel_id: str = KERNEL_SD_T2I
    created_at: datetime
    state: JobState = "QUEUED"
    scheduler_policy: str = "adaptive_pull"
    requirements: JobRequirements = Field(default_factory=JobRequirements)
    prompts: list[PromptSpec] = Field(default_factory=list)


class Task(BaseModel):
    task_id: str
    job_id: str
    shard_index: int
    kernel_id: str = KERNEL_SD_T2I
    prompt: str
    seed: int
    steps: int = 4
    width: int = 512
    height: int = 512
    min_memory_mb: int = 4096
    allowed_backends: list[Backend] = Field(default_factory=lambda: ["cuda", "rocm", "metal"])
    state: TaskState = "QUEUED"
    assigned_node: str | None = None
    lease_gen: int = 0
    lease_expires_at: datetime | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_sha256: str | None = None
    result_path: str | None = None
    backend_used: str | None = None
    device_id_used: str | None = None
    execution_ms: int | None = None
    last_failed_node: str | None = None  # one-cycle blacklist


class KernelResult(BaseModel):
    png_bytes: bytes
    sha256: str
    execution_ms: int
    device_id: str
    backend: Backend


class NodeIdentity(BaseModel):
    node_id: str
    created_at: datetime
