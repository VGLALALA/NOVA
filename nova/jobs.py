"""Gallery splitter: yaml → Job → Tasks."""

from __future__ import annotations

import uuid
from pathlib import Path

import yaml

from nova.clock import Clock, now_utc
from nova.models import Job, JobRequirements, PromptSpec, Task

DEFAULT_MAX_ATTEMPTS = 3


def load_gallery(
    path: str | Path,
    *,
    job_id: str | None = None,
    clock: Clock | None = None,
) -> Job:
    """Load a committed gallery yaml into a Job. Same yaml → same 24 slots."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"gallery yaml must be a mapping: {path}")

    job_block = raw.get("job") or {}
    model_block = raw.get("model") or {}
    req_block = raw.get("requirements") or {}
    prompt_rows = raw.get("prompts") or []

    name = str(job_block.get("name") or "nova-gallery")
    requirements = JobRequirements(
        min_memory_mb=int(req_block.get("min_memory_mb", 4096)),
        allowed_backends=list(req_block.get("allowed_backends") or ["cuda", "rocm", "metal"]),
        width=int(model_block.get("width", 512)),
        height=int(model_block.get("height", 512)),
        steps=int(model_block.get("steps", 4)),
        model_id=str(model_block.get("id") or "stabilityai/sd-turbo"),
    )
    prompts = [
        PromptSpec(text=str(row["text"]), seed=int(row["seed"]))
        for row in prompt_rows
    ]
    created = clock.now() if clock is not None else now_utc()
    resolved_id = job_id or str(job_block.get("id") or f"{name}-{uuid.uuid4().hex[:8]}")
    return Job(
        job_id=resolved_id,
        name=name,
        job_type=str(job_block.get("type") or "sd_gallery"),
        kernel_id=str(job_block.get("kernel") or "sd.t2i.v1"),
        created_at=created,
        state="QUEUED",
        scheduler_policy=str(job_block.get("scheduler_policy") or "adaptive_pull"),
        requirements=requirements,
        prompts=prompts,
    )


def split_job(job: Job, clock: Clock) -> list[Task]:
    """One Task per prompt. shard_index 0..N-1. Requirements copied from the job."""
    now = clock.now()
    req = job.requirements
    tasks: list[Task] = []
    for shard_index, spec in enumerate(job.prompts):
        tasks.append(
            Task(
                task_id=f"{job.job_id}-{shard_index:02d}",
                job_id=job.job_id,
                shard_index=shard_index,
                kernel_id=job.kernel_id,
                prompt=spec.text,
                seed=spec.seed,
                steps=req.steps,
                width=req.width,
                height=req.height,
                min_memory_mb=req.min_memory_mb,
                allowed_backends=list(req.allowed_backends),
                state="QUEUED",
                attempt_count=0,
                max_attempts=DEFAULT_MAX_ATTEMPTS,
                created_at=now,
            )
        )
    return tasks


def load_and_split(
    path: str | Path,
    clock: Clock,
    *,
    job_id: str | None = None,
) -> tuple[Job, list[Task]]:
    job = load_gallery(path, job_id=job_id, clock=clock)
    return job, split_job(job, clock)
