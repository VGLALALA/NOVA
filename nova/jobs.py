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
    count: int | None = None,
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
    if count is not None:
        prompts = take_prompts(prompts, count)
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
        quotas=dict(job_block.get("quotas") or {}),
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
    apply_quotas(job, tasks)
    return tasks


def apply_quotas(job: Job, tasks: list[Task]) -> None:
    """Stamp reserved_node from job.quotas in shard order. Quota mode only."""
    if (job.scheduler_policy or "adaptive_pull") != "quota":
        return
    remaining: list[tuple[str, int]] = []
    for node_id, n in (job.quotas or {}).items():
        try:
            count = int(n)
        except (TypeError, ValueError):
            continue
        if count > 0 and node_id:
            remaining.append((str(node_id), count))
    idx = 0
    for task in tasks:
        while idx < len(remaining) and remaining[idx][1] <= 0:
            idx += 1
        if idx >= len(remaining):
            break
        node_id, left = remaining[idx]
        task.reserved_node = node_id
        remaining[idx] = (node_id, left - 1)


def load_and_split(
    path: str | Path,
    clock: Clock,
    *,
    job_id: str | None = None,
    count: int | None = None,
) -> tuple[Job, list[Task]]:
    job = load_gallery(path, job_id=job_id, clock=clock, count=count)
    return job, split_job(job, clock)


def take_prompts(prompts: list[PromptSpec], count: int) -> list[PromptSpec]:
    """First N prompts, cycling the list if the UI asks for more than yaml has."""
    n = int(count)
    if n < 1:
        raise ValueError("count must be >= 1")
    if n > 96:
        raise ValueError("count must be <= 96")
    if not prompts:
        raise ValueError("gallery has no prompts")
    if n <= len(prompts):
        return list(prompts[:n])
    out: list[PromptSpec] = []
    for i in range(n):
        src = prompts[i % len(prompts)]
        out.append(PromptSpec(text=src.text, seed=int(src.seed) + (i // len(prompts)) * 1000))
    return out
