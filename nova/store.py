"""In-memory job / task / node store. PNG tiles land on disk under data_dir."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from nova.models import Job, NodeManifest, Task


class Store:
    """Coordinator gallery store. Objects are live; mutate then put_* to persist intent."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, Task] = {}
        self._nodes: dict[str, NodeManifest] = {}
        self._last_seen: dict[str, datetime] = {}
        self.tiles_dir = self.data_dir / "tiles"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

    # --- jobs ---

    def put_job(self, job: Job) -> Job:
        self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    # --- tasks ---

    def put_task(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, job_id: str | None = None) -> list[Task]:
        if job_id is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.job_id == job_id]

    # --- results ---

    def tile_path(self, job_id: str, task_id: str) -> Path:
        return self.data_dir / "tiles" / job_id / f"{task_id}.png"

    def save_result(self, task: Task, png_bytes: bytes, sha256: str) -> Path:
        """Decode PNG, check w/h if known, write `{data_dir}/tiles/{job_id}/{task_id}.png`."""
        _validate_png(png_bytes, width=task.width, height=task.height)
        digest = hashlib.sha256(png_bytes).hexdigest()
        if sha256 and sha256.lower() != digest:
            raise ValueError(f"sha256 mismatch: header {sha256} != body {digest}")
        path = self.tile_path(task.job_id, task.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes)
        task.result_path = str(path)
        task.result_sha256 = digest
        self.put_task(task)
        return path

    # --- nodes ---

    def put_node(self, node: NodeManifest) -> NodeManifest:
        self._nodes[node.node_id] = node
        return node

    def upsert_node(self, node: NodeManifest) -> NodeManifest:
        return self.put_node(node)

    def get_node(self, node_id: str) -> NodeManifest | None:
        return self._nodes.get(node_id)

    def list_nodes(self) -> list[NodeManifest]:
        return list(self._nodes.values())

    def touch_node(self, node_id: str, when: datetime) -> None:
        self._last_seen[node_id] = when

    def get_last_seen(self, node_id: str) -> datetime | None:
        return self._last_seen.get(node_id)

    # --- snapshots for the API / dashboard ---

    def job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        tasks = sorted(self.list_tasks(job_id), key=lambda t: t.shard_index)
        return {
            "job": job.model_dump(mode="json"),
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "completed": sum(1 for t in tasks if t.state == "COMPLETED"),
            "failed": sum(1 for t in tasks if t.state == "FAILED"),
            "total": len(tasks),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "jobs": [self.job_snapshot(j.job_id) for j in self.list_jobs()],
            "nodes": [n.model_dump(mode="json") for n in self.list_nodes()],
        }


def _validate_png(png_bytes: bytes, *, width: int | None, height: int | None) -> None:
    if not png_bytes:
        raise ValueError("empty result body")
    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("not a decodable PNG") from exc
    if img.format != "PNG":
        raise ValueError(f"not a PNG (got {img.format})")
    if width and img.width != width:
        raise ValueError(f"png width {img.width} != {width}")
    if height and img.height != height:
        raise ValueError(f"png height {img.height} != {height}")
