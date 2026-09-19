"""Shared fakes and helpers for NOVA failover tests.

Prefers real `nova.scheduler` / `nova.store` / `nova.network.inprocess` when
they import. Fallback fakes implement the same live-demo contract:

- NODE_DISCONNECTED requeues LEASED/RUNNING immediately (no 30s wait)
- Heartbeat does not renew a task lease; TASK_PROGRESS does
- Stale lease_gen COMPLETE / PUT is ignored
- attempt_count increments on grant, not on disconnect
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from nova.clock import Clock
from nova.models import Device, Job, JobRequirements, NodeManifest, PromptSpec, Task

KERNEL = "sd.t2i.v1"
IN_FLIGHT = frozenset({"LEASED", "RUNNING"})


def dummy_png(width: int = 64, height: int = 64, color: tuple[int, int, int] = (12, 24, 48)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_node(
    node_id: str,
    *,
    backend: str = "cuda",
    vendor: str = "nvidia",
    model: str = "RTX 4090",
    memory_total_mb: int = 24576,
) -> NodeManifest:
    return NodeManifest(
        node_id=node_id,
        hostname=node_id,
        devices=[
            Device(
                device_id=f"{backend}:0",
                backend=backend,  # type: ignore[arg-type]
                vendor=vendor,  # type: ignore[arg-type]
                model=model,
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_total_mb // 2,
            )
        ],
        supported_kernels=[KERNEL],
        benchmark_scores={KERNEL: 0.5},
        max_concurrency=1,
        current_slots_used=0,
        status="online",
    )


def dummy_job(clock: Clock, n: int = 8, job_id: str = "failover") -> Job:
    now = clock.now()
    return Job(
        job_id=job_id,
        name="failover-gallery",
        created_at=now,
        requirements=JobRequirements(
            min_memory_mb=4096,
            allowed_backends=["cuda", "rocm", "metal"],
            width=64,
            height=64,
            steps=4,
        ),
        prompts=[PromptSpec(text=f"dummy tile {i}", seed=1000 + i) for i in range(n)],
    )


class FakeEventBus:
    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        event = {"type": type_, **fields}
        self._history.append(event)
        return event

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)


class FakeStore:
    """In-memory store matching nova.store.Store's surface used by tests."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, Task] = {}
        self._nodes: dict[str, NodeManifest] = {}
        self._last_seen: dict[str, datetime] = {}

    def put_job(self, job: Job) -> Job:
        self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def put_task(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, job_id: str | None = None) -> list[Task]:
        if job_id is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.job_id == job_id]

    def tile_path(self, job_id: str, task_id: str) -> Path:
        return self.data_dir / "tiles" / job_id / f"{task_id}.png"

    def save_result(self, task: Task, png_bytes: bytes, sha256: str) -> Path:
        if not png_bytes:
            raise ValueError("empty result body")
        try:
            img = Image.open(io.BytesIO(png_bytes))
            img.load()
        except Exception as exc:
            raise ValueError("not a decodable PNG") from exc
        if img.format != "PNG":
            raise ValueError(f"not a PNG (got {img.format})")
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

    def put_node(self, node: NodeManifest) -> NodeManifest:
        self._nodes[node.node_id] = node
        return node

    def get_node(self, node_id: str) -> NodeManifest | None:
        return self._nodes.get(node_id)

    def list_nodes(self) -> list[NodeManifest]:
        return list(self._nodes.values())

    def touch_node(self, node_id: str, when: datetime) -> None:
        self._last_seen[node_id] = when

    def get_last_seen(self, node_id: str) -> datetime | None:
        return self._last_seen.get(node_id)


class FakeScheduler:
    """Fallback scheduler. Same method names as nova.scheduler.Scheduler."""

    def __init__(
        self,
        store: FakeStore,
        clock: Clock,
        event_bus: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self.event_bus = event_bus
        self.settings = settings

    def submit_job(self, job: Job) -> list[Task]:
        self.store.put_job(job)
        try:
            from nova.jobs import split_job

            tasks = split_job(job, self.clock)
        except ImportError:
            tasks = _split_job(job, self.clock)
        for task in tasks:
            self.store.put_task(task)
        self._emit("JOB_ANNOUNCE", job_id=job.job_id, n_tasks=len(tasks))
        return tasks

    def register_node(self, manifest: NodeManifest) -> NodeManifest:
        existing = self.store.get_node(manifest.node_id)
        if existing is not None and existing.status != "offline":
            manifest.current_slots_used = existing.current_slots_used
        else:
            manifest.current_slots_used = 0
        manifest.status = "online"
        self.store.put_node(manifest)
        self.store.touch_node(manifest.node_id, self.clock.now())
        self._emit("NODE_MANIFEST", node_id=manifest.node_id)
        return manifest

    def on_work_request(self, node_id: str, available_slots: int | None = None) -> Task | None:
        node = self.store.get_node(node_id)
        if node is None or node.status != "online":
            return None
        if available_slots is not None and available_slots <= 0:
            return None
        self.store.touch_node(node_id, self.clock.now())
        if node.current_slots_used >= node.max_concurrency:
            return None
        task = self._pick(node)
        if task is None:
            return None
        return self._grant(node_id, task)

    def on_heartbeat(self, node_id: str) -> None:
        node = self.store.get_node(node_id)
        if node is None:
            return
        self.store.touch_node(node_id, self.clock.now())
        if node.status == "suspect":
            node.status = "online"
            self.store.put_node(node)

    def on_progress(
        self,
        node_id: str,
        task_id: str,
        lease_gen: int,
        lease_seconds: int | None = None,
    ) -> bool:
        task = self.store.get_task(task_id)
        if not _lease_matches(task, node_id, lease_gen):
            return False
        assert task is not None
        seconds = lease_seconds if lease_seconds is not None else 20
        task.lease_expires_at = self.clock.now() + timedelta(seconds=seconds)
        if task.state == "LEASED":
            task.state = "RUNNING"
            if task.started_at is None:
                task.started_at = self.clock.now()
        self.store.put_task(task)
        return True

    def on_complete(
        self,
        node_id: str,
        task_id: str,
        lease_gen: int,
        sha256: str,
        path: str,
        backend: str,
        *,
        device_id: str | None = None,
        execution_ms: int | None = None,
    ) -> bool:
        task = self.store.get_task(task_id)
        if not _lease_matches(task, node_id, lease_gen):
            return False
        assert task is not None
        _release_slot(self.store, task)
        task.state = "COMPLETED"
        task.assigned_node = node_id
        task.lease_expires_at = None
        task.completed_at = self.clock.now()
        task.result_sha256 = sha256
        task.result_path = path
        task.backend_used = backend
        task.device_id_used = device_id
        task.execution_ms = execution_ms
        self.store.put_task(task)
        _refresh_job(self.store, task.job_id)
        return True

    def accept_result(
        self,
        node_id: str,
        task_id: str,
        lease_gen: int,
        png_bytes: bytes,
        sha256: str,
        backend: str,
        *,
        device_id: str | None = None,
        execution_ms: int | None = None,
    ) -> bool:
        task = self.store.get_task(task_id)
        if not _lease_matches(task, node_id, lease_gen):
            return False
        assert task is not None
        path = self.store.save_result(task, png_bytes, sha256)
        return self.on_complete(
            node_id,
            task_id,
            lease_gen,
            sha256 or task.result_sha256 or "",
            str(path),
            backend,
            device_id=device_id,
            execution_ms=execution_ms,
        )

    def on_disconnect(self, node_id: str) -> list[Task]:
        node = self.store.get_node(node_id)
        if node is not None:
            node.status = "offline"
            self.store.put_node(node)
        affected: list[Task] = []
        for task in self.store.list_tasks():
            if task.assigned_node == node_id and task.state in IN_FLIGHT:
                self._requeue_or_fail(task, "disconnected")
                affected.append(task)
        self._emit("NODE_DISCONNECTED", node_id=node_id, requeued=[t.task_id for t in affected])
        return affected

    def expire_leases(self) -> list[Task]:
        now = self.clock.now()
        expired: list[Task] = []
        for task in self.store.list_tasks():
            if task.state not in IN_FLIGHT:
                continue
            if task.lease_expires_at is None or task.lease_expires_at > now:
                continue
            self._requeue_or_fail(task, "expired")
            expired.append(task)
        return expired

    def check_node_liveness(self, suspect_s: float = 15.0, offline_s: float = 30.0) -> list[str]:
        now = self.clock.now()
        offlined: list[str] = []
        for node in self.store.list_nodes():
            if node.status == "offline":
                continue
            seen = self.store.get_last_seen(node.node_id)
            if seen is None:
                continue
            age = (now - seen).total_seconds()
            if age >= offline_s:
                self.on_disconnect(node.node_id)
                offlined.append(node.node_id)
            elif age >= suspect_s and node.status != "suspect":
                node.status = "suspect"
                self.store.put_node(node)
        return offlined

    def tick(self) -> None:
        self.expire_leases()
        self.check_node_liveness()

    def _pick(self, node: NodeManifest) -> Task | None:
        queued = [t for t in self.store.list_tasks() if t.state == "QUEUED" and _compatible(node, t)]
        queued.sort(key=lambda t: (t.created_at, t.shard_index))
        return queued[0] if queued else None

    def _grant(self, node_id: str, task: Task) -> Task:
        task.lease_gen += 1
        task.attempt_count += 1
        task.state = "LEASED"
        task.assigned_node = node_id
        task.lease_expires_at = self.clock.now() + timedelta(seconds=20)
        self.store.put_task(task)
        node = self.store.get_node(node_id)
        if node is not None:
            node.current_slots_used += 1
            self.store.put_node(node)
        _refresh_job(self.store, task.job_id)
        return task

    def _requeue_or_fail(self, task: Task, reason: str) -> None:
        if task.state not in IN_FLIGHT:
            return
        _release_slot(self.store, task)
        task.assigned_node = None
        task.lease_expires_at = None
        task.state = "QUEUED" if task.attempt_count < task.max_attempts else "FAILED"
        self.store.put_task(task)
        _refresh_job(self.store, task.job_id)
        self._emit("TASK_REQUEUED" if task.state == "QUEUED" else "TASK_FAILED", task_id=task.task_id, reason=reason)

    def _emit(self, type_: str, **fields: Any) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(type_, **fields)


class FakeInProcessBroker:
    def __init__(self) -> None:
        self._transports: dict[str, FakeInProcessTransport] = {}

    def register(self, transport: FakeInProcessTransport) -> list[FakeInProcessTransport]:
        others = [t for t in self._transports.values() if t.node_id != transport.node_id]
        self._transports[transport.node_id] = transport
        return others

    def unregister(self, node_id: str) -> list[FakeInProcessTransport]:
        self._transports.pop(node_id, None)
        return list(self._transports.values())

    def get(self, node_id: str) -> FakeInProcessTransport | None:
        return self._transports.get(node_id)


class FakeInProcessTransport:
    """Ctrl+C stand-in: stop() fires on_peer_disconnect on remaining peers."""

    def __init__(self, broker: FakeInProcessBroker, node_id: str) -> None:
        self.broker = broker
        self.node_id = node_id
        self._started = False
        self._on_disconnect: Any = None

    def on_peer_disconnect(self, handler: Any) -> None:
        self._on_disconnect = handler

    def on_peer_connect(self, handler: Any) -> None:
        return None

    def on_message(self, handler: Any) -> None:
        return None

    async def start(self) -> None:
        if self._started:
            return
        self.broker.register(self)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        others = self.broker.unregister(self.node_id)
        for other in others:
            if other._on_disconnect is None:
                continue
            res = other._on_disconnect(self.node_id)
            if hasattr(res, "__await__"):
                await res

    async def send(self, peer_id: str, env: Any) -> None:
        return None

    async def broadcast(self, env: Any) -> None:
        return None


class FakePullWorker:
    """Pull-client stand-in. No GPU. Completes with dummy PNGs via the scheduler."""

    def __init__(
        self,
        node_id: str,
        scheduler: Any,
        *,
        backend: str = "cuda",
        vendor: str = "nvidia",
        model: str = "RTX 4090",
        transport: Any | None = None,
    ) -> None:
        self.node_id = node_id
        self.scheduler = scheduler
        self.backend = backend
        self.transport = transport
        self.in_flight: Task | None = None
        self.held_lease_gen: int | None = None
        self.completed: list[str] = []
        scheduler.register_node(make_node(node_id, backend=backend, vendor=vendor, model=model))

    def pull(self) -> Task | None:
        task = self.scheduler.on_work_request(self.node_id)
        self.in_flight = task
        self.held_lease_gen = task.lease_gen if task is not None else None
        return task

    def start_running(self) -> bool:
        if self.in_flight is None or self.held_lease_gen is None:
            return False
        return bool(
            self.scheduler.on_progress(self.node_id, self.in_flight.task_id, self.held_lease_gen)
        )

    def complete(self, color: tuple[int, int, int] = (10, 20, 30)) -> bool:
        task = self.in_flight
        if task is None or self.held_lease_gen is None:
            return False
        png = dummy_png(task.width, task.height, color=color)
        digest = hashlib.sha256(png).hexdigest()
        ok = bool(
            self.scheduler.accept_result(
                self.node_id, task.task_id, self.held_lease_gen, png, digest, self.backend
            )
        )
        if ok:
            self.completed.append(task.task_id)
            self.in_flight = None
            self.held_lease_gen = None
        return ok

    def complete_stale(self, task_id: str, lease_gen: int) -> bool:
        task = self.scheduler.store.get_task(task_id)
        width = task.width if task is not None else 64
        height = task.height if task is not None else 64
        png = dummy_png(width, height, color=(99, 0, 0))
        digest = hashlib.sha256(png).hexdigest()
        return bool(
            self.scheduler.accept_result(self.node_id, task_id, lease_gen, png, digest, self.backend)
        )

    def drain(self, limit: int = 32) -> int:
        n = 0
        if self.in_flight is not None:
            if not self.complete():
                raise AssertionError(f"{self.node_id} failed to complete in-flight task")
            n += 1
        while n < limit:
            if self.pull() is None:
                break
            if not self.complete():
                raise AssertionError(f"{self.node_id} failed to complete pulled task")
            n += 1
        return n

    async def kill(self) -> None:
        """Simulate Ctrl+C / connection_lost. Uses transport.stop when wired."""
        if self.transport is not None:
            await self.transport.stop()
        else:
            self.scheduler.on_disconnect(self.node_id)
        self.in_flight = None


def resolve_stack() -> SimpleNamespace:
    """Prefer real modules; fall back to fakes in this file."""
    try:
        from nova.scheduler import Scheduler
        from nova.store import Store

        using_real_scheduler = True
    except ImportError:
        Scheduler = FakeScheduler  # type: ignore[assignment,misc]
        Store = FakeStore  # type: ignore[assignment,misc]
        using_real_scheduler = False

    try:
        from nova.network.inprocess import InProcessBroker, InProcessTransport

        using_real_inprocess = True
    except ImportError:
        InProcessBroker = FakeInProcessBroker  # type: ignore[assignment,misc]
        InProcessTransport = FakeInProcessTransport  # type: ignore[assignment,misc]
        using_real_inprocess = False

    try:
        from nova.events import EventBus
    except ImportError:
        EventBus = FakeEventBus  # type: ignore[assignment,misc]

    return SimpleNamespace(
        Scheduler=Scheduler,
        Store=Store,
        InProcessBroker=InProcessBroker,
        InProcessTransport=InProcessTransport,
        EventBus=EventBus,
        using_real_scheduler=using_real_scheduler,
        using_real_inprocess=using_real_inprocess,
    )


def _compatible(node: NodeManifest, task: Task) -> bool:
    if node.status != "online":
        return False
    if task.kernel_id not in node.supported_kernels:
        return False
    if node.current_slots_used >= node.max_concurrency:
        return False
    allowed = set(task.allowed_backends)
    for device in node.devices:
        if device.backend in allowed and device.memory_total_mb >= task.min_memory_mb:
            return True
    return False


def _lease_matches(task: Task | None, node_id: str, lease_gen: int) -> bool:
    if task is None:
        return False
    if task.state not in IN_FLIGHT:
        return False
    if task.assigned_node != node_id:
        return False
    if task.lease_gen != lease_gen:
        return False
    return True


def _release_slot(store: FakeStore, task: Task) -> None:
    node_id = task.assigned_node
    if not node_id:
        return
    node = store.get_node(node_id)
    if node is None:
        return
    node.current_slots_used = max(0, node.current_slots_used - 1)
    store.put_node(node)


def _refresh_job(store: FakeStore, job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None or job.state == "CANCELLED":
        return
    tasks = store.list_tasks(job_id)
    if not tasks:
        return
    states = [t.state for t in tasks]
    if all(s == "COMPLETED" for s in states):
        job.state = "COMPLETED"
    elif all(s in {"COMPLETED", "FAILED", "CANCELLED"} for s in states):
        job.state = "FAILED" if any(s == "FAILED" for s in states) else "CANCELLED"
    elif any(s != "QUEUED" for s in states) or any(t.attempt_count > 0 for t in tasks):
        job.state = "RUNNING"
    else:
        job.state = "QUEUED"
    store.put_job(job)


def _split_job(job: Job, clock: Clock) -> list[Task]:
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
                max_attempts=3,
                created_at=now,
            )
        )
    return tasks
