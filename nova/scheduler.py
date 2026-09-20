"""Adaptive pull scheduler: compatibility filter, fenced leases, failover."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from nova.clock import Clock
from nova.events import EventBus
from nova.jobs import split_job
from nova.models import KERNEL_SD_T2I, Job, NodeManifest, Task
from nova.protocol import (
    JOB_ANNOUNCE,
    NODE_DISCONNECTED,
    NODE_MANIFEST,
    NODE_OFFLINE,
    NODE_RECOVERED,
    TASK_COMPLETE,
    TASK_FAILED,
    TASK_OFFER,
    TASK_PROGRESS,
)
from nova.store import Store

IN_FLIGHT = frozenset({"LEASED", "RUNNING"})
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

LEASE_MIN_S = 20
LEASE_MAX_S = 90
DEFAULT_ESTIMATE_S = 15.0
SUSPECT_S = 15.0
OFFLINE_S = 30.0
MISSING_SCORE = 0.1


def is_compatible(node: NodeManifest, task: Task) -> bool:
    """Single source of truth. Do not use memory_free_mb."""
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


def score_node(node: NodeManifest, task: Task) -> float:
    """Missing benchmark score is 0.1, never KeyError. Higher = faster generate."""
    return float(node.benchmark_scores.get(task.kernel_id, MISSING_SCORE))


def generate_seconds(node: NodeManifest, task: Task | None = None) -> float:
    """Steady-state seconds per tile. Prefer measured generate_ms."""
    if node.generate_ms and node.generate_ms > 0:
        return max(float(node.generate_ms) / 1000.0, 0.05)
    score = 0.0
    if task is not None:
        score = float(node.benchmark_scores.get(task.kernel_id, 0.0) or 0.0)
    else:
        scores = node.benchmark_scores or {}
        score = float(next(iter(scores.values()), 0.0) or 0.0)
    if score > MISSING_SCORE:
        return max(1.0 / score, 0.05)
    return float(DEFAULT_ESTIMATE_S)


def node_busy_s(store: Store, node: NodeManifest) -> float:
    """Estimated remaining generate time already assigned to this node."""
    per = generate_seconds(node)
    busy = 0.0
    for task in store.list_tasks():
        if task.assigned_node != node.node_id:
            continue
        if task.state in IN_FLIGHT:
            busy += per
    return busy


def lease_seconds_for(
    node: NodeManifest,
    task: Task,
    settings: Any | None = None,
    *,
    default_estimate_s: float = DEFAULT_ESTIMATE_S,
    lease_min_s: int = LEASE_MIN_S,
    lease_max_s: int = LEASE_MAX_S,
) -> int:
    """score = 1000 / generate_ms → estimate_s = 1 / score. Missing/slow (≤0.1) → 15s default.

    lease_seconds = clamp(estimate * 2.5, 20, 90)
    """
    min_s = _setting(settings, "lease_min_s", lease_min_s)
    max_s = _setting(settings, "lease_max_s", lease_max_s)
    default_s = _setting(settings, "default_estimate_s", default_estimate_s)
    score = node.benchmark_scores.get(task.kernel_id, 0.0)
    if score <= MISSING_SCORE:
        estimate_s = float(default_s)
    else:
        estimate_s = 1.0 / float(score)
    return int(max(int(min_s), min(int(max_s), estimate_s * 2.5)))


def grant_lease(
    store: Store,
    node_id: str,
    task: Task,
    clock: Clock,
    lease_seconds: int,
    event_bus: EventBus | None = None,
) -> Task:
    """Fence the task. attempt_count increments here, not on fail/expire."""
    task.lease_gen += 1
    task.attempt_count += 1
    task.state = "LEASED"
    task.assigned_node = node_id
    task.lease_expires_at = clock.now() + timedelta(seconds=lease_seconds)
    task.last_failed_node = None
    store.put_task(task)

    node = store.get_node(node_id)
    if node is not None:
        node.current_slots_used += 1
        store.put_node(node)

    _refresh_job_state(store, task.job_id, event_bus)
    _emit(
        event_bus,
        TASK_OFFER,
        node_id=node_id,
        task_id=task.task_id,
        job_id=task.job_id,
        shard_index=task.shard_index,
        lease_gen=task.lease_gen,
        lease_seconds=lease_seconds,
    )
    return task


def _quota_remaining(store: Store, job: Job, node_id: str) -> int | None:
    if (job.scheduler_policy or "adaptive_pull") != "quota":
        return None
    assigned = int((job.quotas or {}).get(node_id, 0))
    used = 0
    for task in store.list_tasks(job.job_id):
        holder = task.assigned_node or task.reserved_node
        if holder != node_id:
            continue
        if task.state in ("LEASED", "RUNNING", "COMPLETED"):
            used += 1
    return assigned - used


def _faster_peer_finishes_sooner(store: Store, node: NodeManifest, task: Task) -> bool:
    """True if some other online worker would finish this tile sooner (lower wall)."""
    mine = node_busy_s(store, node) + generate_seconds(node, task)
    for peer in store.list_nodes():
        if peer.node_id == node.node_id or peer.status != "online":
            continue
        if not is_compatible(peer, task):
            continue
        theirs = node_busy_s(store, peer) + generate_seconds(peer, task)
        if theirs + 0.05 < mine:
            return True
    return False


def pick_task(store: Store, node: NodeManifest, clock: Clock | None = None) -> Task | None:  # noqa: ARG001
    """Oldest QUEUED compatible task. Skip if a faster peer finishes sooner."""
    if node.status != "online":
        return None

    eligible: list[Task] = []
    poison: list[Task] = []
    for task in store.list_tasks():
        if task.state != "QUEUED":
            continue
        if not is_compatible(node, task):
            continue
        job = store.get_job(task.job_id)
        if job is not None and (job.scheduler_policy or "adaptive_pull") == "quota":
            remaining = _quota_remaining(store, job, node.node_id)
            reserved = task.reserved_node
            if reserved and reserved != node.node_id:
                continue
            if remaining is not None and remaining <= 0:
                continue
        if task.last_failed_node == node.node_id:
            poison.append(task)
            continue
        if (job is None or (job.scheduler_policy or "adaptive_pull") == "adaptive_pull") and _faster_peer_finishes_sooner(
            store, node, task
        ):
            continue
        eligible.append(task)

    if eligible:
        eligible.sort(key=lambda t: (t.created_at, t.shard_index, -score_node(node, t)))
        return eligible[0]

    # This pull is the skipped cycle. Lift the blacklist so the next request may take it.
    for task in poison:
        task.last_failed_node = None
        store.put_task(task)
    return None


def on_work_request(
    store: Store,
    node_id: str,
    clock: Clock,
    *,
    settings: Any | None = None,
    event_bus: EventBus | None = None,
    available_slots: int | None = None,
) -> Task | None:
    node = store.get_node(node_id)
    if node is None or node.status != "online":
        return None
    if available_slots is not None and available_slots <= 0:
        return None
    store.touch_node(node_id, clock.now())
    task = pick_task(store, node, clock)
    if task is None:
        return None
    seconds = lease_seconds_for(node, task, settings)
    return grant_lease(store, node_id, task, clock, seconds, event_bus=event_bus)


def requeue_or_fail(
    store: Store,
    task: Task,
    clock: Clock | None = None,  # noqa: ARG001
    reason: str = "",
    event_bus: EventBus | None = None,
) -> Task:
    """Clear the lease. Do not increment attempt_count (it incremented on grant)."""
    if task.state not in IN_FLIGHT:
        return task
    _release_slot(store, task.assigned_node)
    task.assigned_node = None
    task.lease_expires_at = None
    if task.attempt_count < task.max_attempts:
        task.state = "QUEUED"
        _emit(
            event_bus,
            "TASK_REQUEUED",
            task_id=task.task_id,
            job_id=task.job_id,
            shard_index=task.shard_index,
            reason=reason,
            attempt_count=task.attempt_count,
        )
    else:
        task.state = "FAILED"
        _emit(
            event_bus,
            TASK_FAILED,
            task_id=task.task_id,
            job_id=task.job_id,
            shard_index=task.shard_index,
            reason=reason,
            attempt_count=task.attempt_count,
        )
    store.put_task(task)
    _refresh_job_state(store, task.job_id, event_bus)
    return task


def expire_leases(
    store: Store,
    clock: Clock,
    event_bus: EventBus | None = None,
) -> list[Task]:
    now = clock.now()
    expired: list[Task] = []
    for task in store.list_tasks():
        if task.state not in IN_FLIGHT:
            continue
        if task.lease_expires_at is None or task.lease_expires_at > now:
            continue
        requeue_or_fail(store, task, clock, "expired", event_bus=event_bus)
        expired.append(task)
    return expired


def on_disconnect(
    store: Store,
    node_id: str,
    clock: Clock,
    event_bus: EventBus | None = None,
) -> list[Task]:
    """Immediate requeue. Must not wait for the 30s heartbeat timeout."""
    return _mark_offline(
        store,
        node_id,
        clock,
        reason="disconnected",
        event_type=NODE_DISCONNECTED,
        event_bus=event_bus,
    )


def on_heartbeat(
    store: Store,
    node_id: str,
    clock: Clock,
    event_bus: EventBus | None = None,
) -> None:
    """Update last_seen. Suspect → online. Does NOT renew task leases."""
    node = store.get_node(node_id)
    if node is None:
        return
    store.touch_node(node_id, clock.now())
    if node.status == "suspect":
        node.status = "online"
        store.put_node(node)
        _emit(event_bus, NODE_RECOVERED, node_id=node_id, from_status="suspect")


def on_progress(
    store: Store,
    node_id: str,
    task_id: str,
    lease_gen: int,
    clock: Clock,
    lease_seconds: int,
    event_bus: EventBus | None = None,
) -> bool:
    task = store.get_task(task_id)
    if not _lease_matches(task, node_id, lease_gen):
        return False
    assert task is not None
    task.lease_expires_at = clock.now() + timedelta(seconds=lease_seconds)
    if task.state == "LEASED":
        task.state = "RUNNING"
        if task.started_at is None:
            task.started_at = clock.now()
    store.put_task(task)
    _emit(
        event_bus,
        TASK_PROGRESS,
        node_id=node_id,
        task_id=task_id,
        lease_gen=lease_gen,
    )
    return True


def _record_generate_ms(store: Store, node_id: str, execution_ms: int | None) -> None:
    if not execution_ms or execution_ms <= 0:
        return
    node = store.get_node(node_id)
    if node is None:
        return
    prev = node.generate_ms
    if prev and prev > 0:
        node.generate_ms = 0.7 * float(prev) + 0.3 * float(execution_ms)
    else:
        node.generate_ms = float(execution_ms)
    node.benchmark_scores[KERNEL_SD_T2I] = 1000.0 / max(float(node.generate_ms), 1.0)
    store.put_node(node)


def on_complete(
    store: Store,
    node_id: str,
    task_id: str,
    lease_gen: int,
    sha256: str,
    path: str,
    backend: str,
    clock: Clock,
    *,
    device_id: str | None = None,
    execution_ms: int | None = None,
    event_bus: EventBus | None = None,
) -> bool:
    """First valid matching lease_gen wins. Stale gen / wrong node → False."""
    task = store.get_task(task_id)
    if not _lease_matches(task, node_id, lease_gen):
        return False
    assert task is not None
    _release_slot(store, node_id)
    task.state = "COMPLETED"
    task.assigned_node = node_id
    task.lease_expires_at = None
    task.completed_at = clock.now()
    if task.started_at is None:
        task.started_at = task.completed_at
    if sha256:
        task.result_sha256 = sha256
    if path:
        task.result_path = str(path)
    if backend:
        task.backend_used = backend
    task.device_id_used = device_id
    task.execution_ms = execution_ms
    store.put_task(task)
    _record_generate_ms(store, node_id, execution_ms)
    _refresh_job_state(store, task.job_id, event_bus)
    _emit(
        event_bus,
        TASK_COMPLETE,
        node_id=node_id,
        task_id=task_id,
        job_id=task.job_id,
        shard_index=task.shard_index,
        backend=backend,
        sha256=sha256,
    )
    return True


def on_task_failed(
    store: Store,
    node_id: str,
    task_id: str,
    lease_gen: int,
    clock: Clock,
    reason: str = "failed",
    event_bus: EventBus | None = None,
) -> bool:
    task = store.get_task(task_id)
    if not _lease_matches(task, node_id, lease_gen):
        return False
    assert task is not None
    task.last_failed_node = node_id
    requeue_or_fail(store, task, clock, reason, event_bus=event_bus)
    return True


def check_node_liveness(
    store: Store,
    clock: Clock,
    suspect_s: float = SUSPECT_S,
    offline_s: float = OFFLINE_S,
    event_bus: EventBus | None = None,
) -> list[str]:
    """Heartbeat fallback. Offline path is the same as disconnect (requeue now)."""
    now = clock.now()
    offlined: list[str] = []
    for node in store.list_nodes():
        if node.status == "offline":
            continue
        seen = store.get_last_seen(node.node_id)
        if seen is None:
            _mark_offline(
                store,
                node.node_id,
                clock,
                reason="never_seen",
                event_type=NODE_OFFLINE,
                event_bus=event_bus,
            )
            offlined.append(node.node_id)
            continue
        age = (now - seen).total_seconds()
        if age >= offline_s:
            _mark_offline(
                store,
                node.node_id,
                clock,
                reason="offline",
                event_type=NODE_OFFLINE,
                event_bus=event_bus,
            )
            offlined.append(node.node_id)
        elif age >= suspect_s and node.status != "suspect":
            node.status = "suspect"
            store.put_node(node)
            _emit(event_bus, "NODE_SUSPECT", node_id=node.node_id)
    return offlined


def cancel_job(
    store: Store,
    job_id: str,
    clock: Clock | None = None,  # noqa: ARG001
    event_bus: EventBus | None = None,
) -> Job | None:
    job = store.get_job(job_id)
    if job is None:
        return None
    for task in store.list_tasks(job_id):
        if task.state not in ("QUEUED", "LEASED", "RUNNING"):
            continue
        if task.state in IN_FLIGHT:
            _release_slot(store, task.assigned_node)
        task.state = "CANCELLED"
        task.assigned_node = None
        task.lease_expires_at = None
        store.put_task(task)
    job.state = "CANCELLED"
    store.put_job(job)
    _emit(event_bus, "JOB_CANCELLED", job_id=job_id)
    return job


class Scheduler:
    """Coordinator facade. Holds store, clock, optional event bus, lease numbers."""

    def __init__(
        self,
        store: Store,
        clock: Clock | None = None,
        event_bus: EventBus | None = None,
        settings: Any | None = None,
        bus: EventBus | None = None,
        *,
        suspect_s: float = SUSPECT_S,
        offline_s: float = OFFLINE_S,
        default_estimate_s: float = DEFAULT_ESTIMATE_S,
        lease_min_s: int = LEASE_MIN_S,
        lease_max_s: int = LEASE_MAX_S,
    ) -> None:
        self.store = store
        self.clock = clock or Clock()
        self.event_bus = event_bus or bus
        self.bus = self.event_bus
        self.settings = settings
        self.suspect_s = _setting(settings, "suspect_s", suspect_s)
        self.offline_s = _setting(settings, "offline_s", offline_s)
        self.default_estimate_s = _setting(settings, "default_estimate_s", default_estimate_s)
        self.lease_min_s = int(_setting(settings, "lease_min_s", lease_min_s))
        self.lease_max_s = int(_setting(settings, "lease_max_s", lease_max_s))

    def submit_job(self, job: Job) -> list[Task]:
        self.store.put_job(job)
        tasks = split_job(job, self.clock)
        for task in tasks:
            if self.store.get_task(task.task_id) is None:
                self.store.put_task(task)
        _emit(
            self.event_bus,
            JOB_ANNOUNCE,
            job_id=job.job_id,
            name=job.name,
            n_tasks=len(self.store.list_tasks(job.job_id)),
        )
        return self.store.list_tasks(job.job_id)

    def register_node(self, manifest: NodeManifest) -> NodeManifest:
        existing = self.store.get_node(manifest.node_id)
        recovered = existing is not None and existing.status in ("offline", "suspect")
        if existing is not None and existing.status != "offline":
            manifest.current_slots_used = existing.current_slots_used
        else:
            manifest.current_slots_used = 0
        manifest.status = "online"
        self.store.put_node(manifest)
        self.store.touch_node(manifest.node_id, self.clock.now())
        if recovered:
            _emit(self.event_bus, NODE_RECOVERED, node_id=manifest.node_id)
        else:
            _emit(self.event_bus, NODE_MANIFEST, node_id=manifest.node_id)
        return manifest

    def is_compatible(self, node: NodeManifest, task: Task) -> bool:
        return is_compatible(node, task)

    def score_node(self, node: NodeManifest, task: Task) -> float:
        return score_node(node, task)

    def lease_seconds_for(self, node: NodeManifest, task: Task) -> int:
        return lease_seconds_for(
            node,
            task,
            self.settings,
            default_estimate_s=self.default_estimate_s,
            lease_min_s=self.lease_min_s,
            lease_max_s=self.lease_max_s,
        )

    def pick_task(self, node: NodeManifest) -> Task | None:
        return pick_task(self.store, node, self.clock)

    def grant_lease(self, node_id: str, task: Task, lease_seconds: int | None = None) -> Task:
        node = self.store.get_node(node_id)
        if lease_seconds is None:
            seconds = self.lease_seconds_for(node, task) if node is not None else int(self.lease_min_s)
        else:
            seconds = lease_seconds
        return grant_lease(
            self.store, node_id, task, self.clock, seconds, event_bus=self.event_bus
        )

    def on_work_request(self, node_id: str, available_slots: int | None = None, **_payload: Any) -> Task | None:
        return on_work_request(
            self.store,
            node_id,
            self.clock,
            settings=self.settings,
            event_bus=self.event_bus,
            available_slots=available_slots,
        )

    def requeue_or_fail(self, task: Task, reason: str = "") -> Task:
        return requeue_or_fail(self.store, task, self.clock, reason, event_bus=self.event_bus)

    def on_heartbeat(self, node_id: str, clock: Clock | None = None) -> None:
        if clock is not None:
            self.clock = clock
        on_heartbeat(self.store, node_id, self.clock, event_bus=self.event_bus)

    def on_progress(self, *args: Any, **kwargs: Any) -> bool:
        node_id, task_id, lease_gen, lease_seconds = self._parse_progress(*args, **kwargs)
        node = self.store.get_node(node_id)
        task = self.store.get_task(task_id)
        seconds = lease_seconds
        if seconds is None and task is not None and node is not None:
            seconds = self.lease_seconds_for(node, task)
        if seconds is None:
            seconds = int(self.lease_min_s)
        return on_progress(
            self.store,
            node_id,
            task_id,
            lease_gen,
            self.clock,
            int(seconds),
            event_bus=self.event_bus,
        )

    def on_complete(self, *args: Any, **kwargs: Any) -> bool:
        parsed = self._parse_complete(*args, **kwargs)
        if parsed is None:
            return False
        node_id, task_id, lease_gen, sha256, path, backend, device_id, execution_ms = parsed
        return on_complete(
            self.store,
            node_id,
            task_id,
            lease_gen,
            sha256,
            str(path),
            backend,
            self.clock,
            device_id=device_id,
            execution_ms=execution_ms,
            event_bus=self.event_bus,
        )

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
        """Validate lease, persist PNG, then complete. Raises ValueError on garbage PNG."""
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

    def on_task_failed(self, *args: Any, **kwargs: Any) -> bool:
        node_id, task_id, lease_gen, reason = self._parse_failed(*args, **kwargs)
        return on_task_failed(
            self.store,
            node_id,
            task_id,
            lease_gen,
            self.clock,
            reason=reason,
            event_bus=self.event_bus,
        )

    def on_disconnect(self, node_id: str, clock: Clock | None = None) -> list[Task]:
        if clock is not None:
            self.clock = clock
        return on_disconnect(self.store, node_id, self.clock, event_bus=self.event_bus)

    def expire_leases(self, clock: Clock | None = None) -> list[Task]:
        if clock is not None:
            self.clock = clock
        return expire_leases(self.store, self.clock, event_bus=self.event_bus)

    def check_node_liveness(
        self,
        clock: Clock | None = None,
        suspect_s: float | None = None,
        offline_s: float | None = None,
    ) -> list[str]:
        if clock is not None:
            self.clock = clock
        return check_node_liveness(
            self.store,
            self.clock,
            suspect_s=self.suspect_s if suspect_s is None else suspect_s,
            offline_s=self.offline_s if offline_s is None else offline_s,
            event_bus=self.event_bus,
        )

    def cancel_job(self, job_id: str) -> Job | None:
        return cancel_job(self.store, job_id, self.clock, event_bus=self.event_bus)

    def tick(self) -> None:
        """Coordinator loop helper: expire leases, then heartbeat liveness."""
        self.expire_leases()
        self.check_node_liveness()

    def offer_payload(self, task: Task, *, http_base: str, lease_seconds: int) -> dict[str, Any]:
        base = http_base.rstrip("/")
        return {
            "task_id": task.task_id,
            "job_id": task.job_id,
            "kernel_id": task.kernel_id,
            "prompt": task.prompt,
            "seed": task.seed,
            "steps": task.steps,
            "width": task.width,
            "height": task.height,
            "min_memory_mb": task.min_memory_mb,
            "allowed_backends": list(task.allowed_backends),
            "lease_seconds": lease_seconds,
            "lease_gen": task.lease_gen,
            "fetch_url": f"{base}/jobs/{task.job_id}/tasks/{task.task_id}/input.json",
            "upload_url": f"{base}/jobs/{task.job_id}/tasks/{task.task_id}/result",
        }

    def _parse_progress(self, *args: Any, **kwargs: Any) -> tuple[str, str, int, int | None]:
        if args and isinstance(args[0], Task):
            task = args[0]
            lease_gen = int(args[1] if len(args) > 1 else kwargs.get("lease_gen", task.lease_gen))
            lease_seconds = args[2] if len(args) > 2 else kwargs.get("lease_seconds")
            return task.assigned_node or "", task.task_id, lease_gen, lease_seconds
        node_id = args[0] if args else kwargs.get("node_id")
        task_id = args[1] if len(args) > 1 else kwargs.get("task_id")
        lease_gen = args[2] if len(args) > 2 else kwargs.get("lease_gen")
        lease_seconds = args[3] if len(args) > 3 else kwargs.get("lease_seconds")
        return str(node_id), str(task_id), int(lease_gen), lease_seconds

    def _parse_complete(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, str, int, str, str, str, str | None, int | None] | None:
        device_id = kwargs.get("device_id")
        execution_ms = kwargs.get("execution_ms")
        if args and isinstance(args[0], Task):
            task = args[0]
            path = str(args[1] if len(args) > 1 else kwargs.get("path") or task.result_path or "")
            sha256 = str(kwargs.get("sha256") or task.result_sha256 or "")
            backend = str(kwargs.get("backend") or task.backend_used or "")
            node_id = task.assigned_node or ""
            return node_id, task.task_id, task.lease_gen, sha256, path, backend, device_id, execution_ms
        if len(args) < 3:
            return None
        node_id, task_id, lease_gen = args[0], args[1], args[2]
        sha256 = args[3] if len(args) > 3 else kwargs.get("sha256") or ""
        path = args[4] if len(args) > 4 else kwargs.get("path") or ""
        backend = args[5] if len(args) > 5 else kwargs.get("backend") or ""
        return (
            str(node_id),
            str(task_id),
            int(lease_gen),
            str(sha256),
            str(path),
            str(backend),
            device_id,
            execution_ms,
        )

    def _parse_failed(self, *args: Any, **kwargs: Any) -> tuple[str, str, int, str]:
        reason = str(kwargs.get("reason") or "failed")
        if args and isinstance(args[0], Task):
            task = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                reason = args[1]
            return task.assigned_node or "", task.task_id, task.lease_gen, reason
        node_id = args[0] if args else kwargs.get("node_id")
        task_id = args[1] if len(args) > 1 else kwargs.get("task_id")
        lease_gen = args[2] if len(args) > 2 else kwargs.get("lease_gen", 0)
        if len(args) > 3 and isinstance(args[3], str):
            reason = args[3]
        return str(node_id), str(task_id), int(lease_gen), reason


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


def _release_slot(store: Store, node_id: str | None) -> None:
    if not node_id:
        return
    node = store.get_node(node_id)
    if node is None:
        return
    node.current_slots_used = max(0, node.current_slots_used - 1)
    store.put_node(node)


def _mark_offline(
    store: Store,
    node_id: str,
    clock: Clock,
    *,
    reason: str,
    event_type: str,
    event_bus: EventBus | None,
) -> list[Task]:
    node = store.get_node(node_id)
    if node is not None:
        node.status = "offline"
        store.put_node(node)
    affected: list[Task] = []
    for task in store.list_tasks():
        if task.assigned_node == node_id and task.state in IN_FLIGHT:
            requeue_or_fail(store, task, clock, reason, event_bus=event_bus)
            affected.append(task)
    if node is not None:
        node.current_slots_used = 0
        store.put_node(node)
    _emit(event_bus, event_type, node_id=node_id, requeued=[t.task_id for t in affected])
    return affected


def _refresh_job_state(store: Store, job_id: str, event_bus: EventBus | None = None) -> None:
    job = store.get_job(job_id)
    if job is None or job.state == "CANCELLED":
        return
    tasks = store.list_tasks(job_id)
    if not tasks:
        return
    prev = job.state
    states = [t.state for t in tasks]
    if all(s == "COMPLETED" for s in states):
        job.state = "COMPLETED"
    elif all(s in TERMINAL for s in states):
        job.state = "FAILED" if any(s == "FAILED" for s in states) else "CANCELLED"
    elif any(s != "QUEUED" for s in states) or any(t.attempt_count > 0 for t in tasks):
        job.state = "RUNNING"
    else:
        job.state = "QUEUED"
    store.put_job(job)
    if job.state != prev and job.state in ("COMPLETED", "FAILED"):
        _emit(event_bus, f"JOB_{job.state}", job_id=job_id)


def _emit(event_bus: EventBus | None, type_: str, **fields: Any) -> None:
    if event_bus is not None:
        event_bus.emit(type_, **fields)


def _setting(settings: Any | None, name: str, default: Any) -> Any:
    if settings is None:
        return default
    value = getattr(settings, name, default)
    try:
        if isinstance(value, bool) or value is None:
            return default
        return type(default)(value)
    except (TypeError, ValueError):
        return default
