"""Coordinator control-plane message loop. Uses fakes so scheduler.py is optional."""

from __future__ import annotations

import inspect
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from nova.config import Settings
from nova.coordinator import Coordinator
from nova.events import EventBus
from nova.models import Device, Job, JobRequirements, NodeIdentity, NodeManifest, PromptSpec, Task
from nova.protocol import (
    HEARTBEAT,
    HELLO,
    JOB_ANNOUNCE,
    NODE_DISCONNECTED,
    NODE_GOODBYE,
    NODE_MANIFEST,
    NODE_OFFLINE,
    NODE_UPDATE,
    RESULT_ACK,
    TASK_ACCEPT,
    TASK_COMPLETE,
    TASK_FAILED,
    TASK_OFFER,
    TASK_PROGRESS,
    TASK_REJECT,
    TASK_STARTED,
    WORK_REQUEST,
    Envelope,
    msg,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self) -> None:
        self._on_message = None
        self._on_connect = None
        self._on_disconnect = None
        self.started = False
        self.sent: list[tuple[str, Envelope]] = []
        self.broadcasts: list[Envelope] = []

    def on_message(self, handler) -> None:
        self._on_message = handler

    def on_peer_connect(self, handler) -> None:
        self._on_connect = handler

    def on_peer_disconnect(self, handler) -> None:
        self._on_disconnect = handler

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def send(self, peer_id: str, env: Envelope) -> None:
        self.sent.append((peer_id, env))

    async def broadcast(self, env: Envelope) -> None:
        self.broadcasts.append(env)

    async def inject(self, peer_id: str, env: Envelope) -> None:
        if self._on_message is None:
            return
        res = self._on_message(peer_id, env)
        if inspect.isawaitable(res):
            await res

    def fire_disconnect(self, peer_id: str) -> None:
        if self._on_disconnect is None:
            return
        self._on_disconnect(peer_id)


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, Task] = {}
        self.nodes: dict[str, NodeManifest] = {}
        self.results: dict[tuple[str, str], Any] = {}
        self._last_seen: dict[str, datetime] = {}

    def put_job(self, job: Job) -> None:
        self.jobs[job.job_id] = job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    def put_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def list_tasks(self, job_id: str | None = None) -> list[Task]:
        tasks = list(self.tasks.values())
        if job_id is not None:
            tasks = [t for t in tasks if t.job_id == job_id]
        return tasks

    def put_node(self, node: NodeManifest) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> NodeManifest | None:
        return self.nodes.get(node_id)

    def list_nodes(self) -> list[NodeManifest]:
        return list(self.nodes.values())

    def touch_node(self, node_id: str, when=None) -> None:
        self._last_seen[node_id] = when or datetime.now(timezone.utc)

    def get_last_seen(self, node_id: str):
        return self._last_seen.get(node_id)

    def save_result(self, job_id: str, task_id: str, data: bytes, **kwargs: Any) -> None:
        self.results[(job_id, task_id)] = data


class FakeScheduler:
    def __init__(self, store: FakeStore, clock) -> None:
        self.store = store
        self.clock = clock
        self.calls: list[tuple] = []
        self.heartbeats: list[str] = []
        self.disconnects: list[str] = []
        self.progress: list[tuple] = []
        self.completes: list[tuple] = []
        self.failures: list[tuple] = []
        self.requeues: list[tuple] = []
        self.expire_calls = 0
        self.liveness_calls: list[tuple] = []
        self.cancelled: list[str] = []
        self.lease_seconds = 20

    def is_compatible(self, node: NodeManifest, task: Task) -> bool:
        if node.status != "online":
            return False
        if task.kernel_id not in node.supported_kernels:
            return False
        if node.current_slots_used >= node.max_concurrency:
            return False
        for device in node.devices:
            if device.backend in task.allowed_backends and device.memory_total_mb >= task.min_memory_mb:
                return True
        return False

    def pick_task(self, node: NodeManifest) -> Task | None:
        self.calls.append(("pick_task", node.node_id))
        queued = [t for t in self.store.list_tasks() if t.state == "QUEUED"]
        queued.sort(key=lambda t: (t.created_at, t.shard_index))
        for task in queued:
            if task.last_failed_node == node.node_id:
                continue
            if self.is_compatible(node, task):
                return task
        return None

    def grant_lease(self, task: Task, node: NodeManifest) -> Task:
        self.calls.append(("grant_lease", task.task_id, node.node_id))
        task.lease_gen += 1
        task.attempt_count += 1
        task.state = "LEASED"
        task.assigned_node = node.node_id
        task.lease_expires_at = self.clock.now() + timedelta(seconds=self.lease_seconds)
        self.store.put_task(task)
        return task

    def lease_seconds_for(self, node: NodeManifest, task: Task) -> int:
        self.calls.append(("lease_seconds_for", node.node_id, task.task_id))
        return self.lease_seconds

    def on_work_request(
        self,
        node_id: str,
        available_slots: int = 1,
        device_ids: list[str] | None = None,
    ) -> Task | None:
        self.calls.append(("on_work_request", node_id, available_slots, tuple(device_ids or [])))
        node = self.store.get_node(node_id)
        if node is None or node.status != "online" or available_slots <= 0:
            return None
        task = self.pick_task(node)
        if task is None:
            return None
        return self.grant_lease(task, node)

    def requeue_or_fail(self, task: Task, reason: str = "") -> Task:
        self.requeues.append((task.task_id, reason))
        if task.attempt_count < task.max_attempts:
            task.state = "QUEUED"
        else:
            task.state = "FAILED"
        task.assigned_node = None
        task.lease_expires_at = None
        self.store.put_task(task)
        return task

    def expire_leases(self) -> list[Task]:
        self.expire_calls += 1
        now = self.clock.now()
        expired: list[Task] = []
        for task in self.store.list_tasks():
            if task.state in ("LEASED", "RUNNING") and task.lease_expires_at and task.lease_expires_at <= now:
                expired.append(self.requeue_or_fail(task, reason="expired"))
        return expired

    def on_disconnect(self, node_id: str) -> list[Task]:
        self.disconnects.append(node_id)
        node = self.store.get_node(node_id)
        if node is not None:
            node.status = "offline"
            self.store.put_node(node)
        out: list[Task] = []
        for task in self.store.list_tasks():
            if task.assigned_node == node_id and task.state in ("LEASED", "RUNNING"):
                out.append(self.requeue_or_fail(task, reason="disconnect"))
        return out

    def on_heartbeat(self, node_id: str) -> None:
        self.heartbeats.append(node_id)

    def on_progress(self, node_id: str, task_id: str, lease_gen: int | None = None, **kwargs: Any) -> Task | None:
        self.progress.append((node_id, task_id, lease_gen))
        task = self.store.get_task(task_id)
        if task is None or task.assigned_node != node_id:
            return None
        if lease_gen is not None and task.lease_gen != lease_gen:
            return None
        if task.state in ("LEASED", "RUNNING"):
            task.lease_expires_at = self.clock.now() + timedelta(seconds=self.lease_seconds)
            self.store.put_task(task)
        return task

    def on_complete(
        self,
        node_id: str,
        task_id: str,
        lease_gen: int | None = None,
        **kwargs: Any,
    ) -> Task | None:
        self.completes.append((node_id, task_id, lease_gen, kwargs))
        task = self.store.get_task(task_id)
        if task is None or task.assigned_node != node_id:
            return None
        if lease_gen is not None and task.lease_gen != lease_gen:
            return None
        if task.state not in ("LEASED", "RUNNING"):
            return None
        task.state = "COMPLETED"
        task.completed_at = self.clock.now()
        task.result_sha256 = kwargs.get("sha256")
        task.execution_ms = kwargs.get("execution_ms")
        self.store.put_task(task)
        return task

    def on_task_failed(self, node_id: str, task_id: str, **kwargs: Any) -> Task | None:
        self.failures.append((node_id, task_id, kwargs))
        task = self.store.get_task(task_id)
        if task is None:
            return None
        task.last_failed_node = node_id
        return self.requeue_or_fail(task, reason=str(kwargs.get("error") or "failed"))

    def check_node_liveness(
        self,
        heartbeat_s: float = 5.0,
        suspect_s: float = 15.0,
        offline_s: float = 30.0,
    ) -> list[str]:
        self.liveness_calls.append((heartbeat_s, suspect_s, offline_s))
        return []

    def cancel_job(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        for task in self.store.list_tasks(job_id):
            if task.state != "COMPLETED":
                task.state = "CANCELLED"
                self.store.put_task(task)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PEER = "pear-peer-1"
WORKER = "nova-gpu"


def make_node(node_id: str = WORKER, status: str = "online") -> NodeManifest:
    return NodeManifest(
        node_id=node_id,
        hostname="worker",
        devices=[
            Device(
                device_id="cuda:0",
                backend="cuda",
                vendor="nvidia",
                model="RTX 4090",
                memory_total_mb=24576,
            )
        ],
        supported_kernels=["sd.t2i.v1"],
        status=status,  # type: ignore[arg-type]
        max_concurrency=1,
        benchmark_scores={"sd.t2i.v1": 2.0},
    )


def make_task(clock, task_id: str = "task-1", job_id: str = "job-1", shard: int = 0) -> Task:
    return Task(
        task_id=task_id,
        job_id=job_id,
        shard_index=shard,
        prompt="neon street at night, cinematic, 35mm",
        seed=1000,
        created_at=clock.now(),
        state="QUEUED",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(advertise_host="192.168.1.5", http_port=8080)


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def scheduler(store, clock) -> FakeScheduler:
    return FakeScheduler(store, clock)


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def identity(clock) -> NodeIdentity:
    return NodeIdentity(node_id="nova-coord", created_at=clock.now())


@pytest.fixture
def coord(store, scheduler, transport, bus, settings, clock, identity) -> Coordinator:
    return Coordinator(store, scheduler, transport, bus, settings, clock, identity)


def event_types(bus: EventBus) -> list[str]:
    return [e["type"] for e in bus.history()]


# ---------------------------------------------------------------------------
# Wiring / lifecycle
# ---------------------------------------------------------------------------


def test_on_peer_disconnect_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(Coordinator.on_peer_disconnect)


async def test_start_stop_wires_transport_and_tick(coord: Coordinator, transport: FakeTransport, scheduler: FakeScheduler):
    await coord.start()
    assert transport.started
    assert transport._on_message == coord.handle_message
    assert transport._on_disconnect == coord.on_peer_disconnect
    assert coord._tick_task is not None
    coord._tick()
    assert scheduler.expire_calls >= 1
    assert scheduler.liveness_calls[-1] == (5.0, 15.0, 30.0)
    await coord.stop()
    assert transport.started is False


def test_tick_uses_settings_liveness_windows(coord: Coordinator, scheduler: FakeScheduler):
    coord._tick()
    assert scheduler.expire_calls == 1
    assert scheduler.liveness_calls == [(5.0, 15.0, 30.0)]


# ---------------------------------------------------------------------------
# HELLO / manifest
# ---------------------------------------------------------------------------


async def test_hello_records_protocol_version_and_maps_peer(coord: Coordinator, bus: EventBus):
    await coord.handle_message(PEER, msg(HELLO, WORKER, node_id=WORKER, protocol_version=1))
    assert coord._peer_to_node[PEER] == WORKER
    assert coord._node_to_peer[WORKER] == PEER
    assert coord._protocol_versions[WORKER] == 1
    assert "HELLO" in event_types(bus)


async def test_node_manifest_stored_and_emitted(coord: Coordinator, store: FakeStore, bus: EventBus):
    node = make_node()
    await coord.handle_message(PEER, msg(NODE_MANIFEST, WORKER, **node.model_dump()))
    stored = store.get_node(WORKER)
    assert stored is not None
    assert stored.status == "online"
    assert stored.devices[0].backend == "cuda"
    assert NODE_MANIFEST in event_types(bus)


async def test_node_update_merges_existing(coord: Coordinator, store: FakeStore):
    store.put_node(make_node())
    await coord.handle_message(
        PEER,
        msg(NODE_UPDATE, WORKER, node_id=WORKER, current_slots_used=1, hostname="worker-2"),
    )
    stored = store.get_node(WORKER)
    assert stored is not None
    assert stored.current_slots_used == 1
    assert stored.hostname == "worker-2"
    assert stored.devices  # preserved from original


async def test_rejoin_emits_node_recovered(coord: Coordinator, store: FakeStore, bus: EventBus):
    node = make_node(status="offline")
    store.put_node(node)
    await coord.handle_message(PEER, msg(NODE_MANIFEST, WORKER, **make_node().model_dump()))
    assert "NODE_RECOVERED" in event_types(bus)
    assert store.get_node(WORKER).status == "online"


# ---------------------------------------------------------------------------
# Heartbeat must not renew leases
# ---------------------------------------------------------------------------


async def test_heartbeat_does_not_renew_leases(coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, clock):
    node = make_node()
    store.put_node(node)
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 1
    expires = clock.now() + timedelta(seconds=20)
    task.lease_expires_at = expires
    store.put_task(task)

    await coord.handle_message(PEER, msg(HEARTBEAT, WORKER, node_id=WORKER))

    assert scheduler.heartbeats == [WORKER]
    assert ("grant_lease", task.task_id, WORKER) not in scheduler.calls
    refreshed = store.get_task(task.task_id)
    assert refreshed is not None
    assert refreshed.lease_expires_at == expires
    assert scheduler.expire_calls == 0


# ---------------------------------------------------------------------------
# WORK_REQUEST → TASK_OFFER
# ---------------------------------------------------------------------------


async def test_work_request_sends_inline_task_offer(
    coord: Coordinator,
    store: FakeStore,
    transport: FakeTransport,
    settings: Settings,
    clock,
):
    store.put_node(make_node())
    store.put_task(make_task(clock))

    await coord.handle_message(
        PEER,
        msg(WORK_REQUEST, WORKER, node_id=WORKER, available_slots=1, device_ids=["cuda:0"]),
    )

    assert len(transport.sent) == 1
    peer, env = transport.sent[0]
    assert peer == PEER
    assert env.type == TASK_OFFER
    p = env.payload
    assert p["prompt"] == "neon street at night, cinematic, 35mm"
    assert p["seed"] == 1000
    assert p["lease_gen"] == 1
    assert p["lease_seconds"] == 20
    base = settings.public_http_url()
    assert p["upload_url"] == f"{base}/jobs/job-1/tasks/task-1/result"
    assert p["fetch_url"] == f"{base}/jobs/job-1/tasks/task-1/input.json"
    stored = store.get_task("task-1")
    assert stored is not None
    assert stored.state == "LEASED"
    assert stored.assigned_node == WORKER


async def test_work_request_dropped_if_node_offline(
    coord: Coordinator, store: FakeStore, transport: FakeTransport, clock
):
    store.put_node(make_node(status="offline"))
    store.put_task(make_task(clock))
    await coord.handle_message(PEER, msg(WORK_REQUEST, WORKER, available_slots=1))
    assert transport.sent == []


async def test_work_request_no_queued_work_sends_nothing(
    coord: Coordinator, store: FakeStore, transport: FakeTransport
):
    store.put_node(make_node())
    await coord.handle_message(PEER, msg(WORK_REQUEST, WORKER, available_slots=1))
    assert transport.sent == []


# ---------------------------------------------------------------------------
# ACCEPT / REJECT
# ---------------------------------------------------------------------------


async def test_task_accept_marks_running_if_still_leased(
    coord: Coordinator, store: FakeStore, clock, bus: EventBus
):
    store.put_node(make_node())
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 1
    store.put_task(task)

    await coord.handle_message(PEER, msg(TASK_ACCEPT, WORKER, task_id="task-1", lease_gen=1))

    stored = store.get_task("task-1")
    assert stored is not None
    assert stored.state == "RUNNING"
    assert stored.started_at is not None
    assert "TASK_ACCEPT" in event_types(bus)


async def test_task_accept_ignored_if_not_leased_to_them(
    coord: Coordinator, store: FakeStore, clock
):
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = "nova-other"
    task.lease_gen = 1
    store.put_task(task)
    await coord.handle_message(PEER, msg(TASK_ACCEPT, WORKER, task_id="task-1", lease_gen=1))
    assert store.get_task("task-1").state == "LEASED"


async def test_task_accept_ignored_stale_lease_gen(coord: Coordinator, store: FakeStore, clock):
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 3
    store.put_task(task)
    await coord.handle_message(PEER, msg(TASK_ACCEPT, WORKER, task_id="task-1", lease_gen=1))
    assert store.get_task("task-1").state == "LEASED"


async def test_task_reject_requeues(coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, clock):
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 1
    task.attempt_count = 1
    store.put_task(task)
    await coord.handle_message(PEER, msg(TASK_REJECT, WORKER, task_id="task-1", reason="busy"))
    stored = store.get_task("task-1")
    assert stored is not None
    assert stored.state == "QUEUED"
    assert stored.assigned_node is None
    assert scheduler.requeues == [("task-1", "busy")]


# ---------------------------------------------------------------------------
# Progress / complete / failed
# ---------------------------------------------------------------------------


async def test_task_started_and_progress_go_to_scheduler(
    coord: Coordinator, scheduler: FakeScheduler, store: FakeStore, clock
):
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 1
    store.put_task(task)
    await coord.handle_message(PEER, msg(TASK_STARTED, WORKER, task_id="task-1", lease_gen=1))
    await coord.handle_message(PEER, msg(TASK_PROGRESS, WORKER, task_id="task-1", lease_gen=1))
    assert scheduler.progress == [(WORKER, "task-1", 1), (WORKER, "task-1", 1)]


async def test_task_complete_rejects_live_task_without_stored_result(
    coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, transport: FakeTransport, clock
):
    task = make_task(clock)
    task.state = "RUNNING"
    task.assigned_node = WORKER
    task.lease_gen = 2
    store.put_task(task)
    await coord.handle_message(
        PEER,
        msg(
            TASK_COMPLETE,
            WORKER,
            task_id="task-1",
            job_id="job-1",
            lease_gen=2,
            sha256="abc",
            execution_ms=40,
            result_url="http://192.168.1.5:8080/jobs/job-1/tiles/task-1.png",
        ),
    )
    stored = store.get_task("task-1")
    assert stored is not None
    assert stored.state == "RUNNING"
    assert scheduler.completes == []
    assert transport.sent[-1][1].type == RESULT_ACK
    assert transport.sent[-1][1].payload["accepted"] is False
    assert transport.sent[-1][0] == PEER


async def test_task_complete_acks_matching_already_stored_result_without_scheduler_mutation(
    coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, transport: FakeTransport, clock
):
    task = make_task(clock)
    task.state = "COMPLETED"
    task.assigned_node = WORKER
    task.lease_gen = 2
    task.completed_at = clock.now()
    task.result_sha256 = "abc"
    task.result_path = "/tiles/job-1/task-1.png"
    store.put_task(task)

    await coord.handle_message(
        PEER,
        msg(
            TASK_COMPLETE,
            WORKER,
            task_id="task-1",
            job_id="job-1",
            lease_gen=2,
            sha256="abc",
            execution_ms=40,
            result_url="http://192.168.1.5:8080/jobs/job-1/tiles/task-1.png",
        ),
    )

    stored = store.get_task("task-1")
    assert stored is task
    assert stored.state == "COMPLETED"
    assert stored.result_sha256 == "abc"
    assert stored.result_path == "/tiles/job-1/task-1.png"
    assert scheduler.completes == []
    peer, ack = transport.sent[-1]
    assert peer == PEER
    assert ack.type == RESULT_ACK
    assert ack.payload["accepted"] is True
    assert ack.payload["status"] == "already_accepted"


async def test_stale_complete_still_sends_result_ack(
    coord: Coordinator,
    store: FakeStore,
    scheduler: FakeScheduler,
    transport: FakeTransport,
    clock,
):
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.lease_gen = 4
    store.put_task(task)
    await coord.handle_message(
        PEER,
        msg(TASK_COMPLETE, WORKER, task_id="task-1", job_id="job-1", lease_gen=1, sha256="old"),
    )
    assert store.get_task("task-1").state == "LEASED"
    ack = transport.sent[-1][1]
    assert ack.type == RESULT_ACK
    assert ack.payload["accepted"] is False
    assert scheduler.completes == []


async def test_task_failed_calls_scheduler(
    coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, clock
):
    task = make_task(clock)
    task.state = "RUNNING"
    task.assigned_node = WORKER
    task.lease_gen = 1
    task.attempt_count = 1
    store.put_task(task)
    await coord.handle_message(
        PEER, msg(TASK_FAILED, WORKER, task_id="task-1", lease_gen=1, error="OOM")
    )
    assert scheduler.failures[0][0] == WORKER
    assert scheduler.failures[0][2]["error"] == "OOM"
    assert store.get_task("task-1").state == "QUEUED"
    assert store.get_task("task-1").last_failed_node == WORKER


# ---------------------------------------------------------------------------
# Disconnect / goodbye — immediate, no 30s wait
# ---------------------------------------------------------------------------


async def test_peer_disconnect_requeues_immediately(
    coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, bus: EventBus, clock
):
    store.put_node(make_node())
    await coord.handle_message(PEER, msg(HELLO, WORKER, node_id=WORKER))
    task = make_task(clock)
    task.state = "RUNNING"
    task.assigned_node = WORKER
    task.lease_gen = 1
    task.attempt_count = 1
    store.put_task(task)

    t0 = time.monotonic()
    coord.on_peer_disconnect(PEER)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.25  # must not wait for heartbeat/offline (30s)
    assert scheduler.disconnects == [WORKER]
    stored = store.get_task("task-1")
    assert stored is not None
    assert stored.state == "QUEUED"
    assert store.get_node(WORKER).status == "offline"
    types = event_types(bus)
    assert NODE_DISCONNECTED in types
    assert NODE_OFFLINE in types
    disc = [e for e in bus.history() if e["type"] == NODE_DISCONNECTED][0]
    assert disc["node_id"] == WORKER
    assert disc["tasks"] == ["task-1"]


async def test_node_goodbye_same_as_disconnect(
    coord: Coordinator, store: FakeStore, scheduler: FakeScheduler, clock
):
    store.put_node(make_node())
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.attempt_count = 1
    store.put_task(task)
    await coord.handle_message(PEER, msg(NODE_GOODBYE, WORKER, node_id=WORKER))
    assert scheduler.disconnects == [WORKER]
    assert store.get_task("task-1").state == "QUEUED"
    assert store.get_node(WORKER).status == "offline"


def test_transport_disconnect_callback_is_immediate(
    coord: Coordinator, transport: FakeTransport, store: FakeStore, scheduler: FakeScheduler, clock
):
    store.put_node(make_node())
    coord._map_peer(PEER, WORKER)
    task = make_task(clock)
    task.state = "LEASED"
    task.assigned_node = WORKER
    task.attempt_count = 1
    store.put_task(task)
    transport.fire_disconnect(PEER)
    assert scheduler.disconnects == [WORKER]
    assert store.get_task("task-1").state == "QUEUED"


# ---------------------------------------------------------------------------
# submit_job
# ---------------------------------------------------------------------------


async def test_submit_job_skips_when_jobs_module_missing(coord: Coordinator):
    jobs = pytest.importorskip("nova.jobs")
    load = getattr(jobs, "load_gallery", None)
    split = getattr(jobs, "split_job", None)
    if load is None or split is None:
        pytest.skip("nova.jobs splitter not available")

    job = await coord.submit_job("demo/gallery.yaml")
    assert job.state == "RUNNING"
    assert coord.store.get_job(job.job_id) is not None
    assert coord.store.list_tasks(job.job_id)
    assert any(e.type == JOB_ANNOUNCE for e in coord.transport.broadcasts)


async def test_submit_job_with_job_object_when_splitter_present(coord: Coordinator, clock, transport: FakeTransport):
    jobs = pytest.importorskip("nova.jobs")
    split = getattr(jobs, "split_job", None)
    if split is None:
        pytest.skip("nova.jobs.split_job not available")

    job = Job(
        job_id="gallery-1",
        created_at=clock.now(),
        prompts=[PromptSpec(text="a", seed=1), PromptSpec(text="b", seed=2)],
        requirements=JobRequirements(),
    )
    out = await coord.submit_job(job)
    assert out.state == "RUNNING"
    assert transport.broadcasts[-1].type == JOB_ANNOUNCE
    assert transport.broadcasts[-1].payload["http_url"] == "http://192.168.1.5:8080"
    assert transport.broadcasts[-1].payload["task_count"] == len(coord.store.list_tasks("gallery-1"))


async def test_ignore_self_messages(coord: Coordinator, bus: EventBus):
    await coord.handle_message(PEER, msg(HELLO, coord.node_id, node_id=coord.node_id))
    assert "HELLO" not in event_types(bus)
