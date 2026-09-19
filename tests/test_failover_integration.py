"""Live-demo failover contract.

Two in-process workers, dummy tiles (no GPU). Ctrl+C worker A →
NODE_DISCONNECTED requeues immediately (not the 30s heartbeat path) →
worker B finishes → stale lease_gen from A is ignored.

Prefers nova.scheduler / nova.store / nova.network.inprocess when they
import; otherwise uses tests.fakes.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeClock
from tests.fakes import FakePullWorker, dummy_job, resolve_stack

STACK = resolve_stack()
Scheduler = STACK.Scheduler
Store = STACK.Store
InProcessBroker = STACK.InProcessBroker
InProcessTransport = STACK.InProcessTransport
EventBus = STACK.EventBus
USING_REAL_SCHEDULER = STACK.using_real_scheduler
USING_REAL_INPROCESS = STACK.using_real_inprocess

N_TASKS = 8


def _stack(clock: FakeClock, data_dir: Path):
    store = Store(data_dir)
    bus = EventBus()
    sched = Scheduler(store, clock, event_bus=bus)
    return sched, store, bus


async def _mesh(sched):
    """Coordinator + two worker transports. Worker stop() → scheduler.on_disconnect."""
    broker = InProcessBroker()
    coord = InProcessTransport(broker, "coord")
    ta = InProcessTransport(broker, "worker-a")
    tb = InProcessTransport(broker, "worker-b")

    async def on_disconnect(peer_id: str) -> None:
        sched.on_disconnect(peer_id)

    coord.on_peer_disconnect(on_disconnect)
    await coord.start()
    await ta.start()
    await tb.start()
    return coord, ta, tb


async def test_disconnect_requeues_without_waiting_30s(clock: FakeClock, data_dir: Path) -> None:
    """Ctrl+C / connection_lost requeues mid-job tiles immediately. B finishes the job."""
    sched, store, bus = _stack(clock, data_dir)
    coord_t, a_t, b_t = await _mesh(sched)
    t0 = clock.now()
    try:
        worker_a = FakePullWorker(
            "worker-a", sched, backend="cuda", vendor="nvidia", model="RTX 4090", transport=a_t
        )
        worker_b = FakePullWorker(
            "worker-b", sched, backend="rocm", vendor="amd", model="RX 7900 XTX", transport=b_t
        )
        tasks = sched.submit_job(dummy_job(clock, n=N_TASKS))
        assert len(tasks) == N_TASKS

        # ~40% through: each worker completes two tiles, then A is mid-generation.
        for _ in range(2):
            assert worker_a.pull() is not None
            assert worker_a.complete(color=(1, 2, 3))
            assert worker_b.pull() is not None
            assert worker_b.complete(color=(3, 2, 1))

        held = worker_a.pull()
        assert held is not None
        assert worker_a.start_running()
        stale_id = held.task_id
        stale_gen = held.lease_gen
        attempts = held.attempt_count
        assert attempts == 1
        assert worker_b.pull() is not None
        assert worker_b.start_running()

        # Kill A. Do not advance the 30s heartbeat/offline clock.
        await worker_a.kill()
        assert clock.now() == t0

        requeued = store.get_task(stale_id)
        assert requeued is not None
        assert requeued.state == "QUEUED"
        assert requeued.assigned_node is None
        assert requeued.attempt_count == attempts  # not incremented on disconnect
        assert requeued.lease_gen == stale_gen  # bumps on the next grant
        assert store.get_node("worker-a") is not None
        assert store.get_node("worker-a").status == "offline"

        event_types = [e["type"] for e in bus.history()]
        assert "NODE_DISCONNECTED" in event_types

        # Heartbeat liveness was not the path: we never ticked 30s.
        sched.tick()
        assert store.get_task(stale_id).state == "QUEUED"

        # Stale complete from A with the old gen is ignored (task not leased to A).
        assert worker_a.complete_stale(stale_id, stale_gen) is False
        assert store.get_task(stale_id).state == "QUEUED"

        # B finishes its current tile, then the requeued one, then the rest.
        n = worker_b.drain()
        assert n >= 1
        retry = store.get_task(stale_id)
        assert retry is not None
        assert retry.state == "COMPLETED"
        assert retry.assigned_node == "worker-b"
        assert retry.lease_gen == stale_gen + 1
        assert retry.backend_used == "rocm"

        # Late PUT from A after B won is still ignored.
        assert worker_a.complete_stale(stale_id, stale_gen) is False
        assert store.get_task(stale_id).assigned_node == "worker-b"
        assert store.get_task(stale_id).backend_used == "rocm"

        all_tasks = store.list_tasks("failover")
        assert len(all_tasks) == N_TASKS
        assert {t.state for t in all_tasks} == {"COMPLETED"}
        assert store.get_job("failover") is not None
        assert store.get_job("failover").state == "COMPLETED"
        for task in all_tasks:
            assert task.result_path is not None
            assert Path(task.result_path).is_file()
    finally:
        await a_t.stop()
        await b_t.stop()
        await coord_t.stop()


async def test_stale_lease_gen_ignored(clock: FakeClock, data_dir: Path) -> None:
    """After failover, A's old lease_gen cannot steal the tile from B."""
    sched, store, _bus = _stack(clock, data_dir)
    coord_t, a_t, b_t = await _mesh(sched)
    try:
        worker_a = FakePullWorker("worker-a", sched, backend="cuda", vendor="nvidia", transport=a_t)
        worker_b = FakePullWorker(
            "worker-b", sched, backend="rocm", vendor="amd", model="RX 7900 XTX", transport=b_t
        )
        sched.submit_job(dummy_job(clock, n=4, job_id="stale"))

        held = worker_a.pull()
        assert held is not None
        stale_id = held.task_id
        stale_gen = held.lease_gen

        await worker_a.kill()
        assert store.get_task(stale_id).state == "QUEUED"

        taken = worker_b.pull()
        assert taken is not None
        assert taken.task_id == stale_id
        assert taken.lease_gen == stale_gen + 1
        assert worker_b.start_running()

        assert worker_a.complete_stale(stale_id, stale_gen) is False
        current = store.get_task(stale_id)
        assert current is not None
        assert current.state == "RUNNING"
        assert current.assigned_node == "worker-b"

        assert worker_b.complete(color=(4, 5, 6))
        won = store.get_task(stale_id)
        assert won is not None
        assert won.state == "COMPLETED"
        assert won.assigned_node == "worker-b"
        assert won.lease_gen == stale_gen + 1
        assert worker_a.complete_stale(stale_id, stale_gen) is False
        assert store.get_task(stale_id).result_sha256 == won.result_sha256
    finally:
        await a_t.stop()
        await b_t.stop()
        await coord_t.stop()


async def test_heartbeats_do_not_renew_lease(clock: FakeClock, data_dir: Path) -> None:
    """A hung GPU that still pings must still expire. Offline 30s path is not required."""
    sched, store, _bus = _stack(clock, data_dir)
    worker = FakePullWorker("worker-a", sched, backend="cuda", vendor="nvidia")
    sched.submit_job(dummy_job(clock, n=2, job_id="hb"))

    held = worker.pull()
    assert held is not None
    task_id = held.task_id
    original = store.get_task(task_id).lease_expires_at
    assert original is not None
    granted_at = clock.now()

    # Keep the node alive past the lease. Heartbeat must not extend lease_expires_at.
    while clock.now() <= original:
        clock.advance(5)
        sched.on_heartbeat("worker-a")
        assert store.get_task(task_id).lease_expires_at == original
        assert store.get_task(task_id).state in {"LEASED", "RUNNING"}

    # FakeClock only: no wall-clock sleep. Elapsed fake time is the lease (~20s), not 30s offline.
    assert (clock.now() - granted_at).total_seconds() < 30

    sched.tick()
    task = store.get_task(task_id)
    assert task is not None
    assert task.state == "QUEUED"
    assert task.assigned_node is None
    node = store.get_node("worker-a")
    assert node is not None
    assert node.status == "online"  # heartbeats prevented NODE_OFFLINE
