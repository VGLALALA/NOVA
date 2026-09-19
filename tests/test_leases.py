from __future__ import annotations

from nova.clock import now_utc
from nova.models import Device, Job, JobRequirements, NodeManifest, Task
from nova.scheduler import Scheduler
from nova.store import Store
from tests.conftest import FakeClock


def _setup(clock: FakeClock, tmp_path):
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    node = NodeManifest(
        node_id="n1",
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
        max_concurrency=1,
        current_slots_used=0,
        status="online",
        benchmark_scores={"sd.t2i.v1": 0.5},
    )
    store.put_node(node)
    store.touch_node("n1", clock.now())
    job = Job(
        job_id="j",
        created_at=clock.now(),
        requirements=JobRequirements(allowed_backends=["cuda", "rocm", "metal"]),
        prompts=[],
    )
    store.put_job(job)
    task = Task(
        task_id="j-00",
        job_id="j",
        shard_index=0,
        prompt="neon",
        seed=1,
        created_at=clock.now(),
        min_memory_mb=4096,
        allowed_backends=["cuda", "rocm", "metal"],
        max_attempts=3,
    )
    store.put_task(task)
    return store, sched, task


def test_grant_increments_attempt_and_lease_gen(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    leased = sched.grant_lease("n1", task, lease_seconds=20)
    assert leased.attempt_count == 1
    assert leased.lease_gen == 1
    assert leased.state == "LEASED"
    assert store.get_node("n1").current_slots_used == 1


def test_expiry_requeues_without_incrementing_attempt(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    sched.grant_lease("n1", task, lease_seconds=20)
    clock.advance(21)
    expired = sched.expire_leases()
    assert len(expired) == 1
    t = store.get_task("j-00")
    assert t.state == "QUEUED"
    assert t.attempt_count == 1
    assert t.assigned_node is None
    assert store.get_node("n1").current_slots_used == 0


def test_three_leases_then_failed(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    for _ in range(3):
        t = store.get_task("j-00")
        sched.grant_lease("n1", t, lease_seconds=20)
        clock.advance(21)
        sched.expire_leases()
    t = store.get_task("j-00")
    assert t.attempt_count == 3
    assert t.state == "FAILED"


def test_stale_lease_gen_complete_ignored(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    sched.grant_lease("n1", task, lease_seconds=20)
    gen = store.get_task("j-00").lease_gen
    clock.advance(21)
    sched.expire_leases()
    t = store.get_task("j-00")
    sched.grant_lease("n1", t, lease_seconds=20)
    assert sched.on_complete("n1", "j-00", gen, sha256="abc") is False
    assert store.get_task("j-00").state == "LEASED"


def test_first_valid_complete_wins(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    sched.grant_lease("n1", task, lease_seconds=20)
    gen = store.get_task("j-00").lease_gen
    assert sched.on_complete("n1", "j-00", gen, sha256="aa", backend="cuda") is True
    assert store.get_task("j-00").state == "COMPLETED"
    assert sched.on_complete("n1", "j-00", gen, sha256="bb") is False


def test_progress_renews_lease_heartbeat_does_not(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    sched.grant_lease("n1", task, lease_seconds=20)
    expires = store.get_task("j-00").lease_expires_at
    sched.on_heartbeat("n1")
    assert store.get_task("j-00").lease_expires_at == expires
    gen = store.get_task("j-00").lease_gen
    clock.advance(5)
    assert sched.on_progress("n1", "j-00", gen, lease_seconds=20) is True
    assert store.get_task("j-00").lease_expires_at > expires
    assert store.get_task("j-00").state == "RUNNING"


def test_garbage_png_rejected(clock: FakeClock, tmp_path) -> None:
    store, sched, task = _setup(clock, tmp_path)
    sched.grant_lease("n1", task, lease_seconds=20)
    import pytest

    with pytest.raises(ValueError):
        store.save_result(task, b"not-a-png", "deadbeef")
    with pytest.raises(ValueError):
        store.save_result(task, b"", "deadbeef")
    assert store.get_task("j-00").state == "LEASED"
