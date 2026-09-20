from __future__ import annotations

from pathlib import Path

from nova.jobs import load_and_split
from nova.models import Device, Job, JobRequirements, NodeManifest, Task
from nova.scheduler import Scheduler
from nova.store import Store
from tests.conftest import FakeClock


def _cuda_node(node_id: str, score: float = 0.5) -> NodeManifest:
    return NodeManifest(
        node_id=node_id,
        devices=[
            Device(
                device_id="cuda:0",
                backend="cuda",
                vendor="nvidia",
                model="RTX",
                memory_total_mb=24576,
            )
        ],
        supported_kernels=["sd.t2i.v1"],
        max_concurrency=1,
        current_slots_used=0,
        status="online",
        benchmark_scores={"sd.t2i.v1": score},
    )


def test_fifo_pull(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    store.put_node(_cuda_node("n1"))
    job = Job(job_id="j", created_at=clock.now(), requirements=JobRequirements(), prompts=[])
    store.put_job(job)
    for i in range(3):
        store.put_task(
            Task(
                task_id=f"j-{i:02d}",
                job_id="j",
                shard_index=i,
                prompt=str(i),
                seed=i,
                created_at=clock.now(),
                min_memory_mb=4096,
                allowed_backends=["cuda", "rocm", "metal"],
            )
        )
    first = sched.on_work_request("n1")
    assert first is not None and first.shard_index == 0


def test_poison_blacklist_one_cycle(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    store.put_node(_cuda_node("n1"))
    store.put_node(_cuda_node("n2"))
    job = Job(job_id="j", created_at=clock.now(), requirements=JobRequirements(), prompts=[])
    store.put_job(job)
    store.put_task(
        Task(
            task_id="j-00",
            job_id="j",
            shard_index=0,
            prompt="a",
            seed=1,
            created_at=clock.now(),
            min_memory_mb=4096,
            allowed_backends=["cuda"],
        )
    )
    store.put_task(
        Task(
            task_id="j-01",
            job_id="j",
            shard_index=1,
            prompt="b",
            seed=2,
            created_at=clock.now(),
            min_memory_mb=4096,
            allowed_backends=["cuda"],
        )
    )
    t = sched.on_work_request("n1")
    assert t is not None and t.task_id == "j-00"
    sched.on_task_failed("n1", "j-00", t.lease_gen, reason="oom")
    nxt = sched.on_work_request("n1")
    assert nxt is not None and nxt.task_id == "j-01"


def test_disconnect_requeues_immediately(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    store.put_node(_cuda_node("amd"))
    store.put_node(_cuda_node("nv"))
    job = Job(job_id="j", created_at=clock.now(), requirements=JobRequirements(), prompts=[])
    store.put_job(job)
    store.put_task(
        Task(
            task_id="j-00",
            job_id="j",
            shard_index=0,
            prompt="a",
            seed=1,
            created_at=clock.now(),
            min_memory_mb=4096,
            allowed_backends=["cuda"],
        )
    )
    leased = sched.on_work_request("amd")
    assert leased is not None
    gen = leased.lease_gen
    requeued = sched.on_disconnect("amd")
    assert [t.task_id for t in requeued] == ["j-00"]
    t = store.get_task("j-00")
    assert t is not None
    assert t.state == "QUEUED"
    assert t.assigned_node is None
    # do not need to advance 30s
    nxt = sched.on_work_request("nv")
    assert nxt is not None and nxt.task_id == "j-00"
    assert nxt.lease_gen == gen + 1


def test_never_seen_node_is_offline(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    store.put_node(_cuda_node("ghost"))
    offlined = sched.check_node_liveness()
    assert "ghost" in offlined
    node = store.get_node("ghost")
    assert node is not None and node.status == "offline"


def test_split_gallery_yaml_24_tasks(clock: FakeClock) -> None:
    job, tasks = load_and_split("demo/gallery.yaml", clock)
    assert len(job.prompts) == 24
    assert len(tasks) == 24
    assert [t.shard_index for t in tasks] == list(range(24))
    assert tasks[0].task_id.endswith("-00")
    assert tasks[0].max_attempts == 3
    assert tasks[0].state == "QUEUED"


def test_split_gallery_count_truncates(clock: FakeClock) -> None:
    job, tasks = load_and_split("demo/gallery.yaml", clock, count=6)
    assert len(job.prompts) == 6
    assert len(tasks) == 6
    assert [t.shard_index for t in tasks] == list(range(6))


def test_quota_pick_respects_reservation(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    sched = Scheduler(store, clock=clock)
    store.put_node(_cuda_node("n1", 0.9))
    store.put_node(_cuda_node("n2", 0.2))
    job = Job(
        job_id="j",
        created_at=clock.now(),
        scheduler_policy="quota",
        quotas={"n1": 1, "n2": 1},
        requirements=JobRequirements(),
        prompts=[],
    )
    store.put_job(job)
    for i, reserved in enumerate(("n1", "n2")):
        store.put_task(
            Task(
                task_id=f"j-{i:02d}",
                job_id="j",
                shard_index=i,
                prompt=str(i),
                seed=i,
                created_at=clock.now(),
                reserved_node=reserved,
            )
        )
    first = sched.on_work_request("n2")
    assert first is not None and first.reserved_node == "n2"
    second = sched.on_work_request("n2")
    assert second is None
    other = sched.on_work_request("n1")
    assert other is not None and other.reserved_node == "n1"
