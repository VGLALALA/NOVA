"""Result PUT: first valid lease_gen wins. No GPU required."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from nova.api import create_app
from nova.config import Settings
from nova.events import EventBus
from nova.models import Device, Job, JobRequirements, NodeManifest, PromptSpec, Task
from nova.scheduler import Scheduler
from nova.store import Store
from tests.conftest import FakeClock

NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)


def _png(color: tuple[int, int, int] = (30, 80, 40), size: tuple[int, int] = (16, 16)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeStore:
    def __init__(self, data_dir: Path) -> None:
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, Task] = {}
        self.nodes: dict[str, object] = {}
        self.tiles_dir = data_dir / "tiles"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.saved: list[tuple[str, str, bytes]] = []

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    def list_tasks(self, job_id: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.job_id == job_id]

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def list_nodes(self) -> list[object]:
        return list(self.nodes.values())

    def save_result(self, task: Task, png_bytes: bytes, sha256: str) -> Path:
        dest = self.tiles_dir / task.job_id
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{task.task_id}.png"
        path.write_bytes(png_bytes)
        task.result_sha256 = sha256
        task.result_path = str(path)
        task.state = "COMPLETED"
        task.completed_at = NOW
        self.tasks[task.task_id] = task
        self.saved.append((task.task_id, sha256, png_bytes))
        return path


class FakeScheduler:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.accepted: list[tuple[str, str, int, str]] = []

    def accept_result(
        self,
        job_id: str,
        task_id: str,
        png_bytes: bytes,
        lease_gen: int,
        sha256: str,
        node_id: str | None = None,
    ) -> dict[str, str]:
        self.accepted.append((job_id, task_id, lease_gen, sha256))
        return {"status": "accepted"}

    def on_complete(self, *args: object, **kwargs: object) -> None:
        return None


def _seed(store: FakeStore, *, lease_gen: int = 1, state: str = "LEASED") -> Task:
    job = Job(
        job_id="j1",
        name="nova-gallery",
        created_at=NOW,
        requirements=JobRequirements(),
        prompts=[PromptSpec(text="test", seed=1000)],
        state="RUNNING",
    )
    task = Task(
        task_id="t1",
        job_id="j1",
        shard_index=0,
        prompt="test",
        seed=1000,
        state=state,  # type: ignore[arg-type]
        assigned_node="nova-test",
        lease_gen=lease_gen,
        created_at=NOW,
        backend_used="cuda",
    )
    store.jobs[job.job_id] = job
    store.tasks[task.task_id] = task
    return task


def _client(tmp_path: Path, store: FakeStore) -> TestClient:
    settings = Settings(data_dir=tmp_path, http_host="127.0.0.1")
    app = create_app(store, FakeScheduler(store), EventBus(), settings, tmp_path)
    return TestClient(app)


def test_put_valid_png_matching_lease(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store, lease_gen=1, state="LEASED")
    png = _png((10, 20, 30))
    digest = hashlib.sha256(png).hexdigest()
    with _client(tmp_path, store) as client:
        res = client.put(
            "/jobs/j1/tasks/t1/result",
            content=png,
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "accepted"
        assert body["sha256"] == digest
        tile = client.get("/jobs/j1/tiles/t1.png")
        assert tile.status_code == 200
        assert tile.content == png
    assert store.tasks["t1"].state == "COMPLETED"
    assert len(store.saved) == 1


def test_put_real_scheduler_preserves_metadata_and_releases_slot(
    clock: FakeClock, tmp_path: Path
) -> None:
    store = Store(tmp_path)
    bus = EventBus()
    scheduler = Scheduler(store, clock=clock, event_bus=bus)
    node = NodeManifest(
        node_id="nova-real",
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
        status="online",
    )
    scheduler.register_node(node)
    store.put_job(
        Job(
            job_id="j-real",
            name="real-put",
            created_at=clock.now(),
            requirements=JobRequirements(
                min_memory_mb=4096,
                allowed_backends=["cuda"],
                width=16,
                height=16,
            ),
            prompts=[PromptSpec(text="test", seed=1000)],
            state="RUNNING",
        )
    )
    store.put_task(
        Task(
            task_id="t-real",
            job_id="j-real",
            shard_index=0,
            prompt="test",
            seed=1000,
            width=16,
            height=16,
            min_memory_mb=4096,
            allowed_backends=["cuda"],
            created_at=clock.now(),
        )
    )

    leased = scheduler.on_work_request(node.node_id)
    assert leased is not None
    assert leased.task_id == "t-real"
    assert leased.state == "LEASED"
    leased_node = store.get_node(node.node_id)
    assert leased_node is not None
    assert leased_node.current_slots_used == 1

    png = _png((20, 40, 60), size=(16, 16))
    digest = hashlib.sha256(png).hexdigest()
    settings = Settings(data_dir=tmp_path, http_host="127.0.0.1")
    app = create_app(store, scheduler, bus, settings, tmp_path)
    with TestClient(app) as client:
        res = client.put(
            "/jobs/j-real/tasks/t-real/result",
            content=png,
            headers={
                "X-Nova-Lease-Gen": str(leased.lease_gen),
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": node.node_id,
                "X-Nova-Backend": "cuda",
                "X-Nova-Device-Id": "cuda:0",
                "X-Nova-Execution-Ms": "1234",
            },
        )

    assert res.status_code == 200
    assert res.json()["status"] == "accepted"
    completed = store.get_task("t-real")
    assert completed is not None
    assert completed.state == "COMPLETED"
    assert completed.assigned_node == node.node_id
    assert completed.backend_used == "cuda"
    assert completed.device_id_used == "cuda:0"
    assert completed.execution_ms == 1234
    assert completed.result_sha256 == digest
    assert store.tile_path("j-real", "t-real").read_bytes() == png
    completed_node = store.get_node(node.node_id)
    assert completed_node is not None
    assert completed_node.current_slots_used == 0


def test_put_garbage_rejected(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store)
    with _client(tmp_path, store) as client:
        res = client.put(
            "/jobs/j1/tasks/t1/result",
            content=b"this is not a png",
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": "abc",
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert res.status_code == 400
    assert store.tasks["t1"].state == "LEASED"
    assert store.saved == []
    assert not (store.tiles_dir / "j1" / "t1.png").exists()


def test_put_stale_lease_does_not_overwrite_first_winner(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store, lease_gen=1, state="LEASED")
    winner = _png((200, 10, 10))
    loser = _png((10, 10, 200))
    winner_sha = hashlib.sha256(winner).hexdigest()
    loser_sha = hashlib.sha256(loser).hexdigest()
    with _client(tmp_path, store) as client:
        first = client.put(
            "/jobs/j1/tasks/t1/result",
            content=winner,
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": winner_sha,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert first.status_code == 200
        assert first.json()["status"] == "accepted"

        stale = client.put(
            "/jobs/j1/tasks/t1/result",
            content=loser,
            headers={
                "X-Nova-Lease-Gen": "0",
                "X-Nova-Sha256": loser_sha,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert stale.status_code in (200, 409)
        if stale.status_code == 200:
            assert stale.json()["status"] == "ignored"

        dup = client.put(
            "/jobs/j1/tasks/t1/result",
            content=loser,
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": loser_sha,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert dup.status_code in (200, 409)
        if dup.status_code == 200:
            assert dup.json()["status"] == "ignored"

        tile = client.get("/jobs/j1/tiles/t1.png")
        assert tile.status_code == 200
        assert tile.content == winner
        assert tile.content != loser
    assert len(store.saved) == 1
    assert store.saved[0][2] == winner


def test_put_empty_and_missing_task(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store)
    png = _png()
    digest = hashlib.sha256(png).hexdigest()
    with _client(tmp_path, store) as client:
        empty = client.put(
            "/jobs/j1/tasks/t1/result",
            content=b"",
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert empty.status_code == 400
        missing = client.put(
            "/jobs/j1/tasks/nope/result",
            content=png,
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": "nova-test",
            },
        )
        assert missing.status_code == 404
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"


def test_put_missing_node_id_rejected(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store)
    png = _png()
    digest = hashlib.sha256(png).hexdigest()
    with _client(tmp_path, store) as client:
        res = client.put(
            "/jobs/j1/tasks/t1/result",
            content=png,
            headers={"X-Nova-Lease-Gen": "1", "X-Nova-Sha256": digest},
        )
        assert res.status_code == 400
        assert "X-Nova-Node-Id" in res.json()["detail"]
    assert store.tasks["t1"].state == "LEASED"
    assert store.saved == []


def test_put_wrong_node_id_ignored(tmp_path: Path) -> None:
    store = FakeStore(tmp_path)
    _seed(store)
    png = _png((9, 9, 9))
    digest = hashlib.sha256(png).hexdigest()
    with _client(tmp_path, store) as client:
        res = client.put(
            "/jobs/j1/tasks/t1/result",
            content=png,
            headers={
                "X-Nova-Lease-Gen": "1",
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": "someone-else",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ignored"
        assert body["reason"] == "wrong_node"
        missing = client.get("/jobs/j1/tiles/t1.png")
        assert missing.status_code == 404
    assert store.tasks["t1"].state == "LEASED"
    assert store.saved == []


def test_put_real_scheduler_rejects_wrong_uploader(clock: FakeClock, tmp_path: Path) -> None:
    store = Store(tmp_path)
    bus = EventBus()
    scheduler = Scheduler(store, clock=clock, event_bus=bus)
    node = NodeManifest(
        node_id="nova-real",
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
        status="online",
    )
    scheduler.register_node(node)
    store.put_job(
        Job(
            job_id="j-fence",
            name="fence-put",
            created_at=clock.now(),
            requirements=JobRequirements(
                min_memory_mb=4096,
                allowed_backends=["cuda"],
                width=16,
                height=16,
            ),
            prompts=[PromptSpec(text="test", seed=1000)],
            state="RUNNING",
        )
    )
    store.put_task(
        Task(
            task_id="t-fence",
            job_id="j-fence",
            shard_index=0,
            prompt="test",
            seed=1000,
            width=16,
            height=16,
            min_memory_mb=4096,
            allowed_backends=["cuda"],
            created_at=clock.now(),
        )
    )
    leased = scheduler.on_work_request(node.node_id)
    assert leased is not None
    png = _png((1, 2, 3), size=(16, 16))
    digest = hashlib.sha256(png).hexdigest()
    settings = Settings(data_dir=tmp_path, http_host="127.0.0.1")
    app = create_app(store, scheduler, bus, settings, tmp_path)
    with TestClient(app) as client:
        res = client.put(
            "/jobs/j-fence/tasks/t-fence/result",
            content=png,
            headers={
                "X-Nova-Lease-Gen": str(leased.lease_gen),
                "X-Nova-Sha256": digest,
                "X-Nova-Node-Id": "impostor",
                "X-Nova-Backend": "cuda",
            },
        )
    assert res.status_code == 200
    assert res.json()["status"] == "ignored"
    current = store.get_task("t-fence")
    assert current is not None
    assert current.state == "LEASED"
    assert current.assigned_node == node.node_id
    assert not store.tile_path("j-fence", "t-fence").exists()
    held = store.get_node(node.node_id)
    assert held is not None
    assert held.current_slots_used == 1
