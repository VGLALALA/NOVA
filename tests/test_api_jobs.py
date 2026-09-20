"""Dashboard job dispatch: unique ids, gallery shortcut, newest job first."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from nova.api import _job_payload, _node_payload, create_app, split_gallery_doc
from nova.clock import now_utc
from nova.config import Settings
from nova.events import EventBus
from nova.main import FallbackScheduler, FallbackStore
from nova.models import Job, Task
from tests.fakes import make_node as cuda_node


def _client(tmp_path: Path) -> tuple[TestClient, FallbackStore]:
    store = FallbackStore(tmp_path)
    bus = EventBus()
    settings = Settings(data_dir=tmp_path, http_host="127.0.0.1")
    app = create_app(store, FallbackScheduler(store, bus), bus, settings, tmp_path)
    return TestClient(app), store


def test_split_gallery_doc_mints_unique_ids() -> None:
    doc = {
        "job": {"name": "nova-gallery", "kernel": "sd.t2i.v1"},
        "model": {"id": "stabilityai/sd-turbo", "steps": 4, "width": 512, "height": 512},
        "prompts": [{"text": "a", "seed": 1}, {"text": "b", "seed": 2}],
    }
    job_a, tasks_a = split_gallery_doc(doc)
    job_b, tasks_b = split_gallery_doc(doc)
    assert job_a.job_id != job_b.job_id
    assert job_a.job_id.startswith("nova-gallery-")
    assert len(tasks_a) == 2
    assert tasks_a[0].task_id.startswith(job_a.job_id)
    assert tasks_b[0].task_id.startswith(job_b.job_id)


def test_split_gallery_doc_honors_explicit_id() -> None:
    doc = {
        "job": {"name": "nova-gallery", "id": "fixed-id"},
        "prompts": [{"text": "a", "seed": 1}],
    }
    job, tasks = split_gallery_doc(doc)
    assert job.job_id == "fixed-id"
    assert tasks[0].task_id == "fixed-id-00"


def test_post_jobs_gallery_dispatches_24_tiles(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with client:
        first = client.post("/jobs/gallery")
        assert first.status_code == 201
        body = first.json()
        assert body["total"] == 24
        assert body["job_id"].startswith("nova-gallery-")
        assert body["state"] == "RUNNING"
        second = client.post("/jobs/gallery")
        assert second.status_code == 201
        assert second.json()["job_id"] != body["job_id"]
        listed = client.get("/jobs")
        assert listed.status_code == 200
        jobs = listed.json()
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == second.json()["job_id"]
        gallery = client.get("/gallery/default")
        assert gallery.status_code == 200
        assert gallery.json()["total"] == 24
    assert len(store.jobs) == 2


def test_post_custom_prompt_job(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with client:
        res = client.post(
            "/jobs",
            json={
                "job": {"name": "nova-custom", "kernel": "sd.t2i.v1"},
                "model": {"id": "stabilityai/sd-turbo", "steps": 4, "width": 512, "height": 512},
                "prompts": [{"text": "neon street", "seed": 1000}],
            },
        )
        assert res.status_code == 201
        body = res.json()
        assert body["total"] == 1
        assert body["job_id"].startswith("nova-custom-")
        tasks = client.get(f"/jobs/{body['job_id']}/tasks")
        assert tasks.status_code == 200
        assert tasks.json()[0]["prompt"] == "neon street"
    job = next(iter(store.jobs.values()))
    assert isinstance(job, Job)
    assert job.name == "nova-custom"


def test_post_jobs_gallery_respects_count(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with client:
        res = client.post("/jobs/gallery", json={"count": 4})
        assert res.status_code == 201
        body = res.json()
        assert body["total"] == 4
        tasks = client.get(f"/jobs/{body['job_id']}/tasks")
        assert len(tasks.json()) == 4
        bad = client.post("/jobs/gallery", json={"count": 0})
        assert bad.status_code == 400
    job = next(iter(store.jobs.values()))
    assert len(job.prompts) == 4


def test_gallery_quota_mode_reserves_nodes(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with client:
        res = client.post(
            "/jobs/gallery",
            json={
                "count": 4,
                "job": {"scheduler_policy": "quota", "quotas": {"n1": 3, "n2": 1}},
            },
        )
        assert res.status_code == 201
        job_id = res.json()["job_id"]
        tasks = client.get(f"/jobs/{job_id}/tasks").json()
        reserved = [t.get("reserved_node") for t in tasks]
        assert reserved.count("n1") == 3
        assert reserved.count("n2") == 1
    job = store.jobs[job_id]
    assert job.scheduler_policy == "quota"


def test_post_nodes_tcp_requires_coordinator(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    with client:
        res = client.post("/nodes/tcp", json={"host": "127.0.0.1:7946"})
        assert res.status_code == 503


def test_delete_node_marks_offline(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    store.put_node(cuda_node("n1"))
    with client:
        res = client.post("/nodes/n1/remove")
        assert res.status_code == 200
        assert client.get("/nodes").json() == []
        gone = client.delete("/nodes/n1")
        assert gone.status_code == 200


def test_node_payload_stale_last_seen_is_lost(tmp_path: Path) -> None:
    store = FallbackStore(tmp_path)
    node = cuda_node("ghost")
    store.put_node(node)
    payload = _node_payload(node, store, Settings(data_dir=tmp_path))
    assert payload["status"] == "offline"
    assert payload["display_status"] == "LOST"
    store.touch_node("ghost")
    live = _node_payload(store.get_node("ghost"), store, Settings(data_dir=tmp_path))
    assert live["status"] == "online"
    assert live["display_status"] == "ACTIVE"


def test_job_payload_elapsed_and_inference(tmp_path: Path) -> None:
    store = FallbackStore(tmp_path)
    now = now_utc()
    job = Job(
        job_id="j",
        created_at=now - timedelta(seconds=5),
        started_at=now - timedelta(seconds=4),
        finished_at=now,
        state="COMPLETED",
    )
    store.put_job(job)
    store.put_task(
        Task(
            task_id="j-00",
            job_id="j",
            shard_index=0,
            prompt="a",
            seed=1,
            state="COMPLETED",
            created_at=now,
            execution_ms=1234,
        )
    )
    data = _job_payload(store, job)
    assert data["elapsed_ms"] is not None
    assert 3500 <= data["elapsed_ms"] <= 4500
    assert data["inference_ms"] == 1234
