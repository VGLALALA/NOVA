"""Dashboard job dispatch: unique ids, gallery shortcut, newest job first."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nova.api import create_app, split_gallery_doc
from nova.config import Settings
from nova.events import EventBus
from nova.main import FallbackScheduler, FallbackStore
from nova.models import Job


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
