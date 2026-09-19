"""Worker must not emit TASK_COMPLETE unless the PNG PUT was accepted."""

from __future__ import annotations

from typing import Any

import pytest

from nova.config import Settings
from nova.kernels.dummy import DummyKernel
from nova.models import Device, NodeIdentity
from nova.protocol import TASK_ACCEPT, TASK_COMPLETE, TASK_FAILED, TASK_OFFER, TASK_STARTED, Envelope, msg
from nova.worker import Worker
from tests.conftest import FakeClock


class RecordingTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Envelope]] = []
        self.broadcasts: list[Envelope] = []
        self._on_message = None
        self._on_connect = None
        self._on_disconnect = None

    def on_message(self, handler) -> None:
        self._on_message = handler

    def on_peer_connect(self, handler) -> None:
        self._on_connect = handler

    def on_peer_disconnect(self, handler) -> None:
        self._on_disconnect = handler

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, peer_id: str, env: Envelope) -> None:
        self.sent.append((peer_id, env))

    async def broadcast(self, env: Envelope) -> None:
        self.broadcasts.append(env)


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHttp:
    def __init__(self, response: FakeResponse | None = None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def put(self, url: str, content: bytes | None = None, headers: dict[str, str] | None = None) -> FakeResponse:
        self.calls.append({"url": url, "content": content, "headers": dict(headers or {})})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _worker(tmp_path, http: FakeHttp | None) -> tuple[Worker, RecordingTransport]:
    transport = RecordingTransport()
    worker = Worker(
        Settings(data_dir=tmp_path, dummy=True, heartbeat_s=60.0, progress_s=60.0),
        transport,
        identity=NodeIdentity(node_id="nova-worker", created_at=FakeClock().now()),
        kernel=DummyKernel(),
        devices=[
            Device(
                device_id="cpu:0",
                backend="cpu",
                vendor="cpu",
                model="cpu",
                memory_total_mb=4096,
            )
        ],
        http_client=http,
        coordinator_id="coord",
    )
    return worker, transport


def _offer(**overrides: Any) -> Envelope:
    payload = {
        "task_id": "t-1",
        "job_id": "job-1",
        "prompt": "neon street",
        "seed": 1000,
        "steps": 4,
        "width": 16,
        "height": 16,
        "allowed_backends": ["cpu"],
        "upload_url": "http://127.0.0.1:8080/jobs/job-1/tasks/t-1/result",
        "lease_gen": 3,
    }
    payload.update(overrides)
    return msg(TASK_OFFER, "coord", **payload)


def _types(transport: RecordingTransport) -> list[str]:
    return [env.type for _peer, env in transport.sent]


@pytest.mark.asyncio
async def test_accepted_put_sends_complete_with_node_id(tmp_path) -> None:
    http = FakeHttp(FakeResponse(200, {"status": "accepted"}))
    worker, transport = _worker(tmp_path, http)
    await worker.handle_message("coord", _offer())
    assert worker._compute_task is not None
    await worker._compute_task
    assert TASK_ACCEPT in _types(transport)
    assert TASK_STARTED in _types(transport)
    assert TASK_COMPLETE in _types(transport)
    assert TASK_FAILED not in _types(transport)
    assert len(http.calls) == 1
    headers = http.calls[0]["headers"]
    assert headers["X-Nova-Node-Id"] == "nova-worker"
    assert headers["X-Nova-Lease-Gen"] == "3"
    assert headers["X-Nova-Backend"] == "cpu"
    assert headers["X-Nova-Device-Id"] == "cpu:0"
    complete = next(env for _peer, env in transport.sent if env.type == TASK_COMPLETE)
    assert complete.payload["task_id"] == "t-1"
    assert complete.payload["lease_gen"] == 3
    assert complete.payload["sha256"] == headers["X-Nova-Sha256"]


@pytest.mark.asyncio
async def test_missing_upload_url_sends_failed_not_complete(tmp_path) -> None:
    http = FakeHttp(FakeResponse(200, {"status": "accepted"}))
    worker, transport = _worker(tmp_path, http)
    await worker.handle_message("coord", _offer(upload_url=""))
    assert worker._compute_task is not None
    await worker._compute_task
    assert http.calls == []
    assert TASK_FAILED in _types(transport)
    assert TASK_COMPLETE not in _types(transport)


@pytest.mark.asyncio
async def test_http_error_sends_failed_not_complete(tmp_path) -> None:
    http = FakeHttp(error=RuntimeError("connection reset"))
    worker, transport = _worker(tmp_path, http)
    await worker.handle_message("coord", _offer())
    assert worker._compute_task is not None
    await worker._compute_task
    assert TASK_FAILED in _types(transport)
    assert TASK_COMPLETE not in _types(transport)


def test_real_worker_refuses_cpu_only_without_dummy(tmp_path) -> None:
    worker, _transport = _worker(tmp_path, FakeHttp(FakeResponse(200, {"status": "accepted"})))
    worker.settings = worker.settings.model_copy(update={"dummy": False})
    try:
        worker._select_device()
    except RuntimeError as exc:
        assert "Metal" in str(exc)
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("cpu-only real worker must fail closed")


def test_dummy_worker_may_use_cpu(tmp_path) -> None:
    worker, _transport = _worker(tmp_path, FakeHttp(FakeResponse(200, {"status": "accepted"})))
    device = worker._select_device()
    assert device.backend == "cpu"


def test_real_worker_prefers_metal_over_cpu(tmp_path) -> None:
    worker, _transport = _worker(tmp_path, FakeHttp(FakeResponse(200, {"status": "accepted"})))
    worker.settings = worker.settings.model_copy(update={"dummy": False})
    worker.devices = [
        Device(
            device_id="cpu:0",
            backend="cpu",
            vendor="cpu",
            model="cpu",
            memory_total_mb=8192,
        ),
        Device(
            device_id="mps:0",
            backend="metal",
            vendor="apple",
            model="M3 Max",
            memory_total_mb=18432,
        ),
    ]
    assert worker._select_device().backend == "metal"


def test_real_worker_prefers_cuda_over_metal(tmp_path) -> None:
    worker, _transport = _worker(tmp_path, FakeHttp(FakeResponse(200, {"status": "accepted"})))
    worker.settings = worker.settings.model_copy(update={"dummy": False})
    worker.devices = [
        Device(
            device_id="mps:0",
            backend="metal",
            vendor="apple",
            model="M3 Max",
            memory_total_mb=18432,
        ),
        Device(
            device_id="cuda:0",
            backend="cuda",
            vendor="nvidia",
            model="RTX 4090",
            memory_total_mb=24576,
        ),
    ]
    assert worker._select_device().device_id == "cuda:0"


@pytest.mark.asyncio
async def test_ignored_put_sends_failed_not_complete(tmp_path) -> None:
    http = FakeHttp(FakeResponse(200, {"status": "ignored", "reason": "stale_lease_gen"}))
    worker, transport = _worker(tmp_path, http)
    await worker.handle_message("coord", _offer())
    assert worker._compute_task is not None
    await worker._compute_task
    assert TASK_FAILED in _types(transport)
    assert TASK_COMPLETE not in _types(transport)
    failed = next(env for _peer, env in transport.sent if env.type == TASK_FAILED)
    assert "not accepted" in failed.payload["error"]
