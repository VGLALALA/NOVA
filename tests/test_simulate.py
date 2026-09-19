from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from nova.models import KERNEL_SD_T2I
from nova.protocol import (
    HELLO,
    HEARTBEAT,
    NODE_GOODBYE,
    NODE_MANIFEST,
    TASK_ACCEPT,
    TASK_COMPLETE,
    TASK_FAILED,
    TASK_OFFER,
    TASK_STARTED,
    WORK_REQUEST,
    Envelope,
    msg,
)
from nova.simulate import (
    InProcessBroker,
    SimulatedWorker,
    build_manifests,
    build_worker_profiles,
    generate_dummy_png,
    kill_worker,
    run_simulated_workers,
    stop_graceful,
)


async def wait_for(
    predicate,
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(interval)


class FakeAcceptedHttp:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def put(self, url, content=None, headers=None, **_kwargs):
        self.calls.append({"url": url, "content": content, "headers": dict(headers or {})})
        return FakeAcceptedResponse()


class FakeAcceptedResponse:
    status_code = 200
    is_success = True

    def json(self) -> dict[str, str]:
        return {"status": "accepted"}


def envelopes_of(received: list[Envelope], type_: str, from_id: str | None = None) -> list[Envelope]:
    out = [e for e in received if e.type == type_]
    if from_id is not None:
        out = [e for e in out if e.from_id == from_id]
    return out


@pytest.fixture
def recorder() -> tuple[InProcessBroker, list[tuple[str, Envelope]], list[str]]:
    broker = InProcessBroker()
    coord = broker.attach("coordinator", coordinator=True)
    received: list[Envelope] = []
    disconnects: list[str] = []

    async def on_message(peer_id: str, env: Envelope) -> None:
        received.append(env)

    async def on_disconnect(peer_id: str) -> None:
        disconnects.append(peer_id)

    coord.on_message(on_message)
    coord.on_peer_disconnect(on_disconnect)
    return broker, received, disconnects


def test_profiles_cycle_cuda_rocm_metal() -> None:
    profiles = build_worker_profiles(3)
    assert [p.backend for p in profiles] == ["cuda", "rocm", "metal"]
    assert [p.model for p in profiles] == ["RTX 4090", "RX 7900 XTX", "M3 Max"]
    scores = [p.score for p in profiles]
    assert scores[0] == pytest.approx(1000 / 1200, rel=1e-3)
    assert scores[1] == pytest.approx(0.35, rel=1e-2)
    assert scores[2] == pytest.approx(0.15, rel=1e-2)
    assert scores[0] > scores[1] > scores[2]


def test_profiles_n_two_and_wrap() -> None:
    two = build_worker_profiles(2)
    assert [p.backend for p in two] == ["cuda", "rocm"]
    four = build_worker_profiles(4)
    assert [p.backend for p in four] == ["cuda", "rocm", "metal", "cuda"]
    assert four[0].node_id != four[3].node_id
    custom = build_worker_profiles(2, backends=["metal", "cuda"])
    assert [p.backend for p in custom] == ["metal", "cuda"]
    assert custom[0].model == "M3 Max"


def test_manifests_are_scheduler_shaped() -> None:
    manifests = build_manifests(3)
    cuda = manifests[0]
    assert cuda.node_id == "nova-sim-cuda-0"
    assert cuda.devices[0].backend == "cuda"
    assert cuda.devices[0].vendor == "nvidia"
    assert cuda.devices[0].model == "RTX 4090"
    assert cuda.devices[0].memory_total_mb >= 4096
    assert KERNEL_SD_T2I in cuda.supported_kernels
    assert cuda.benchmark_scores[KERNEL_SD_T2I] > manifests[2].benchmark_scores[KERNEL_SD_T2I]
    assert cuda.max_concurrency == 1
    metal = manifests[2]
    assert metal.devices[0].device_id == "mps:0"
    assert metal.os == "Darwin"


def test_kill_flag_without_start() -> None:
    broker = InProcessBroker()
    profile = build_worker_profiles(1)[0]
    worker = SimulatedWorker(profile, broker.attach(profile.node_id))
    assert worker.killed is False
    assert worker.alive is False


@pytest.mark.asyncio
async def test_kill_before_start_sets_flag() -> None:
    broker = InProcessBroker()
    profile = build_worker_profiles(1)[0]
    worker = SimulatedWorker(profile, broker.attach(profile.node_id))
    await worker.kill()
    assert worker.killed is True
    assert worker.alive is False


def test_dummy_png_decodes() -> None:
    raw = generate_dummy_png(64, 48, prompt="neon street", seed=1000, backend="cuda", model="RTX 4090")
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(raw))
    assert img.size == (64, 48)
    assert img.format == "PNG"


@pytest.mark.asyncio
async def test_two_simulated_workers_hello_and_manifest(recorder) -> None:
    broker, received, _disconnects = recorder
    cluster = await run_simulated_workers(
        broker, 2, heartbeat_s=0.05, pull_s=0.05, time_scale=0.0
    )
    try:
        await wait_for(lambda: len(envelopes_of(received, HELLO)) >= 2)
        await wait_for(lambda: len(envelopes_of(received, NODE_MANIFEST)) >= 2)
        hellos = {e.from_id for e in envelopes_of(received, HELLO)}
        manifests = envelopes_of(received, NODE_MANIFEST)
        assert hellos == {w.node_id for w in cluster.workers}
        backends = {e.payload["devices"][0]["backend"] for e in manifests}
        models = {e.payload["devices"][0]["model"] for e in manifests}
        assert backends == {"cuda", "rocm"}
        assert models == {"RTX 4090", "RX 7900 XTX"}
        await wait_for(lambda: len(envelopes_of(received, WORK_REQUEST)) >= 2)
    finally:
        await cluster.stop_all()


@pytest.mark.asyncio
async def test_empty_upload_url_fails_closed_without_complete(recorder) -> None:
    broker, received, _disconnects = recorder
    cluster = await run_simulated_workers(
        broker, 1, heartbeat_s=0.05, pull_s=0.05, time_scale=0.0
    )
    try:
        worker = cluster.workers[0]
        await wait_for(lambda: envelopes_of(received, WORK_REQUEST, worker.node_id))
        coord = broker.get("coordinator")
        assert coord is not None
        await coord.send(
            worker.node_id,
            msg(
                TASK_OFFER,
                "coordinator",
                task_id="t-fail",
                job_id="job-1",
                prompt="misty pine forest",
                seed=1002,
                steps=4,
                width=64,
                height=64,
                lease_seconds=20,
                lease_gen=1,
                upload_url="",
            ),
        )
        await wait_for(lambda: envelopes_of(received, TASK_FAILED, worker.node_id))
        types = [e.type for e in received if e.from_id == worker.node_id]
        assert TASK_ACCEPT in types
        assert TASK_STARTED in types
        assert TASK_COMPLETE not in types
        failed = envelopes_of(received, TASK_FAILED, worker.node_id)[0]
        assert failed.payload["task_id"] == "t-fail"
        assert "upload" in str(failed.payload.get("error") or "").lower()
    finally:
        await cluster.stop_all()


@pytest.mark.asyncio
async def test_task_offer_accept_and_complete(recorder) -> None:
    broker, received, _disconnects = recorder
    http = FakeAcceptedHttp()
    cluster = await run_simulated_workers(
        broker, 2, heartbeat_s=0.05, pull_s=0.05, time_scale=0.0, http_client=http
    )
    try:
        worker = cluster.workers[0]
        await wait_for(lambda: envelopes_of(received, WORK_REQUEST, worker.node_id))
        coord = broker.get("coordinator")
        assert coord is not None
        await coord.send(
            worker.node_id,
            msg(
                TASK_OFFER,
                "coordinator",
                task_id="t-1",
                job_id="job-1",
                prompt="misty pine forest",
                seed=1002,
                steps=4,
                width=64,
                height=64,
                lease_seconds=20,
                lease_gen=1,
                upload_url="http://127.0.0.1:8080/jobs/job-1/tasks/t-1/result",
            ),
        )
        await wait_for(lambda: envelopes_of(received, TASK_COMPLETE, worker.node_id))
        types = [e.type for e in received if e.from_id == worker.node_id]
        assert TASK_ACCEPT in types
        assert TASK_STARTED in types
        complete = envelopes_of(received, TASK_COMPLETE, worker.node_id)[0]
        assert complete.payload["task_id"] == "t-1"
        assert complete.payload["lease_gen"] == 1
        assert complete.payload["sha256"]
        assert complete.payload["backend"] == "cuda"
        assert len(complete.payload["sha256"]) == 64
        assert http.calls
        assert http.calls[0]["headers"]["X-Nova-Node-Id"] == worker.node_id
        assert http.calls[0]["headers"]["X-Nova-Lease-Gen"] == "1"
    finally:
        await cluster.stop_all()


@pytest.mark.asyncio
async def test_heartbeats_continue_during_compute(recorder) -> None:
    broker, received, _disconnects = recorder
    cluster = await run_simulated_workers(
        broker,
        1,
        heartbeat_s=0.04,
        progress_s=0.5,
        pull_s=0.05,
        time_scale=0.2,  # 4090 estimate 1.2s * 0.2 = 0.24s
        http_client=FakeAcceptedHttp(),
    )
    try:
        worker = cluster.workers[0]
        await wait_for(lambda: envelopes_of(received, WORK_REQUEST, worker.node_id))
        coord = broker.get("coordinator")
        assert coord is not None
        before = len(envelopes_of(received, HEARTBEAT, worker.node_id))
        await coord.send(
            worker.node_id,
            msg(
                TASK_OFFER,
                "coordinator",
                task_id="t-hb",
                job_id="job-1",
                prompt="clockwork planetarium",
                seed=1,
                width=32,
                height=32,
                lease_gen=1,
                upload_url="http://127.0.0.1:8080/jobs/job-1/tasks/t-hb/result",
            ),
        )
        await wait_for(lambda: envelopes_of(received, TASK_COMPLETE, worker.node_id), timeout=2.0)
        after = len(envelopes_of(received, HEARTBEAT, worker.node_id))
        assert after > before
    finally:
        await cluster.stop_all()


@pytest.mark.asyncio
async def test_kill_worker_no_goodbye_but_disconnect(recorder) -> None:
    broker, received, disconnects = recorder
    cluster = await run_simulated_workers(
        broker, 2, heartbeat_s=0.05, pull_s=0.05, time_scale=0.0
    )
    try:
        victim = cluster.workers[1]
        node_id = victim.node_id
        await wait_for(lambda: envelopes_of(received, HELLO, node_id))
        await cluster.kill_worker(node_id)
        await wait_for(lambda: node_id in disconnects)
        assert victim.killed is True
        assert victim.alive is False
        assert envelopes_of(received, NODE_GOODBYE, node_id) == []
        # module-level helper still finds the other worker
        remaining = cluster.workers[0].node_id
        await kill_worker(remaining)
        await wait_for(lambda: remaining in disconnects)
        assert envelopes_of(received, NODE_GOODBYE, remaining) == []
    finally:
        await cluster.stop_all(graceful=False)


@pytest.mark.asyncio
async def test_stop_graceful_sends_goodbye(recorder) -> None:
    broker, received, disconnects = recorder
    cluster = await run_simulated_workers(
        broker, 1, heartbeat_s=0.05, pull_s=0.05, time_scale=0.0
    )
    node_id = cluster.workers[0].node_id
    await wait_for(lambda: envelopes_of(received, HELLO, node_id))
    await stop_graceful(node_id)
    await wait_for(lambda: envelopes_of(received, NODE_GOODBYE, node_id))
    await wait_for(lambda: node_id in disconnects)
    goodbye = envelopes_of(received, NODE_GOODBYE, node_id)[0]
    assert goodbye.payload.get("node_id") == node_id
    assert cluster.workers[0].killed is False


@pytest.mark.asyncio
async def test_kill_stops_heartbeat_loop(recorder) -> None:
    broker, received, disconnects = recorder
    cluster = await run_simulated_workers(
        broker, 1, heartbeat_s=0.03, pull_s=0.05, time_scale=0.0
    )
    try:
        node_id = cluster.workers[0].node_id
        await wait_for(lambda: len(envelopes_of(received, HEARTBEAT, node_id)) >= 1)
        await cluster.kill_worker(node_id)
        await wait_for(lambda: node_id in disconnects)
        await asyncio.sleep(0.1)
        n = len(envelopes_of(received, HEARTBEAT, node_id))
        await asyncio.sleep(0.12)
        assert len(envelopes_of(received, HEARTBEAT, node_id)) == n
    finally:
        await cluster.stop_all(graceful=False)
