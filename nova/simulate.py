"""One-machine rehearsal: fake CUDA / ROCm / Metal workers.

CLI later::

    from nova.simulate import run_simulated_workers, kill_worker, stop_graceful

    cluster = await run_simulated_workers(broker, n)
    await kill_worker(node_id)          # drop socket, no NODE_GOODBYE
    await stop_graceful(node_id)        # NODE_GOODBYE then disconnect
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nova.models import KERNEL_SD_T2I, Backend, Device, NodeManifest, Vendor
from nova.network.transport import ControlTransport
from nova.protocol import (
    HELLO,
    HEARTBEAT,
    NODE_GOODBYE,
    NODE_MANIFEST,
    PROTOCOL_VERSION,
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

log = logging.getLogger("nova.simulate")

BACKEND_CYCLE: tuple[Backend, ...] = ("cuda", "rocm", "metal")

# score = 1000 / latency_ms (higher is faster).
# 4090 ~ 1000/1200 ≈ 0.83; 7900 XTX ~ 0.35; M3 Max ~ 1000/6667 ≈ 0.15.
_BACKEND_SPECS: dict[str, dict[str, Any]] = {
    "cuda": {
        "vendor": "nvidia",
        "model": "RTX 4090",
        "device_id": "cuda:0",
        "memory_total_mb": 24_576,
        "memory_free_mb": 22_528,
        "hostname": "sim-rtx4090",
        "os": "Linux",
        "architecture": "x86_64",
        "latency_ms": 1200.0,
    },
    "rocm": {
        "vendor": "amd",
        "model": "RX 7900 XTX",
        "device_id": "cuda:0",
        "memory_total_mb": 24_576,
        "memory_free_mb": 22_528,
        "hostname": "sim-rx7900xtx",
        "os": "Linux",
        "architecture": "x86_64",
        "latency_ms": 2857.0,
    },
    "metal": {
        "vendor": "apple",
        "model": "M3 Max",
        "device_id": "mps:0",
        "memory_total_mb": 18_432,  # unified RAM, 50% haircut of 36 GB
        "memory_free_mb": 12_288,
        "hostname": "sim-m3max",
        "os": "Darwin",
        "architecture": "arm64",
        "latency_ms": 6667.0,
    },
}

_VENDOR_RGB: dict[str, tuple[int, int, int]] = {
    "cuda": (34, 197, 94),
    "rocm": (249, 115, 22),
    "metal": (96, 165, 250),
    "cpu": (148, 163, 184),
}


@dataclass(frozen=True)
class WorkerProfile:
    """Template for one fake accelerator."""

    backend: Backend
    vendor: Vendor
    model: str
    device_id: str
    memory_total_mb: int
    memory_free_mb: int
    hostname: str
    os: str
    architecture: str
    latency_ms: float
    index: int = 0

    @property
    def score(self) -> float:
        return 1000.0 / self.latency_ms

    @property
    def estimate_s(self) -> float:
        return self.latency_ms / 1000.0

    @property
    def node_id(self) -> str:
        return f"nova-sim-{self.backend}-{self.index}"

    def manifest(self) -> NodeManifest:
        return NodeManifest(
            node_id=self.node_id,
            hostname=self.hostname if self.index < 3 else f"{self.hostname}-{self.index}",
            os=self.os,
            architecture=self.architecture,
            devices=[
                Device(
                    device_id=self.device_id,
                    backend=self.backend,
                    vendor=self.vendor,
                    model=self.model,
                    memory_total_mb=self.memory_total_mb,
                    memory_free_mb=self.memory_free_mb,
                    load=0.0,
                )
            ],
            supported_kernels=[KERNEL_SD_T2I],
            benchmark_scores={KERNEL_SD_T2I: self.score},
            warmup_ms=1000.0 / self.score if self.score else None,
            fp16_tflops={"cuda": 82.6, "rocm": 61.4, "metal": 10.6}.get(self.backend),
            max_concurrency=1,
            current_slots_used=0,
            status="online",
        )


def profile_for(backend: str, index: int = 0) -> WorkerProfile:
    spec = _BACKEND_SPECS.get(backend)
    if spec is None:
        raise ValueError(f"unknown simulated backend {backend!r}; expected {tuple(_BACKEND_SPECS)}")
    return WorkerProfile(backend=backend, index=index, **spec)  # type: ignore[arg-type]


def build_worker_profiles(
    n: int,
    backends: Sequence[str] | None = None,
) -> list[WorkerProfile]:
    """N fake workers, backends cycling cuda → rocm → metal (or `backends`)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    cycle: Sequence[str] = tuple(backends) if backends else BACKEND_CYCLE
    if not cycle:
        raise ValueError("backends must be non-empty")
    return [profile_for(cycle[i % len(cycle)], index=i) for i in range(n)]


def build_manifests(n: int, backends: Sequence[str] | None = None) -> list[NodeManifest]:
    return [p.manifest() for p in build_worker_profiles(n, backends)]


class InProcessBroker:
    """Shared in-process bus. Coordinator and simulated workers attach as peers."""

    def __init__(self) -> None:
        self._transports: dict[str, InProcessTransport] = {}
        self.coordinator_id: str | None = None

    def attach(self, node_id: str, *, coordinator: bool = False) -> InProcessTransport:
        existing = self._transports.get(node_id)
        if existing is not None:
            if coordinator:
                self.coordinator_id = node_id
            return existing
        transport = InProcessTransport(node_id, self)
        self._transports[node_id] = transport
        if coordinator:
            self.coordinator_id = node_id
        return transport

    def get(self, node_id: str) -> InProcessTransport | None:
        return self._transports.get(node_id)

    @property
    def peer_ids(self) -> list[str]:
        return list(self._transports)

    async def drop(self, node_id: str) -> None:
        self._transports.pop(node_id, None)
        for other in list(self._transports.values()):
            await other._fire_disconnect(node_id)


class InProcessTransport(ControlTransport):
    """ControlTransport over an InProcessBroker. No sockets, no GPU."""

    def __init__(self, node_id: str, broker: InProcessBroker) -> None:
        super().__init__()
        self.node_id = node_id
        self.broker = broker
        self._started = False

    async def start(self) -> None:
        self.broker._transports[self.node_id] = self
        self._started = True
        for other_id, other in list(self.broker._transports.items()):
            if other_id == self.node_id:
                continue
            await other._fire_connect(self.node_id)
            await self._fire_connect(other_id)

    async def stop(self) -> None:
        if not self._started and self.node_id not in self.broker._transports:
            return
        self._started = False
        await self.broker.drop(self.node_id)

    async def send(self, peer_id: str, env: Envelope) -> None:
        dest = self.broker._transports.get(peer_id)
        if dest is None:
            return
        await dest._fire_message(self.node_id, env)

    async def broadcast(self, env: Envelope) -> None:
        for peer_id, dest in list(self.broker._transports.items()):
            if peer_id == self.node_id:
                continue
            await dest._fire_message(self.node_id, env)


def _resolve_broker(
    coordinator_or_broker: Any,
    coordinator_id: str | None = None,
) -> tuple[InProcessBroker, str | None]:
    if isinstance(coordinator_or_broker, InProcessBroker):
        cid = coordinator_id or coordinator_or_broker.coordinator_id
        return coordinator_or_broker, cid
    if isinstance(coordinator_or_broker, InProcessTransport):
        broker = coordinator_or_broker.broker
        cid = coordinator_id or coordinator_or_broker.node_id or broker.coordinator_id
        if broker.coordinator_id is None:
            broker.coordinator_id = coordinator_or_broker.node_id
        return broker, cid
    inner = getattr(coordinator_or_broker, "broker", None)
    if isinstance(inner, InProcessBroker):
        cid = (
            coordinator_id
            or getattr(coordinator_or_broker, "node_id", None)
            or inner.coordinator_id
        )
        return inner, cid
    transport = getattr(coordinator_or_broker, "transport", None)
    if isinstance(transport, InProcessTransport):
        broker = transport.broker
        cid = coordinator_id or transport.node_id or broker.coordinator_id
        if broker.coordinator_id is None:
            broker.coordinator_id = transport.node_id
        return broker, cid
    raise TypeError(
        "run_simulated_workers expects an InProcessBroker, InProcessTransport, "
        "or an object with .broker / .transport pointing at one. "
        "CLI: attach the coordinator to InProcessBroker first, then call "
        "await run_simulated_workers(broker, n)."
    )


def generate_dummy_png(
    width: int,
    height: int,
    *,
    prompt: str = "",
    seed: int = 0,
    backend: str = "cpu",
    model: str = "",
) -> bytes:
    """Valid PNG. Prefers nova.kernels.dummy when present; else Pillow solid color."""
    png = _try_dummy_kernel(width, height, prompt=prompt, seed=seed, backend=backend)
    if png:
        return png
    return _pillow_png(width, height, prompt=prompt, backend=backend, model=model, seed=seed)


def _try_dummy_kernel(
    width: int,
    height: int,
    *,
    prompt: str,
    seed: int,
    backend: str,
) -> bytes | None:
    try:
        from nova.kernels.dummy import generate_png as gen
    except Exception:
        gen = None
    if callable(gen):
        try:
            out = gen(width=width, height=height, prompt=prompt, seed=seed, backend=backend)
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
            png = getattr(out, "png_bytes", None)
            if isinstance(png, (bytes, bytearray)):
                return bytes(png)
        except TypeError:
            try:
                out = gen(width, height, prompt)
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
            except Exception:
                pass
        except Exception:
            log.debug("nova.kernels.dummy.generate_png failed; using Pillow", exc_info=True)
    try:
        from nova.kernels.dummy import DummyKernel
    except Exception:
        return None
    try:
        kernel = DummyKernel()
    except TypeError:
        return None
    execute = getattr(kernel, "execute_sync", None) or getattr(kernel, "render", None)
    if not callable(execute):
        return None
    try:
        out = execute(
            {
                "width": width,
                "height": height,
                "prompt": prompt,
                "seed": seed,
                "backend": backend,
            }
        )
    except Exception:
        return None
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    png = getattr(out, "png_bytes", None)
    if isinstance(png, (bytes, bytearray)):
        return bytes(png)
    return None


def _pillow_png(
    width: int,
    height: int,
    *,
    prompt: str,
    backend: str,
    model: str,
    seed: int,
) -> bytes:
    from PIL import Image, ImageDraw

    w = max(int(width or 512), 1)
    h = max(int(height or 512), 1)
    color = _VENDOR_RGB.get(backend, (80, 80, 80))
    image = Image.new("RGB", (w, h), color)
    draw = ImageDraw.Draw(image)
    lines = [model or backend, backend, f"seed {seed}", (prompt or "")[:80]]
    text = "\n".join(line for line in lines if line)
    draw.text((16, 16), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


async def _put_png(
    upload_url: str,
    png: bytes,
    *,
    lease_gen: int,
    sha256: str,
    backend: str,
    device_id: str,
    execution_ms: int,
    node_id: str = "",
    client: Any | None = None,
) -> bool:
    if not upload_url:
        return False
    headers = {
        "Content-Type": "image/png",
        "X-Nova-Lease-Gen": str(lease_gen),
        "X-Nova-Sha256": sha256,
        "X-Nova-Node-Id": node_id,
        "X-Nova-Backend": backend,
        "X-Nova-Device-Id": device_id,
        "X-Nova-Execution-Ms": str(execution_ms),
    }
    closer = None
    http = client
    if http is None:
        try:
            import httpx
        except ImportError:
            return False
        http = httpx.AsyncClient(timeout=10.0)
        closer = http
    try:
        response = await http.put(upload_url, content=png, headers=headers)
        if not response.is_success:
            return False
        body = response.json()
        return isinstance(body, dict) and body.get("status") == "accepted"
    except Exception:
        log.warning("simulated PUT to %s failed", upload_url, exc_info=True)
        return False
    finally:
        if closer is not None:
            await closer.aclose()


class SimulatedWorker:
    """Fake worker: HELLO, manifest, heartbeat, pull, dummy PNG, complete."""

    def __init__(
        self,
        profile: WorkerProfile | NodeManifest,
        transport: ControlTransport,
        *,
        coordinator_id: str | None = None,
        heartbeat_s: float = 5.0,
        progress_s: float = 2.0,
        pull_s: float = 1.0,
        time_scale: float = 1.0,
        http_client: Any | None = None,
    ) -> None:
        if isinstance(profile, NodeManifest):
            device = profile.devices[0]
            score = float(profile.benchmark_scores.get(KERNEL_SD_T2I, 0.1) or 0.1)
            self.profile = WorkerProfile(
                backend=device.backend,
                vendor=device.vendor,
                model=device.model,
                device_id=device.device_id,
                memory_total_mb=device.memory_total_mb,
                memory_free_mb=device.memory_free_mb,
                hostname=profile.hostname,
                os=profile.os,
                architecture=profile.architecture,
                latency_ms=1000.0 / score,
                index=0,
            )
            self.manifest = profile
        else:
            self.profile = profile
            self.manifest = profile.manifest()
        self.transport = transport
        self.coordinator_id = coordinator_id
        self.heartbeat_s = heartbeat_s
        self.progress_s = progress_s
        self.pull_s = pull_s
        self.time_scale = time_scale
        self._http = http_client
        self._alive = False
        self._killed = False
        self._graceful = False
        self._idle = True
        self._awaiting_offer = False
        self._slots_used = 0
        self._task: asyncio.Task[None] | None = None
        self._work_task: asyncio.Task[None] | None = None
        self._bg: list[asyncio.Task[None]] = []
        self._started = asyncio.Event()
        self._done = asyncio.Event()
        self._wake_pull = asyncio.Event()

    @property
    def node_id(self) -> str:
        return self.manifest.node_id

    @property
    def backend(self) -> str:
        return self.profile.backend

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def alive(self) -> bool:
        return self._alive and not self._killed

    async def start(self) -> None:
        if self._task is not None:
            await self._started.wait()
            return
        self._killed = False
        self._graceful = False
        self._done = asyncio.Event()
        self._started = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name=f"sim-worker-{self.node_id}")
        await self._started.wait()

    async def stop(self) -> None:
        """Graceful leave: NODE_GOODBYE, then drop the in-process peer."""
        if self._killed:
            await self._finish_task()
            return
        if self._graceful and self._task is None:
            return
        self._graceful = True
        if self._started.is_set() and self._alive:
            await self._send(
                msg(NODE_GOODBYE, self.node_id, node_id=self.node_id, reason="graceful")
            )
        self._alive = False
        self._done.set()
        await self._finish_task()

    async def kill(self) -> None:
        """Abrupt death: no NODE_GOODBYE. Coordinator should see disconnect."""
        self._killed = True
        self._alive = False
        self._done.set()
        await self._finish_task()

    async def _finish_task(self) -> None:
        task = self._task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        self._alive = True
        self._idle = True
        self._awaiting_offer = False
        self.transport.on_message(self._on_message)
        try:
            await self.transport.start()
            await self._send(
                msg(
                    HELLO,
                    self.node_id,
                    node_id=self.node_id,
                    protocol_version=PROTOCOL_VERSION,
                )
            )
            await self._send(msg(NODE_MANIFEST, self.node_id, **self.manifest.model_dump(mode="json")))
            self._started.set()
            self._bg = [
                asyncio.create_task(self._heartbeat_loop(), name=f"sim-hb-{self.node_id}"),
                asyncio.create_task(self._pull_loop(), name=f"sim-pull-{self.node_id}"),
            ]
            await self._done.wait()
        finally:
            self._alive = False
            self._started.set()
            for bg in self._bg:
                bg.cancel()
            if self._bg:
                await asyncio.gather(*self._bg, return_exceptions=True)
            self._bg = []
            work = self._work_task
            if work is not None and not work.done():
                work.cancel()
                try:
                    await work
                except (asyncio.CancelledError, Exception):
                    pass
            self._work_task = None
            try:
                await self.transport.stop()
            except Exception:
                log.debug("simulated worker %s transport.stop failed", self.node_id, exc_info=True)

    async def _send(self, env: Envelope, peer_id: str | None = None) -> None:
        target = peer_id or self.coordinator_id
        if not target:
            broker = getattr(self.transport, "broker", None)
            target = getattr(broker, "coordinator_id", None)
        if target:
            await self.transport.send(target, env)
        else:
            await self.transport.broadcast(env)

    async def _heartbeat_loop(self) -> None:
        try:
            while self._alive:
                await self._send(
                    msg(
                        HEARTBEAT,
                        self.node_id,
                        node_id=self.node_id,
                        status="online",
                        current_slots_used=self._slots_used,
                        available_slots=max(0, self.manifest.max_concurrency - self._slots_used),
                    )
                )
                await asyncio.sleep(self.heartbeat_s)
        except asyncio.CancelledError:
            return

    async def _pull_loop(self) -> None:
        try:
            while self._alive:
                if self._idle and not self._awaiting_offer:
                    await self._send_work_request()
                    self._awaiting_offer = True
                self._wake_pull.clear()
                try:
                    await asyncio.wait_for(self._wake_pull.wait(), timeout=self.pull_s)
                except TimeoutError:
                    if self._idle:
                        self._awaiting_offer = False
        except asyncio.CancelledError:
            return

    async def _send_work_request(self) -> None:
        await self._send(
            msg(
                WORK_REQUEST,
                self.node_id,
                node_id=self.node_id,
                available_slots=max(0, self.manifest.max_concurrency - self._slots_used),
                device_ids=[d.device_id for d in self.manifest.devices],
            )
        )

    async def _on_message(self, peer_id: str, env: Envelope) -> None:
        if not self._alive:
            return
        if env.type != TASK_OFFER:
            return
        if self.coordinator_id is None:
            self.coordinator_id = peer_id
        if not self._idle:
            await self._send(
                msg(
                    TASK_REJECT,
                    self.node_id,
                    task_id=env.payload.get("task_id"),
                    job_id=env.payload.get("job_id"),
                    lease_gen=env.payload.get("lease_gen", 0),
                    reason="busy",
                    node_id=self.node_id,
                ),
                peer_id,
            )
            return
        allowed = env.payload.get("allowed_backends")
        if allowed and self.backend not in allowed:
            await self._send(
                msg(
                    TASK_REJECT,
                    self.node_id,
                    task_id=env.payload.get("task_id"),
                    job_id=env.payload.get("job_id"),
                    lease_gen=env.payload.get("lease_gen", 0),
                    reason="incompatible_backend",
                    node_id=self.node_id,
                ),
                peer_id,
            )
            return
        self._idle = False
        self._awaiting_offer = False
        self._slots_used = 1
        self._work_task = asyncio.create_task(
            self._handle_offer(peer_id, env),
            name=f"sim-work-{self.node_id}",
        )

    async def _handle_offer(self, peer_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        job_id = payload.get("job_id")
        lease_gen = payload.get("lease_gen", 0)
        prompt = payload.get("prompt") or ""
        seed = int(payload.get("seed") or 0)
        width = int(payload.get("width") or 512)
        height = int(payload.get("height") or 512)
        upload_url = payload.get("upload_url") or ""
        try:
            await self._send(
                msg(
                    TASK_ACCEPT,
                    self.node_id,
                    task_id=task_id,
                    job_id=job_id,
                    lease_gen=lease_gen,
                    node_id=self.node_id,
                ),
                peer_id,
            )
            await self._send(
                msg(
                    TASK_STARTED,
                    self.node_id,
                    task_id=task_id,
                    job_id=job_id,
                    lease_gen=lease_gen,
                    node_id=self.node_id,
                    backend=self.backend,
                    device_id=self.profile.device_id,
                ),
                peer_id,
            )
            duration = self.profile.estimate_s * self.time_scale
            t0 = asyncio.get_running_loop().time()
            await self._sleep_compute(duration, peer_id, task_id, job_id, lease_gen)
            if not self._alive:
                return
            png = generate_dummy_png(
                width,
                height,
                prompt=prompt,
                seed=seed,
                backend=self.backend,
                model=self.profile.model,
            )
            sha = hashlib.sha256(png).hexdigest()
            execution_ms = int((asyncio.get_running_loop().time() - t0) * 1000)
            uploaded = await _put_png(
                upload_url,
                png,
                lease_gen=int(lease_gen or 0),
                sha256=sha,
                backend=self.backend,
                device_id=self.profile.device_id,
                execution_ms=execution_ms,
                node_id=self.node_id,
                client=self._http,
            )
            if not uploaded:
                raise RuntimeError("result upload was not accepted")
            if not self._alive:
                return
            await self._send(
                msg(
                    TASK_COMPLETE,
                    self.node_id,
                    task_id=task_id,
                    job_id=job_id,
                    lease_gen=lease_gen,
                    result_url=upload_url,
                    sha256=sha,
                    execution_ms=execution_ms,
                    backend=self.backend,
                    device_id=self.profile.device_id,
                    node_id=self.node_id,
                ),
                peer_id,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.exception("simulated worker %s failed task %s", self.node_id, task_id)
            if self._alive:
                await self._send(
                    msg(
                        TASK_FAILED,
                        self.node_id,
                        task_id=task_id,
                        job_id=job_id,
                        lease_gen=lease_gen,
                        error=str(exc),
                        node_id=self.node_id,
                    ),
                    peer_id,
                )
        finally:
            self._slots_used = 0
            self._idle = True
            self._awaiting_offer = False
            self._wake_pull.set()

    async def _sleep_compute(
        self,
        duration: float,
        peer_id: str,
        task_id: Any,
        job_id: Any,
        lease_gen: Any,
    ) -> None:
        if duration <= 0:
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        end = loop.time() + duration
        while self._alive:
            remaining = end - loop.time()
            if remaining <= 0:
                return
            wait = min(remaining, self.progress_s if self.progress_s > 0 else remaining)
            await asyncio.sleep(wait)
            if not self._alive:
                return
            if end - loop.time() > 0:
                await self._send(
                    msg(
                        TASK_PROGRESS,
                        self.node_id,
                        task_id=task_id,
                        job_id=job_id,
                        lease_gen=lease_gen,
                        node_id=self.node_id,
                    ),
                    peer_id,
                )


class SimulatedCluster:
    """Running set of SimulatedWorkers plus kill / graceful stop."""

    def __init__(self, workers: list[SimulatedWorker], broker: InProcessBroker) -> None:
        self.workers = workers
        self.broker = broker
        self._by_id = {w.node_id: w for w in workers}

    def get(self, node_id: str) -> SimulatedWorker:
        try:
            return self._by_id[node_id]
        except KeyError as exc:
            raise KeyError(f"no simulated worker {node_id!r}") from exc

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._by_id

    async def start(self) -> None:
        for worker in self.workers:
            await worker.start()

    async def kill_worker(self, node_id: str) -> None:
        await self.get(node_id).kill()
        self._forget_if_idle()

    async def stop_graceful(self, node_id: str | None = None) -> None:
        if node_id is None:
            await self.stop_all(graceful=True)
            return
        await self.get(node_id).stop()
        self._forget_if_idle()

    def _forget_if_idle(self) -> None:
        if not any(w.alive or w._task is not None for w in self.workers):
            _forget_cluster(self)

    async def stop_all(self, *, graceful: bool = True) -> None:
        for worker in self.workers:
            if worker.killed:
                continue
            if graceful:
                await worker.stop()
            else:
                await worker.kill()
        _forget_cluster(self)


_clusters: list[SimulatedCluster] = []


async def run_simulated_workers(
    coordinator_or_broker: Any,
    n: int,
    *,
    backends: Sequence[str] | None = None,
    heartbeat_s: float = 5.0,
    progress_s: float = 2.0,
    pull_s: float = 1.0,
    time_scale: float = 1.0,
    coordinator_id: str | None = None,
    http_client: Any | None = None,
) -> SimulatedCluster:
    """Start N fake workers talking to the coordinator over an in-process bus.

    This is the function the CLI should call for `--simulate-workers N`.
    """
    broker, cid = _resolve_broker(coordinator_or_broker, coordinator_id)
    profiles = build_worker_profiles(n, backends)
    workers: list[SimulatedWorker] = []
    for profile in profiles:
        transport = broker.attach(profile.node_id)
        workers.append(
            SimulatedWorker(
                profile,
                transport,
                coordinator_id=cid,
                heartbeat_s=heartbeat_s,
                progress_s=progress_s,
                pull_s=pull_s,
                time_scale=time_scale,
                http_client=http_client,
            )
        )
    cluster = SimulatedCluster(workers, broker)
    _clusters.append(cluster)
    try:
        await cluster.start()
    except Exception:
        await cluster.stop_all(graceful=False)
        raise
    return cluster


async def kill_worker(node_id: str) -> None:
    """Stop a simulated worker without NODE_GOODBYE (demo failover)."""
    await _find_cluster(node_id).kill_worker(node_id)


async def stop_graceful(node_id: str | None = None) -> None:
    """Stop with NODE_GOODBYE. `node_id=None` stops the whole cluster."""
    if node_id is None:
        for cluster in list(_clusters):
            await cluster.stop_graceful(None)
        return
    await _find_cluster(node_id).stop_graceful(node_id)


def _find_cluster(node_id: str) -> SimulatedCluster:
    for cluster in reversed(_clusters):
        if node_id in cluster:
            return cluster
    raise RuntimeError(f"no simulated worker cluster holds {node_id!r}")


def _forget_cluster(cluster: SimulatedCluster) -> None:
    try:
        _clusters.remove(cluster)
    except ValueError:
        pass


__all__ = [
    "BACKEND_CYCLE",
    "InProcessBroker",
    "InProcessTransport",
    "SimulatedCluster",
    "SimulatedWorker",
    "WorkerProfile",
    "build_manifests",
    "build_worker_profiles",
    "generate_dummy_png",
    "kill_worker",
    "profile_for",
    "run_simulated_workers",
    "stop_graceful",
]
