"""Worker runtime. Async control loop; blocking inference off-thread."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from nova.clock import Clock
from nova.config import Settings
from nova.hardware import build_manifest, preferred_accelerator, preferred_device, probe_devices
from nova.identity import load_or_create
from nova.kernels import load_kernel
from nova.kernels.base import NovaKernel
from nova.models import Device, KernelResult, NodeIdentity, NodeManifest, Task
from nova.network.transport import ControlTransport

from nova.protocol import (
    CLUSTER_SYNC,
    HELLO,
    HEARTBEAT,
    NODE_GOODBYE,
    NODE_MANIFEST,
    NODE_PING,
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

logger = logging.getLogger("nova.worker")


def _is_private_http_url(url: str) -> bool:
    """True for loopback / RFC1918 HTTP bases that a remote GPU cannot PUT to."""
    host = (urlparse(url).hostname or "").strip().lower().strip("[]")
    if not host:
        return True
    if host in {"localhost", "0.0.0.0", "127.0.0.1", "::1", "::"}:
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        try:
            return 16 <= int(parts[1]) <= 31
        except (IndexError, ValueError):
            return False
    return False


class Worker:
    def __init__(
        self,
        settings: Settings,
        transport: ControlTransport,
        *,
        identity: NodeIdentity | None = None,
        kernel: NovaKernel | None = None,
        devices: list[Device] | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        coordinator_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.clock = clock or Clock()
        self.identity = identity or load_or_create(settings.data_dir)
        self.devices = devices if devices is not None else probe_devices()
        self.kernel = kernel or load_kernel(settings.kernel, settings)
        self._http = http_client
        self._owns_http = http_client is None
        self.coordinator_id = coordinator_id
        self.max_concurrency = 1
        self.slots_used = 0
        self.manifest = build_manifest(
            self.identity,
            self.devices,
            max_concurrency=self.max_concurrency,
            current_slots_used=0,
        )
        self._stop = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._compute_task: asyncio.Task[None] | None = None
        self._current: dict[str, Any] | None = None
        self._http_base: str | None = settings.coordinator_url

        transport.on_message(self.handle_message)
        transport.on_peer_connect(self._on_connect)

    @property
    def node_id(self) -> str:
        return self.identity.node_id

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=60.0,
                headers={"ngrok-skip-browser-warning": "1"},
            )
        await self.transport.start()
        await self._warmup()
        await self._hello_and_manifest()
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._control_loop(), name=f"worker-{self.node_id}")

    async def stop(self) -> None:
        self._stop.set()
        try:
            await self.transport.broadcast(msg(NODE_GOODBYE, self.node_id, node_id=self.node_id))
        except Exception:
            pass
        if self._compute_task is not None:
            self._compute_task.cancel()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        await self.transport.stop()
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def run(self) -> None:
        await self.start()
        await self._stop.wait()

    async def handle_message(self, peer_id: str, env: Envelope) -> None:
        if env.type == TASK_OFFER:
            self.coordinator_id = peer_id
            await self._on_offer(peer_id, env.payload)
        elif env.type == NODE_PING:
            self.coordinator_id = peer_id
            await self.transport.send(
                peer_id,
                msg(HEARTBEAT, self.node_id, node_id=self.node_id, ping=True),
            )
        elif env.type == CLUSTER_SYNC:
            nodes = env.payload.get("nodes") or []
            if isinstance(nodes, list):
                for raw in nodes:
                    if not isinstance(raw, dict):
                        continue
                    host = raw.get("control_host") or raw.get("host")
                    port = raw.get("control_port") or raw.get("port")
                    url = raw.get("http_url")
                    if url:
                        self._remember_http_base(str(url))
                    if host and port:
                        h = str(host).strip().lower().strip("[]")
                        mine = {
                            "127.0.0.1",
                            "localhost",
                            "0.0.0.0",
                            "::1",
                            "::",
                            str(self.settings.advertise_host or "").strip().lower(),
                        }
                        if h in mine or h.startswith("10.") or h.startswith("192.168.") or h.startswith("172."):
                            continue
                        add = getattr(self.transport, "add_connect", None)
                        if callable(add):
                            try:
                                add(str(host), int(port))
                            except Exception:
                                logger.debug("cluster dial failed", extra={"host": host, "port": port})
        elif env.type == "JOB_ANNOUNCE":
            self.coordinator_id = peer_id
            url = env.payload.get("coordinator_url") or env.payload.get("http_url")
            if url:
                self._remember_http_base(str(url))

    async def _on_connect(self, peer_id: str) -> None:
        self.coordinator_id = self.coordinator_id or peer_id
        await self._hello_and_manifest(peer_id)

    def _select_device(self, allowed_backends: list[str] | None = None) -> Device:
        if allowed_backends is not None:
            allowed = list(allowed_backends)
            if any(d.backend in ("cuda", "rocm") for d in self.devices):
                allowed = [b for b in allowed if b in ("cuda", "rocm")] or ["cuda", "rocm"]
            elif any(d.backend == "metal" for d in self.devices):
                allowed = [b for b in allowed if b == "metal"] or ["metal"]
            device = preferred_device(self.devices, allowed)
            if device is None:
                raise RuntimeError(f"no device compatible with {allowed_backends}")
            return device
        if self.settings.dummy:
            return preferred_device(self.devices) or self.devices[0]
        device = preferred_accelerator(self.devices)
        if device is None:
            raise RuntimeError(
                "No CUDA, ROCm, or Metal accelerator found. "
                "Install the matching PyTorch build (Mac MPS, Windows CUDA, or Linux CUDA) "
                "or start with --dummy / NOVA_DUMMY=1."
            )
        return device

    async def _warmup(self) -> None:
        device = self._select_device()
        result = await asyncio.to_thread(self.kernel.warmup, device)
        latency_ms = max(int(result.execution_ms), 1)
        from nova.flops import fp16_tflops

        self.manifest.benchmark_scores[self.kernel.kernel_id] = 1000.0 / latency_ms
        self.manifest.warmup_ms = float(latency_ms)
        peak = None
        try:
            from nova.flops import measure_device_fp16_tflops

            peak = await asyncio.to_thread(measure_device_fp16_tflops, device.device_id)
        except Exception:
            peak = None
        if peak is None:
            peak = fp16_tflops(latency_ms, steps=1, width=512, height=512)
        self.manifest.fp16_tflops = round(float(peak), 3)
        logger.info(
            "warmup %sms on %s score=%.3f fp16_tflops=%.2f",
            latency_ms,
            device.backend,
            1000.0 / latency_ms,
            self.manifest.fp16_tflops,
        )

    async def _hello_and_manifest(self, peer_id: str | None = None) -> None:
        hello = msg(HELLO, self.node_id, protocol_version=PROTOCOL_VERSION, node_id=self.node_id)
        manifest = msg(NODE_MANIFEST, self.node_id, **self.manifest.model_dump(mode="json"))
        if peer_id:
            await self.transport.send(peer_id, hello)
            await self.transport.send(peer_id, manifest)
        else:
            await self.transport.broadcast(hello)
            await self.transport.broadcast(manifest)

    async def _control_loop(self) -> None:
        heartbeat_s = self.settings.heartbeat_s
        progress_s = self.settings.progress_s
        last_beat = 0.0
        last_progress = 0.0
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            now = loop.time()
            if now - last_beat >= heartbeat_s:
                await self.transport.broadcast(msg(HEARTBEAT, self.node_id, node_id=self.node_id))
                last_beat = now
            if self._current is not None and now - last_progress >= progress_s:
                offer = self._current
                await self._send_to_coord(
                    msg(
                        TASK_PROGRESS,
                        self.node_id,
                        task_id=offer["task_id"],
                        job_id=offer.get("job_id"),
                        lease_gen=offer.get("lease_gen"),
                    )
                )
                last_progress = now
            if self.slots_used < self.max_concurrency and self._current is None:
                # Broadcast so a remote job owner can lease us. Pinning to the
                # first HELLO peer (often this box's own coordinator) leaves
                # CUDA asking an empty local store while Metal finishes the job.
                await self.transport.broadcast(
                    msg(
                        WORK_REQUEST,
                        self.node_id,
                        node_id=self.node_id,
                        available_slots=self.max_concurrency - self.slots_used,
                        device_ids=[d.device_id for d in self.devices],
                    )
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

    async def _on_offer(self, peer_id: str, payload: dict[str, Any]) -> None:
        if self.slots_used >= self.max_concurrency or self._current is not None:
            await self.transport.send(
                peer_id,
                msg(TASK_REJECT, self.node_id, task_id=payload.get("task_id"), reason="busy"),
            )
            return
        task = _task_from_offer(payload)
        try:
            device = self._select_device(list(task.allowed_backends))
        except RuntimeError:
            device = None
        if device is None or not self.kernel.compatible(device):
            await self.transport.send(
                peer_id,
                msg(TASK_REJECT, self.node_id, task_id=task.task_id, reason="incompatible"),
            )
            return
        self.slots_used += 1
        self.manifest.current_slots_used = self.slots_used
        self._current = payload
        await self.transport.send(
            peer_id,
            msg(TASK_ACCEPT, self.node_id, task_id=task.task_id, lease_gen=payload.get("lease_gen")),
        )
        await self.transport.send(
            peer_id,
            msg(TASK_STARTED, self.node_id, task_id=task.task_id, lease_gen=payload.get("lease_gen")),
        )
        self._compute_task = asyncio.create_task(self._run_task(peer_id, task, device, payload))

    async def _run_task(self, peer_id: str, task: Task, device: Device, payload: dict[str, Any]) -> None:
        lease_gen = payload.get("lease_gen")
        try:
            result: KernelResult = await asyncio.to_thread(self.kernel.execute_sync, task, device)
            await self._put_result(payload, result, lease_gen)
            await self.transport.send(
                peer_id,
                msg(
                    TASK_COMPLETE,
                    self.node_id,
                    task_id=task.task_id,
                    job_id=task.job_id,
                    result_url=payload["upload_url"],
                    sha256=result.sha256,
                    execution_ms=result.execution_ms,
                    lease_gen=lease_gen,
                    backend=result.backend,
                    device_id=result.device_id,
                ),
            )
        except Exception as exc:
            logger.exception("task %s failed", task.task_id)
            await self.transport.send(
                peer_id,
                msg(
                    TASK_FAILED,
                    self.node_id,
                    task_id=task.task_id,
                    job_id=task.job_id,
                    lease_gen=lease_gen,
                    error=str(exc),
                ),
            )
        finally:
            self.slots_used = max(0, self.slots_used - 1)
            self.manifest.current_slots_used = self.slots_used
            self._current = None
            self._compute_task = None

    async def _put_result(self, payload: dict[str, Any], result: KernelResult, lease_gen: Any) -> bool:
        url = self._resolve_upload_url(str(payload.get("upload_url") or ""))
        if not url:
            raise RuntimeError("result upload requires upload_url")
        if self._http is None:
            raise RuntimeError("result upload requires an initialized HTTP client")
        headers = {
            "X-Nova-Lease-Gen": str(lease_gen if lease_gen is not None else 0),
            "X-Nova-Sha256": result.sha256,
            "X-Nova-Node-Id": self.node_id,
            "X-Nova-Backend": result.backend,
            "X-Nova-Device-Id": result.device_id,
            "X-Nova-Execution-Ms": str(result.execution_ms),
            "Content-Type": "image/png",
        }
        try:
            resp = await self._http.put(url, content=result.png_bytes, headers=headers)
        except Exception as exc:
            raise RuntimeError(f"result upload request failed: {exc}") from exc
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(f"result upload failed with HTTP {resp.status_code}")
        try:
            response_body = resp.json()
        except Exception as exc:
            raise RuntimeError("result upload returned malformed JSON") from exc
        if not isinstance(response_body, dict):
            raise RuntimeError("result upload response must be a JSON object")
        status = response_body.get("status")
        if status != "accepted":
            raise RuntimeError(f"result upload was not accepted: status={status!r}")
        return True

    async def _send_to_coord(self, env: Envelope) -> None:
        if self.coordinator_id:
            await self.transport.send(self.coordinator_id, env)
        else:
            await self.transport.broadcast(env)

    def _remember_http_base(self, url: str) -> None:
        text = (url or "").strip().rstrip("/")
        if not text:
            return
        if not self._http_base or _is_private_http_url(self._http_base):
            if not _is_private_http_url(text) or not self._http_base:
                self._http_base = text

    def _resolve_upload_url(self, url: str) -> str:
        text = (url or "").strip()
        if not text:
            return text
        if not _is_private_http_url(text) or not self._http_base:
            return text
        parsed = urlparse(text)
        path = parsed.path or ""
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return f"{str(self._http_base).rstrip('/')}{path}"


def _task_from_offer(payload: dict[str, Any]) -> Task:
    from nova.clock import now_utc

    return Task(
        task_id=str(payload["task_id"]),
        job_id=str(payload.get("job_id") or ""),
        shard_index=int(payload.get("shard_index") or 0),
        kernel_id=str(payload.get("kernel_id") or "sd.t2i.v1"),
        prompt=str(payload.get("prompt") or ""),
        seed=int(payload.get("seed") or 0),
        steps=int(payload.get("steps") or 4),
        width=int(payload.get("width") or 512),
        height=int(payload.get("height") or 512),
        min_memory_mb=int(payload.get("min_memory_mb") or 0),
        allowed_backends=list(payload.get("allowed_backends") or ["cuda", "rocm", "metal", "cpu"]),
        created_at=now_utc(),
        lease_gen=int(payload.get("lease_gen") or 0),
    )
