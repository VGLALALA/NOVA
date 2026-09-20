"""Coordinator message loop — glue between transport and scheduler."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from nova.clock import Clock
from nova.config import Settings
from nova.events import EventBus
from nova.models import Job, NodeIdentity, NodeManifest, Task
from nova.network.transport import ControlTransport
from nova.protocol import (
    HEARTBEAT,
    HELLO,
    JOB_ANNOUNCE,
    NODE_DISCONNECTED,
    NODE_PING,
    NODE_GOODBYE,
    NODE_MANIFEST,
    NODE_OFFLINE,
    NODE_UPDATE,
    RESULT_ACK,
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

log = logging.getLogger("nova.coordinator")

TICK_INTERVAL_S = 1.0


def _task_ids(items: Any) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("task_id"):
            out.append(str(item["task_id"]))
        else:
            tid = getattr(item, "task_id", None)
            if tid:
                out.append(str(tid))
    return out


class Coordinator:
    """Transport-agnostic control-plane loop. Scheduler owns leases; we route."""

    def __init__(
        self,
        store: Any,
        scheduler: Any,
        transport: ControlTransport,
        bus: EventBus,
        settings: Settings,
        clock: Clock,
        identity: NodeIdentity,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.transport = transport
        self.bus = bus
        self.settings = settings
        self.clock = clock
        self.identity = identity
        self.node_id = identity.node_id

        self._peer_to_node: dict[str, str] = {}
        self._node_to_peer: dict[str, str] = {}
        self._protocol_versions: dict[str, int] = {}
        self._running = False
        self._tick_task: asyncio.Task[None] | None = None

        self.transport.on_message(self.handle_message)
        self.transport.on_peer_disconnect(self.on_peer_disconnect)
        self.transport.on_peer_connect(self.on_peer_connect)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.transport.start()
        self._tick_task = asyncio.create_task(self._tick_loop(), name="nova-coordinator-tick")

    async def stop(self) -> None:
        self._running = False
        task = self._tick_task
        self._tick_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.transport.stop()

    def on_peer_connect(self, peer_id: str) -> None:
        self.bus.emit("PEER_CONNECTED", peer_id=peer_id)

    def _parse_tcp_target(self, raw: str, port: int | None = None) -> tuple[str, int]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("host required")
        if port is not None:
            if text.count(":") == 1 and "/" not in text:
                host, _, maybe = text.partition(":")
                if maybe.isdigit():
                    return host.strip(), int(maybe)
            return text.split(":")[0].strip(), int(port)
        if "://" in text:
            from urllib.parse import urlparse

            parsed = urlparse(text if "://" in text else f"tcp://{text}")
            host = parsed.hostname or text
            p = parsed.port
            if p is None:
                raise ValueError("port required")
            return host, int(p)
        if text.count(":") == 1:
            host, _, p = text.partition(":")
            return host.strip(), int(p)
        raise ValueError("expected host:port")

    async def connect_tcp(self, host: str, port: int | None = None) -> dict[str, Any]:
        """Dial a forwarded worker control socket (ngrok tcp / ssh -R)."""
        h, p = self._parse_tcp_target(host, port)
        add = getattr(self.transport, "add_connect", None)
        if not callable(add):
            raise RuntimeError("control plane cannot dial TCP")
        add(h, p)
        self.bus.emit("NODE_TCP_ADDED", host=h, port=p)
        return {"host": h, "port": p, "status": "dialing"}

    async def drop_node(self, node_id: str) -> bool:
        peer = self._node_to_peer.get(node_id) or node_id
        dropped = False
        fn = getattr(self.transport, "drop_peer", None)
        if callable(fn):
            dropped = bool(await fn(peer))
        self._handle_disconnect(peer, node_id)
        deleter = getattr(self.store, "delete_node", None)
        if callable(deleter):
            deleter(node_id)
        self.bus.emit("NODE_REMOVED", node_id=node_id)
        return dropped

    def list_links(self) -> list[dict[str, Any]]:
        addrs = getattr(self.transport, "connect_addrs", None)
        out: list[dict[str, Any]] = []
        if isinstance(addrs, list):
            for item in addrs:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    out.append({"host": item[0], "port": int(item[1])})
        for child in getattr(self.transport, "_transports", []) or []:
            for item in getattr(child, "connect_addrs", []) or []:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    rec = {"host": item[0], "port": int(item[1])}
                    if rec not in out:
                        out.append(rec)
        return out

    def on_peer_disconnect(self, peer_id: str) -> None:
        """Immediate requeue. Do not wait for the 30s heartbeat offline path."""
        node_id = self._peer_to_node.get(peer_id) or peer_id
        self._handle_disconnect(peer_id, node_id)

    def _handle_disconnect(self, peer_id: str | None, node_id: str) -> None:
        tasks = self.scheduler.on_disconnect(node_id) or []
        node = self.store.get_node(node_id)
        if node is not None and node.status != "offline":
            node.status = "offline"
            self.store.put_node(node)
        ids = _task_ids(tasks)
        self.bus.emit(NODE_DISCONNECTED, node_id=node_id, tasks=ids)
        self.bus.emit(NODE_OFFLINE, node_id=node_id, tasks=ids)
        if peer_id:
            self._peer_to_node.pop(peer_id, None)
        mapped = self._node_to_peer.get(node_id)
        if mapped:
            self._peer_to_node.pop(mapped, None)
        self._node_to_peer.pop(node_id, None)

    def _map_peer(self, peer_id: str, node_id: str) -> None:
        prev = self._node_to_peer.get(node_id)
        if prev and prev != peer_id:
            self._peer_to_node.pop(prev, None)
        self._peer_to_node[peer_id] = node_id
        self._node_to_peer[node_id] = peer_id

    def _resolve_node_id(self, peer_id: str, env: Envelope) -> str:
        return str(env.payload.get("node_id") or self._peer_to_node.get(peer_id) or env.from_id)

    async def handle_message(self, peer_id: str, env: Envelope) -> None:
        if env.from_id == self.node_id:
            return
        node_id = self._resolve_node_id(peer_id, env)
        if env.type != HELLO:
            self._map_peer(peer_id, node_id)

        handler = {
            HELLO: self._on_hello,
            NODE_MANIFEST: self._on_node_manifest,
            NODE_UPDATE: self._on_node_update,
            HEARTBEAT: self._on_heartbeat,
            WORK_REQUEST: self._on_work_request,
            TASK_ACCEPT: self._on_task_accept,
            TASK_REJECT: self._on_task_reject,
            TASK_STARTED: self._on_task_progress,
            TASK_PROGRESS: self._on_task_progress,
            TASK_COMPLETE: self._on_task_complete,
            TASK_FAILED: self._on_task_failed,
            NODE_GOODBYE: self._on_goodbye,
        }.get(env.type)
        if handler is None:
            log.debug("ignoring %s from %s", env.type, node_id)
            return
        result = handler(peer_id, node_id, env)
        if asyncio.iscoroutine(result):
            await result

    def _on_hello(self, peer_id: str, node_id: str, env: Envelope) -> None:
        node_id = str(env.payload.get("node_id") or env.from_id)
        version = int(env.payload.get("protocol_version") or env.protocol_version)
        self._map_peer(peer_id, node_id)
        self._protocol_versions[node_id] = version
        self.bus.emit("HELLO", node_id=node_id, peer_id=peer_id, protocol_version=version)

    def _payload_node_data(self, env: Envelope, node_id: str) -> dict[str, Any]:
        raw: dict[str, Any] = dict(env.payload)
        for key in ("manifest", "node"):
            inner = raw.get(key)
            if isinstance(inner, dict):
                raw = dict(inner)
                break
        raw.setdefault("node_id", node_id)
        return raw

    def _on_node_manifest(self, peer_id: str, node_id: str, env: Envelope) -> None:
        data = self._payload_node_data(env, node_id)
        try:
            node = NodeManifest.model_validate(data)
        except Exception:
            log.warning("invalid NODE_MANIFEST from %s", node_id, exc_info=True)
            return
        prev = self.store.get_node(node.node_id)
        node.status = "online"
        self.store.put_node(node)
        touch = getattr(self.store, "touch_node", None)
        if callable(touch):
            touch(node.node_id, self.clock.now())
        self._map_peer(peer_id, node.node_id)
        if prev is not None and prev.status in ("offline", "suspect"):
            self.bus.emit("NODE_RECOVERED", node_id=node.node_id)
        self.bus.emit(
            NODE_MANIFEST,
            node_id=node.node_id,
            hostname=node.hostname,
            status=node.status,
            backends=[d.backend for d in node.devices],
        )

    def _on_node_update(self, peer_id: str, node_id: str, env: Envelope) -> None:
        data = self._payload_node_data(env, node_id)
        existing = self.store.get_node(node_id)
        if existing is not None:
            merged = existing.model_dump()
            merged.update({k: v for k, v in data.items() if v is not None})
            data = merged
        try:
            node = NodeManifest.model_validate(data)
        except Exception:
            log.warning("invalid NODE_UPDATE from %s", node_id, exc_info=True)
            return
        if existing is not None and existing.status == "online":
            node.status = "online"
        self.store.put_node(node)
        self._map_peer(peer_id, node.node_id)
        self.bus.emit(
            NODE_UPDATE,
            node_id=node.node_id,
            hostname=node.hostname,
            status=node.status,
            backends=[d.backend for d in node.devices],
        )

    def _on_heartbeat(self, peer_id: str, node_id: str, env: Envelope) -> None:
        self._map_peer(peer_id, node_id)
        self.scheduler.on_heartbeat(node_id)

    async def ping_workers(self, timeout_s: float = 1.5) -> list[str]:
        """Broadcast NODE_PING; mark silent nodes offline. Returns live node ids."""
        from nova.clock import now_utc

        nodes = list(self.store.list_nodes())
        before = {
            n.node_id: self.store.get_last_seen(n.node_id) if hasattr(self.store, "get_last_seen") else None
            for n in nodes
        }
        pinged_at = now_utc()
        try:
            await self.transport.broadcast(msg(NODE_PING, self.node_id, ts=pinged_at.isoformat()))
        except Exception:
            log.exception("NODE_PING broadcast failed")
        await asyncio.sleep(max(timeout_s, 0.2))
        live: list[str] = []
        for node in self.store.list_nodes():
            seen = self.store.get_last_seen(node.node_id) if hasattr(self.store, "get_last_seen") else None
            prev = before.get(node.node_id)
            if seen is not None and (prev is None or seen > prev or seen >= pinged_at):
                if node.status != "online":
                    node.status = "online"
                    self.store.put_node(node)
                live.append(node.node_id)
                continue
            if node.status != "offline":
                self.scheduler.on_disconnect(node.node_id)
        return live

    async def _on_work_request(self, peer_id: str, node_id: str, env: Envelope) -> None:
        node = self.store.get_node(node_id)
        if node is None or node.status != "online":
            return
        payload = env.payload
        slots = int(payload.get("available_slots") or 1)
        device_ids = list(payload.get("device_ids") or [])
        if slots <= 0:
            return
        leased = self.scheduler.on_work_request(node_id, available_slots=slots, device_ids=device_ids)
        if leased is None:
            return
        seconds = int(self.scheduler.lease_seconds_for(node, leased))
        await self._send_task_offer(peer_id, node_id, leased, seconds)

    async def _send_task_offer(self, peer_id: str, node_id: str, task: Task, lease_seconds: int) -> None:
        base = self.settings.public_http_url().rstrip("/")
        job_id = task.job_id
        task_id = task.task_id
        env = msg(
            TASK_OFFER,
            self.node_id,
            task_id=task_id,
            job_id=job_id,
            kernel_id=task.kernel_id,
            shard_index=task.shard_index,
            prompt=task.prompt,
            seed=task.seed,
            steps=task.steps,
            width=task.width,
            height=task.height,
            min_memory_mb=task.min_memory_mb,
            allowed_backends=list(task.allowed_backends),
            fetch_url=f"{base}/jobs/{job_id}/tasks/{task_id}/input.json",
            upload_url=f"{base}/jobs/{job_id}/tasks/{task_id}/result",
            lease_seconds=lease_seconds,
            lease_gen=task.lease_gen,
        )
        await self.transport.send(peer_id, env)
        self.bus.emit(
            TASK_OFFER,
            node_id=node_id,
            task_id=task_id,
            job_id=job_id,
            shard_index=task.shard_index,
            lease_gen=task.lease_gen,
        )

    def _on_task_accept(self, peer_id: str, node_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        if not task_id:
            return
        task = self.store.get_task(task_id)
        if task is None:
            return
        lease_gen = payload.get("lease_gen")
        if task.assigned_node != node_id:
            return
        if task.state not in ("LEASED", "RUNNING"):
            return
        if lease_gen is not None and int(lease_gen) != task.lease_gen:
            return
        if task.state == "LEASED":
            task.state = "RUNNING"
            if task.started_at is None:
                task.started_at = self.clock.now()
            self.store.put_task(task)
        self.bus.emit("TASK_ACCEPT", node_id=node_id, task_id=task.task_id, job_id=task.job_id)

    def _on_task_reject(self, peer_id: str, node_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        if not task_id:
            return
        task = self.store.get_task(task_id)
        if task is None:
            return
        if task.assigned_node not in (None, node_id):
            return
        self.scheduler.requeue_or_fail(task, reason=str(payload.get("reason") or "rejected"))
        self.bus.emit("TASK_REJECT", node_id=node_id, task_id=task_id, job_id=task.job_id)

    def _on_task_progress(self, peer_id: str, node_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        if not task_id:
            return
        lease_gen = payload.get("lease_gen")
        if lease_gen is None:
            return
        self.scheduler.on_progress(node_id, task_id, int(lease_gen))

    async def _on_task_complete(self, peer_id: str, node_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        lease_gen = payload.get("lease_gen")
        supplied_sha = payload.get("sha256") or payload.get("result_sha256")
        accepted = False
        status = "ignored"
        reason: str | None = None

        task = self.store.get_task(task_id) if task_id else None
        if not task_id:
            reason = "missing_task_id"
        elif task is None:
            reason = "task_not_found"
        elif task.state != "COMPLETED":
            reason = "result_not_committed"
        elif task.assigned_node != node_id:
            reason = "wrong_node"
        elif lease_gen is None:
            reason = "missing_lease_gen"
        else:
            try:
                current_lease = int(lease_gen)
            except (TypeError, ValueError):
                reason = "invalid_lease_gen"
            else:
                if current_lease != task.lease_gen:
                    reason = "stale_lease_gen"
                elif supplied_sha and (
                    not task.result_sha256
                    or str(supplied_sha).lower().removeprefix("sha256:")
                    != task.result_sha256.lower().removeprefix("sha256:")
                ):
                    reason = "sha256_mismatch"
                elif not (task.result_path or task.result_sha256):
                    reason = "result_not_committed"
                else:
                    accepted = True
                    status = "already_accepted"

        ack = msg(
            RESULT_ACK,
            self.node_id,
            task_id=task_id,
            job_id=task.job_id if task is not None else payload.get("job_id"),
            lease_gen=lease_gen,
            accepted=accepted,
            status=status,
            reason=reason,
        )
        await self.transport.send(peer_id, ack)

    def _on_task_failed(self, peer_id: str, node_id: str, env: Envelope) -> None:
        payload = env.payload
        task_id = payload.get("task_id")
        if not task_id:
            return
        lease_gen = payload.get("lease_gen")
        error = str(payload.get("error") or payload.get("reason") or "TASK_FAILED")
        self.scheduler.on_task_failed(
            node_id,
            task_id,
            lease_gen=int(lease_gen) if lease_gen is not None else -1,
            reason=error,
            error=error,
        )

    def _on_goodbye(self, peer_id: str, node_id: str, env: Envelope) -> None:
        self._handle_disconnect(peer_id, node_id)

    def _tick(self) -> None:
        self.scheduler.expire_leases()
        self.scheduler.check_node_liveness(
            suspect_s=self.settings.suspect_s,
            offline_s=self.settings.offline_s,
        )

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("coordinator tick failed")
            try:
                await asyncio.sleep(TICK_INTERVAL_S)
            except asyncio.CancelledError:
                raise

    async def submit_job(self, yaml_path_or_job: str | Path | Job, job_id: str | None = None) -> Job:
        from nova.jobs import load_gallery, split_job

        if isinstance(yaml_path_or_job, Job):
            job = yaml_path_or_job
        else:
            job = load_gallery(yaml_path_or_job, job_id=job_id, clock=self.clock)
        tasks = split_job(job, self.clock)
        job.state = "RUNNING"
        self.store.put_job(job)
        for task in tasks:
            self.store.put_task(task)
        self.bus.emit(
            JOB_ANNOUNCE,
            job_id=job.job_id,
            name=job.name,
            task_count=len(tasks),
            state=job.state,
        )
        req = job.requirements
        await self.transport.broadcast(
            msg(
                JOB_ANNOUNCE,
                self.node_id,
                job_id=job.job_id,
                name=job.name,
                job_type=job.job_type,
                kernel_id=job.kernel_id,
                task_count=len(tasks),
                coordinator_url=self.settings.public_http_url(),
                http_url=self.settings.public_http_url(),
                width=req.width,
                height=req.height,
                steps=req.steps,
                model_id=req.model_id,
            )
        )
        return job

    async def cancel_job(self, job_id: str) -> None:
        self.scheduler.cancel_job(job_id)
        job = self.store.get_job(job_id)
        if job is not None:
            job.state = "CANCELLED"
            self.store.put_job(job)
        self.bus.emit("JOB_CANCELLED", job_id=job_id)
