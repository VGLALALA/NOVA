"""Pear / Hyperswarm sidecar transport. Optional. TCP is the emergency path."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

from nova.network.transport import ControlTransport
from nova.protocol import HELLO, PROTOCOL_VERSION, Envelope, msg

logger = logging.getLogger("nova.network.pear")


class PearUnavailable(RuntimeError):
    """Sidecar cannot run. Use TcpTransport with NOVA_PEERS / NOVA_CONTROL_PORT."""


def _bridge_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pear" / "bridge.js"


def _hyperswarm_dir(bridge: Path) -> Path:
    return bridge.parent / "node_modules" / "hyperswarm"


class PearTransport(ControlTransport):
    """Spawns `node pear/bridge.js`. JSON lines on stdin/stdout."""

    def __init__(
        self,
        *,
        node_id: str,
        swarm_topic: str = "nova-htn-2026",
        node_bin: str = "node",
        bridge_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.node_id = node_id
        self.swarm_topic = swarm_topic
        self._node_bin = node_bin
        self._bridge_path = Path(bridge_path) if bridge_path else _bridge_path()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._peers: set[str] = set()
        self._identified: set[str] = set()
        self._ready = asyncio.Event()
        self._stopping = False

    @classmethod
    def available(cls, node_bin: str = "node", bridge_path: Path | None = None) -> bool:
        if not shutil.which(node_bin):
            return False
        bridge = Path(bridge_path) if bridge_path else _bridge_path()
        if not bridge.is_file():
            return False
        return _hyperswarm_dir(bridge).exists()

    async def start(self) -> None:
        if self._proc is not None:
            return
        self._stopping = False
        self._check_sidecar()
        env = os.environ.copy()
        env["NOVA_SWARM_TOPIC"] = self.swarm_topic
        env["NOVA_NODE_ID"] = self.node_id
        self._proc = await asyncio.create_subprocess_exec(
            self._node_bin,
            str(self._bridge_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self._bridge_path.parent),
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=5.0)
        except TimeoutError as exc:
            await self.stop()
            raise PearUnavailable(
                "pear sidecar did not become ready; use TcpTransport / NOVA_PEERS"
            ) from exc
        if self._proc is None or self._proc.returncode is not None:
            code = None if self._proc is None else self._proc.returncode
            await self.stop()
            raise PearUnavailable(
                f"pear sidecar exited {code}; use TcpTransport / NOVA_PEERS"
            )

    async def stop(self) -> None:
        self._stopping = True
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                await self._stdin_write({"op": "stop"})
            except Exception:
                pass
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._reader_task = None
        self._stderr_task = None
        leftover = list(self._peers | self._identified)
        self._peers.clear()
        self._identified.clear()
        for peer_id in leftover:
            try:
                await self._fire_disconnect(peer_id)
            except Exception:
                logger.exception("pear disconnect handler failed")

    async def send(self, peer_id: str, env: Envelope) -> None:
        await self._stdin_write(
            {"op": "send", "peer_id": peer_id, "envelope": env.model_dump(mode="json")}
        )

    async def broadcast(self, env: Envelope) -> None:
        await self._stdin_write({"op": "broadcast", "envelope": env.model_dump(mode="json")})

    def _check_sidecar(self) -> None:
        if not shutil.which(self._node_bin):
            raise PearUnavailable(
                "node is not on PATH; use TcpTransport with NOVA_PEERS / NOVA_CONTROL_PORT"
            )
        if not self._bridge_path.is_file():
            raise PearUnavailable(
                f"missing {self._bridge_path}; use TcpTransport with NOVA_PEERS"
            )
        if not _hyperswarm_dir(self._bridge_path).exists():
            raise PearUnavailable(
                "hyperswarm not installed (cd pear && npm install); "
                "use TcpTransport with NOVA_PEERS / NOVA_CONTROL_PORT"
            )

    async def _stdin_write(self, obj: dict[str, object]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        async with self._write_lock:
            proc.stdin.write(payload)
            await proc.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    ev = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                await self._handle_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pear stdout reader failed")
        finally:
            self._ready.set()

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                logger.debug("pear: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("pear stderr closed")

    async def _handle_event(self, ev: dict[str, object]) -> None:
        op = ev.get("op")
        if op == "ready":
            self._ready.set()
            return
        if op == "error":
            logger.warning("pear sidecar error: %s", ev.get("error"))
            self._ready.set()
            return
        if op == "peer_connect":
            peer_id = str(ev.get("peer_id", ""))
            if not peer_id:
                return
            self._peers.add(peer_id)
            try:
                await self.send(
                    peer_id,
                    msg(
                        HELLO,
                        self.node_id,
                        protocol_version=PROTOCOL_VERSION,
                        node_id=self.node_id,
                    ),
                )
            except Exception:
                logger.exception("pear HELLO send failed")
            return
        if op == "peer_disconnect":
            peer_id = str(ev.get("peer_id", ""))
            self._peers.discard(peer_id)
            self._identified.discard(peer_id)
            try:
                await self._fire_disconnect(peer_id)
            except Exception:
                logger.exception("on_peer_disconnect failed for %s", peer_id)
            return
        if op == "message":
            peer_id = str(ev.get("peer_id", ""))
            raw = ev.get("envelope")
            if not peer_id or not isinstance(raw, dict):
                return
            try:
                env = Envelope.model_validate(raw)
            except Exception:
                logger.debug("pear skip malformed envelope from %s", peer_id)
                return
            if env.type == HELLO and peer_id not in self._identified:
                real_id = str(env.payload.get("node_id") or env.from_id or peer_id)
                if real_id != peer_id:
                    self._peers.discard(peer_id)
                    peer_id = real_id
                self._peers.add(peer_id)
                self._identified.add(peer_id)
                try:
                    await self._fire_connect(peer_id)
                except Exception:
                    logger.exception("on_peer_connect failed for %s", peer_id)
            try:
                await self._fire_message(peer_id, env)
            except Exception:
                logger.exception("on_message failed for %s", peer_id)
