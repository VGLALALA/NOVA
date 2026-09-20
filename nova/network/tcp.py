"""Direct TCP control plane. Emergency / LAN path. ndjson Envelopes."""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from dataclasses import dataclass, field

from nova.network.transport import ControlTransport
from nova.protocol import HELLO, PROTOCOL_VERSION, Envelope, msg

logger = logging.getLogger("nova.network.tcp")

_MAX_LINE_BUFFER = 1_000_000


def _enable_nodelay(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


@dataclass
class _Peer:
    peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    identified: bool = False
    closed: bool = False


class TcpTransport(ControlTransport):
    """Coordinator listens; workers connect. Socket close = disconnect now."""

    def __init__(
        self,
        *,
        listen_host: str | None,
        listen_port: int | None,
        connect_addrs: list[tuple[str, int]] | None,
        node_id: str,
    ) -> None:
        super().__init__()
        self.node_id = node_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.connect_addrs = list(connect_addrs or [])
        self.bound_host: str | None = None
        self.bound_port: int | None = None
        self._server: asyncio.AbstractServer | None = None
        self._peers: dict[str, _Peer] = {}
        self._client_tasks: list[asyncio.Task[None]] = []
        self._dialed: set[tuple[str, int]] = set()
        self._stopping = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        self._started = True
        if self.listen_host is not None:
            port = 7946 if self.listen_port is None else self.listen_port
            self._server = await asyncio.start_server(
                self._accepted,
                self.listen_host,
                port,
            )
            sock = self._server.sockets[0]
            host, bound_port = sock.getsockname()[:2]
            self.bound_host = host
            self.bound_port = int(bound_port)
            self.listen_port = self.bound_port
            logger.info("control listening on %s:%s", host, self.bound_port)
        for host, port in self.connect_addrs:
            self._spawn_client(host, port)

    def _spawn_client(self, host: str, port: int) -> None:
        key = (str(host), int(port))
        if key not in self.connect_addrs:
            self.connect_addrs.append(key)
        if key in self._dialed:
            return
        self._dialed.add(key)
        if self._started:
            self._client_tasks.append(asyncio.create_task(self._client_loop(host, int(port))))

    def add_connect(self, host: str, port: int) -> None:
        """Dial a forwarded worker TCP. Safe to call after start()."""
        self._spawn_client(host, int(port))

    async def drop_peer(self, peer_id: str) -> bool:
        peer = self._peers.get(peer_id)
        if peer is None:
            return False
        await self._teardown(peer)
        return True

    def peer_ids(self) -> list[str]:
        return [p.peer_id for p in self._peers.values() if not p.closed]

    async def stop(self) -> None:
        self._stopping = True
        for task in self._client_tasks:
            task.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
        self._client_tasks.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for peer in list(self._peers.values()):
            await self._teardown(peer)
        self._started = False

    async def send(self, peer_id: str, env: Envelope) -> None:
        peer = self._peers.get(peer_id)
        if peer is None or peer.closed:
            return
        await self._write(peer, env)

    async def broadcast(self, env: Envelope) -> None:
        for peer in list(self._peers.values()):
            if peer.closed:
                continue
            try:
                await self._write(peer, env)
            except Exception:
                logger.exception("broadcast to %s failed", peer.peer_id)

    async def _accepted(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await self._run_peer(reader, writer)

    async def _client_loop(self, host: str, port: int) -> None:
        delay = 0.25
        while not self._stopping:
            try:
                reader, writer = await asyncio.open_connection(host, port)
                delay = 0.25
                print(f"[nova] control connected {host}:{port}", flush=True)
                await self._run_peer(reader, writer)
            except asyncio.CancelledError:
                raise
            except OSError as exc:
                logger.warning("control connect %s:%s failed: %s", host, port, exc)
                print(f"[nova] control connect {host}:{port} failed: {exc}", flush=True)
            if self._stopping:
                break
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            delay = min(delay * 2, 5.0)

    async def _run_peer(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _enable_nodelay(writer)
        temp_id = f"tmp-{uuid.uuid4().hex[:8]}"
        peer = _Peer(peer_id=temp_id, reader=reader, writer=writer)
        self._peers[temp_id] = peer
        try:
            await self._write(
                peer,
                msg(
                    HELLO,
                    self.node_id,
                    protocol_version=PROTOCOL_VERSION,
                    node_id=self.node_id,
                ),
            )
            await self._read_loop(peer)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("peer %s error", peer.peer_id)
        finally:
            await self._teardown(peer)

    async def _read_loop(self, peer: _Peer) -> None:
        buf = b""
        while not self._stopping and not peer.closed:
            chunk = await peer.reader.read(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > _MAX_LINE_BUFFER:
                logger.warning("ndjson buffer overflow from %s; dropping", peer.peer_id)
                buf = b""
                continue
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl]
                buf = buf[nl + 1 :]
                if not line.strip():
                    continue
                await self._dispatch_line(peer, line)

    async def _dispatch_line(self, peer: _Peer, line: bytes) -> None:
        try:
            env = Envelope.decode(line)
        except Exception:
            logger.debug("skip malformed envelope from %s", peer.peer_id)
            return
        if env.type == HELLO and not peer.identified:
            await self._identify(peer, env)
        try:
            await self._fire_message(peer.peer_id, env)
        except Exception:
            logger.exception("on_message failed for %s", peer.peer_id)

    async def _identify(self, peer: _Peer, env: Envelope) -> None:
        real_id = str(env.payload.get("node_id") or env.from_id)
        old_id = peer.peer_id
        existing = self._peers.get(real_id)
        if existing is not None and existing is not peer:
            await self._teardown(existing)
        if old_id in self._peers and self._peers[old_id] is peer:
            del self._peers[old_id]
        peer.peer_id = real_id
        peer.identified = True
        self._peers[real_id] = peer
        try:
            await self._fire_connect(real_id)
        except Exception:
            logger.exception("on_peer_connect failed for %s", real_id)

    async def _write(self, peer: _Peer, env: Envelope) -> None:
        async with peer.write_lock:
            if peer.closed:
                return
            try:
                peer.writer.write(env.encode())
                await peer.writer.drain()
            except (ConnectionError, BrokenPipeError, OSError):
                try:
                    peer.writer.close()
                except Exception:
                    pass

    async def _teardown(self, peer: _Peer) -> None:
        if peer.closed:
            return
        peer.closed = True
        if self._peers.get(peer.peer_id) is peer:
            del self._peers[peer.peer_id]
        try:
            peer.writer.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(peer.writer.wait_closed(), timeout=1.0)
        except Exception:
            pass
        try:
            await self._fire_disconnect(peer.peer_id)
        except Exception:
            logger.exception("on_peer_disconnect failed for %s", peer.peer_id)
