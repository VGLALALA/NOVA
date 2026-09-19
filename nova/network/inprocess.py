"""In-process control plane for tests. No sockets."""

from __future__ import annotations

import asyncio
import logging

from nova.network.transport import ControlTransport
from nova.protocol import Envelope

logger = logging.getLogger("nova.network.inprocess")


class InProcessBroker:
    """Shared bus. One per test / simulated mesh."""

    def __init__(self) -> None:
        self._transports: dict[str, InProcessTransport] = {}

    def register(self, transport: InProcessTransport) -> list[InProcessTransport]:
        others = [t for t in self._transports.values() if t.node_id != transport.node_id]
        self._transports[transport.node_id] = transport
        return others

    def unregister(self, node_id: str) -> list[InProcessTransport]:
        self._transports.pop(node_id, None)
        return list(self._transports.values())

    def get(self, node_id: str) -> InProcessTransport | None:
        return self._transports.get(node_id)


class InProcessTransport(ControlTransport):
    """Multiple transports share a broker. stop() is the Ctrl+C stand-in."""

    def __init__(self, broker: InProcessBroker, node_id: str) -> None:
        super().__init__()
        self.broker = broker
        self.node_id = node_id
        self._started = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._started:
            return
        others = self.broker.register(self)
        self._started = True
        for other in others:
            await other._fire_connect(self.node_id)
            await self._fire_connect(other.node_id)

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        others = self.broker.unregister(self.node_id)
        for other in others:
            try:
                await other._fire_disconnect(self.node_id)
            except Exception:
                logger.exception("in-process disconnect handler failed")
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def send(self, peer_id: str, env: Envelope) -> None:
        dest = self.broker.get(peer_id)
        if dest is None or not dest._started:
            return
        delivered = Envelope.model_validate(env.model_dump(mode="json"))
        self._spawn(dest._fire_message(self.node_id, delivered))

    async def broadcast(self, env: Envelope) -> None:
        for node_id, dest in list(self.broker._transports.items()):
            if node_id == self.node_id or not dest._started:
                continue
            delivered = Envelope.model_validate(env.model_dump(mode="json"))
            self._spawn(dest._fire_message(self.node_id, delivered))

    def _spawn(self, coro: object) -> None:
        task = asyncio.create_task(coro)  # type: ignore[arg-type]
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
