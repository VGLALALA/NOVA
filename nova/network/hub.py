"""Fan-in multiplexer: run TCP + Pear as one ControlTransport."""

from __future__ import annotations

import logging

from nova.network.transport import ControlTransport
from nova.protocol import Envelope

logger = logging.getLogger("nova.network.hub")


class Hub(ControlTransport):
    """Route send() to the transport that owns the peer. Broadcast to all."""

    def __init__(self, transports: list[ControlTransport]) -> None:
        super().__init__()
        self._transports = list(transports)
        self._route: dict[str, ControlTransport] = {}

    async def start(self) -> None:
        for t in self._transports:
            self._bind(t)
            await t.start()

    async def stop(self) -> None:
        for t in self._transports:
            try:
                await t.stop()
            except Exception:
                logger.exception("hub child stop failed")
        self._route.clear()

    async def send(self, peer_id: str, env: Envelope) -> None:
        t = self._route.get(peer_id)
        if t is not None:
            await t.send(peer_id, env)
            return
        for child in self._transports:
            await child.send(peer_id, env)

    async def broadcast(self, env: Envelope) -> None:
        for child in self._transports:
            await child.broadcast(env)

    def _bind(self, t: ControlTransport) -> None:
        async def on_msg(peer_id: str, env: Envelope) -> None:
            self._route.setdefault(peer_id, t)
            await self._fire_message(peer_id, env)

        async def on_con(peer_id: str) -> None:
            self._route[peer_id] = t
            await self._fire_connect(peer_id)

        async def on_dis(peer_id: str) -> None:
            if self._route.get(peer_id) is t:
                self._route.pop(peer_id, None)
            await self._fire_disconnect(peer_id)

        t.on_message(on_msg)
        t.on_peer_connect(on_con)
        t.on_peer_disconnect(on_dis)
