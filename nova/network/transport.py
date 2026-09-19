"""Transport ABC. Pear, TCP, and in-process all speak Envelope ndjson."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from nova.protocol import Envelope

PeerHandler = Callable[[str], Awaitable[None] | None]
MessageHandler = Callable[[str, Envelope], Awaitable[None] | None]


class ControlTransport(ABC):
    """Discovery + control messages. Not the PNG data plane."""

    def __init__(self) -> None:
        self._on_message: MessageHandler | None = None
        self._on_connect: PeerHandler | None = None
        self._on_disconnect: PeerHandler | None = None

    def on_message(self, handler: MessageHandler) -> None:
        self._on_message = handler

    def on_peer_connect(self, handler: PeerHandler) -> None:
        self._on_connect = handler

    def on_peer_disconnect(self, handler: PeerHandler) -> None:
        self._on_disconnect = handler

    async def _fire_message(self, peer_id: str, env: Envelope) -> None:
        if self._on_message:
            res = self._on_message(peer_id, env)
            if res is not None:
                await res

    async def _fire_connect(self, peer_id: str) -> None:
        if self._on_connect:
            res = self._on_connect(peer_id)
            if res is not None:
                await res

    async def _fire_disconnect(self, peer_id: str) -> None:
        if self._on_disconnect:
            res = self._on_disconnect(peer_id)
            if res is not None:
                await res

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, peer_id: str, env: Envelope) -> None: ...

    @abstractmethod
    async def broadcast(self, env: Envelope) -> None: ...
