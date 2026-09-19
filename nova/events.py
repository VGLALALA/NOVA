"""In-process event bus for dashboard WebSocket + coordinator logging."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any


class EventBus:
    def __init__(self, history: int = 200) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        event = {
            "type": type_,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._history.append(event)
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)
