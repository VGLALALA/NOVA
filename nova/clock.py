"""Coordinator clock is the only clock. All timestamps UTC."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

NowFn = Callable[[], datetime]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Clock:
    """Injectable clock so lease tests do not sleep."""

    def __init__(self, fn: NowFn | None = None) -> None:
        self._fn = fn or now_utc

    def now(self) -> datetime:
        return self._fn()
