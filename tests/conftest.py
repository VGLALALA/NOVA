from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nova.clock import Clock


class FakeClock(Clock):
    def __init__(self, start: datetime | None = None) -> None:
        self._t = start or datetime(2026, 9, 19, tzinfo=timezone.utc)
        super().__init__(fn=lambda: self._t)

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._t = self._t + timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".nova"
    d.mkdir()
    return d
