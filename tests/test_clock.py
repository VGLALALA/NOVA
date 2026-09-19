from datetime import timezone

from nova.clock import now_utc


def test_now_utc_is_timezone_aware() -> None:
    ts = now_utc()
    assert ts.tzinfo is not None
    assert ts.tzinfo == timezone.utc


def test_fake_clock_advances_without_sleeping(clock) -> None:
    t0 = clock.now()
    clock.advance(30)
    t1 = clock.now()
    assert (t1 - t0).total_seconds() == 30
    assert t1.tzinfo == timezone.utc
