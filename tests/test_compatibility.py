from __future__ import annotations

from nova.clock import now_utc
from nova.models import Device, NodeManifest, Task
from nova.scheduler import is_compatible, score_node


def _node(**kwargs) -> NodeManifest:
    device = kwargs.pop(
        "device",
        Device(
            device_id="cuda:0",
            backend="cuda",
            vendor="nvidia",
            model="RTX 4090",
            memory_total_mb=24576,
        ),
    )
    defaults = dict(
        node_id="n1",
        devices=[device],
        supported_kernels=["sd.t2i.v1"],
        max_concurrency=1,
        current_slots_used=0,
        status="online",
    )
    defaults.update(kwargs)
    return NodeManifest(**defaults)


def _task(**kwargs) -> Task:
    defaults = dict(
        task_id="t0",
        job_id="j",
        shard_index=0,
        prompt="x",
        seed=1,
        created_at=now_utc(),
        min_memory_mb=4096,
        allowed_backends=["cuda", "rocm", "metal"],
    )
    defaults.update(kwargs)
    return Task(**defaults)


def test_incompatible_backend_rejected() -> None:
    node = _node(
        device=Device(
            device_id="cpu:0", backend="cpu", vendor="cpu", model="cpu", memory_total_mb=16000
        )
    )
    assert is_compatible(node, _task()) is False


def test_offline_node_rejected() -> None:
    assert is_compatible(_node(status="offline"), _task()) is False


def test_slot_full_rejected() -> None:
    assert is_compatible(_node(current_slots_used=1, max_concurrency=1), _task()) is False


def test_memory_too_small_rejected() -> None:
    node = _node(
        device=Device(
            device_id="cuda:0",
            backend="cuda",
            vendor="nvidia",
            model="GTX",
            memory_total_mb=2048,
        )
    )
    assert is_compatible(node, _task(min_memory_mb=4096)) is False


def test_missing_score_still_eligible() -> None:
    node = _node()
    task = _task()
    assert is_compatible(node, task) is True
    assert score_node(node, task) == 0.1
