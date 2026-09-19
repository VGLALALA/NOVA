from __future__ import annotations

from nova.clock import now_utc
from nova.config import Settings
from nova.kernels import load_kernel
from nova.kernels.dummy import DummyKernel
from nova.models import Device, Task
from nova.pngutil import sha256_bytes, validate_png


def test_dummy_png_size_and_hash() -> None:
    kernel = DummyKernel()
    device = Device(
        device_id="cpu:0", backend="cpu", vendor="cpu", model="cpu", memory_total_mb=4096
    )
    task = Task(
        task_id="t",
        job_id="j",
        shard_index=0,
        prompt="neon street at night",
        seed=1000,
        width=512,
        height=512,
        created_at=now_utc(),
    )
    result = kernel.execute_sync(task, device)
    w, h = validate_png(result.png_bytes, 512, 512)
    assert (w, h) == (512, 512)
    assert result.sha256 == sha256_bytes(result.png_bytes)
    assert result.backend == "cpu"


def test_load_kernel_dummy_flag() -> None:
    k = load_kernel("sd.t2i.v1", Settings(dummy=True))
    assert isinstance(k, DummyKernel)


def test_load_kernel_named_dummy() -> None:
    k = load_kernel("dummy", Settings(dummy=False))
    assert isinstance(k, DummyKernel)


def test_load_kernel_missing_gpu_deps_raises(monkeypatch) -> None:
    from nova.kernels import base as kernel_base

    monkeypatch.setattr(kernel_base, "gpu_runtime_available", lambda: False)
    try:
        load_kernel("sd.t2i.v1", Settings(dummy=False))
    except RuntimeError as exc:
        assert "torch" in str(exc)
        assert "dummy" in str(exc).lower()
    else:
        raise AssertionError("expected real kernel load to fail closed")
