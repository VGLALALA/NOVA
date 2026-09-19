from __future__ import annotations

from nova.hardware import (
    build_manifest,
    preferred_accelerator,
    preferred_device,
    probe_devices,
    to_torch_device,
)
from nova.identity import load_or_create


def test_probe_returns_cpu(data_dir) -> None:
    devices = probe_devices()
    assert devices
    assert any(d.backend == "cpu" for d in devices)
    assert not any(d.backend == "cuda" and d.vendor == "cpu" for d in devices)


def test_preferred_device_order() -> None:
    from nova.models import Device

    cpu = Device(device_id="cpu:0", backend="cpu", vendor="cpu", model="x", memory_total_mb=8)
    metal = Device(device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16)
    cuda = Device(device_id="cuda:0", backend="cuda", vendor="nvidia", model="N", memory_total_mb=24)
    assert preferred_device([cpu, metal, cuda]).backend == "cuda"
    assert preferred_device([cpu, metal], ["metal", "cpu"]).backend == "metal"
    assert preferred_device([cpu, metal], ["rocm"]) is None
    assert preferred_accelerator([cpu, metal, cuda]).backend == "cuda"
    assert preferred_accelerator([cpu, metal]).backend == "metal"
    assert preferred_accelerator([cpu]) is None


def test_to_torch_device_maps_backends() -> None:
    from nova.models import Device

    cuda = Device(device_id="cuda:0", backend="cuda", vendor="nvidia", model="N", memory_total_mb=24)
    rocm = Device(device_id="cuda:0", backend="rocm", vendor="amd", model="A", memory_total_mb=24)
    metal = Device(device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16)
    cpu = Device(device_id="cpu:0", backend="cpu", vendor="cpu", model="x", memory_total_mb=8)
    assert to_torch_device(cuda) == "cuda:0"
    assert to_torch_device(rocm) == "cuda:0"
    assert to_torch_device(metal) == "mps"
    assert to_torch_device(cpu) == "cpu"


def test_manifest_identity(data_dir) -> None:
    ident = load_or_create(data_dir)
    manifest = build_manifest(ident, probe_devices())
    assert manifest.node_id == ident.node_id
    assert "sd.t2i.v1" in manifest.supported_kernels
