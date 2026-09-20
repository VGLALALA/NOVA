"""Probe local accelerators. A node is a list of devices, not one backend string."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

from nova.models import KERNEL_SD_T2I, Backend, Device, NodeIdentity, NodeManifest

_BACKEND_RANK: dict[str, int] = {"cuda": 0, "rocm": 0, "metal": 1, "cpu": 2}


def _try_import_torch() -> Any:
    try:
        import torch  # type: ignore[import-untyped]

        return torch
    except ImportError:
        return None


def _system_ram_mb() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 8192


def _cpu_model() -> str:
    return platform.processor() or platform.machine() or "cpu"


def _cpu_device() -> Device:
    ram = _system_ram_mb()
    return Device(
        device_id="cpu:0",
        backend="cpu",
        vendor="cpu",
        model=_cpu_model(),
        memory_total_mb=ram,
        memory_free_mb=ram,
    )


def _apple_gpu_model() -> str:
    machine = platform.machine()
    if machine == "arm64":
        return "Apple Silicon"
    return f"Apple GPU ({machine or 'unknown'})"


def _probe_cuda(torch: Any) -> list[Device]:
    devices: list[Device] = []
    try:
        if not torch.cuda.is_available():
            return devices
    except Exception:
        return devices

    hip = getattr(getattr(torch, "version", None), "hip", None)
    is_rocm = bool(hip)
    backend: Backend = "rocm" if is_rocm else "cuda"
    vendor = "amd" if is_rocm else "nvidia"

    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0

    for i in range(count):
        name = f"GPU {i}"
        memory_total_mb = 0
        memory_free_mb = 0
        try:
            props = torch.cuda.get_device_properties(i)
            name = getattr(props, "name", None) or name
            total = int(getattr(props, "total_memory", 0) or 0)
            memory_total_mb = total // (1024 * 1024)
        except Exception:
            pass
        try:
            free, total = torch.cuda.mem_get_info(i)
            memory_total_mb = int(total) // (1024 * 1024)
            memory_free_mb = int(free) // (1024 * 1024)
        except Exception:
            pass
        devices.append(
            Device(
                device_id=f"cuda:{i}",
                backend=backend,
                vendor=vendor,
                model=str(name),
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_free_mb,
            )
        )
    return devices


def _probe_mps(torch: Any) -> list[Device]:
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            return []
    except Exception:
        return []

    ram = _system_ram_mb()
    conservative = max(ram // 2, 1)
    return [
        Device(
            device_id="mps:0",
            backend="metal",
            vendor="apple",
            model=_apple_gpu_model(),
            memory_total_mb=conservative,
            memory_free_mb=conservative,
        )
    ]


def probe_devices() -> list[Device]:
    """Return every usable device. CPU is last-resort, never advertised as CUDA."""
    devices: list[Device] = []
    torch = _try_import_torch()
    if torch is not None:
        devices.extend(_probe_cuda(torch))
        devices.extend(_probe_mps(torch))
    # NVIDIA / AMD boxes must not list CPU — a CUDA node never falls back to CPU.
    if not any(d.backend in ("cuda", "rocm") for d in devices):
        devices.append(_cpu_device())
    return devices


def preferred_device(
    devices: list[Device],
    allowed_backends: list[str] | None = None,
) -> Device | None:
    """Fastest compatible device: cuda/rocm > metal > cpu."""
    allowed = set(allowed_backends) if allowed_backends is not None else None
    eligible = [d for d in devices if allowed is None or d.backend in allowed]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda d: (_BACKEND_RANK.get(d.backend, 99), -d.memory_total_mb, d.device_id),
    )


def preferred_accelerator(devices: list[Device]) -> Device | None:
    """CUDA, ROCm, or Metal. Never CPU."""
    return preferred_device(devices, ["cuda", "rocm", "metal"])


def to_torch_device(device: Device) -> str:
    if device.backend in ("cuda", "rocm"):
        return device.device_id
    if device.backend == "metal":
        return "mps"
    return "cpu"


def build_manifest(
    identity: NodeIdentity,
    devices: list[Device],
    **kwargs: Any,
) -> NodeManifest:
    hostname = kwargs.pop("hostname", None) or socket.gethostname()
    os_name = kwargs.pop("os", None) or platform.system()
    architecture = kwargs.pop("architecture", None) or platform.machine()
    if "supported_kernels" not in kwargs:
        kwargs["supported_kernels"] = [KERNEL_SD_T2I]
    return NodeManifest(
        node_id=identity.node_id,
        hostname=hostname,
        os=os_name,
        architecture=architecture,
        devices=list(devices),
        **kwargs,
    )
