"""NovaKernel ABC and kernel loader."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nova.clock import now_utc
from nova.config import Settings
from nova.models import KERNEL_SD_T2I, Device, KernelResult, Task


class NovaKernel(ABC):
    kernel_id: str = KERNEL_SD_T2I

    @classmethod
    @abstractmethod
    def compatible(cls, device: Device) -> bool: ...

    @classmethod
    @abstractmethod
    def estimate(cls, task: Task, device: Device) -> float:
        """Estimated runtime in seconds."""

    @abstractmethod
    def execute_sync(self, task: Task, device: Device) -> KernelResult: ...

    def load(self, device: Device) -> None:
        """Load weights onto device. Default is a no-op."""

    def warmup(self, device: Device) -> KernelResult:
        """Generate one discarded image. Must be called off the asyncio thread."""
        self.load(device)
        task = Task(
            task_id="warmup",
            job_id="warmup",
            shard_index=0,
            kernel_id=self.kernel_id,
            prompt="warmup",
            seed=0,
            steps=1,
            width=512,
            height=512,
            min_memory_mb=0,
            allowed_backends=["cuda", "rocm", "metal", "cpu"],
            created_at=now_utc(),
        )
        return self.execute_sync(task, device)


def gpu_runtime_available() -> bool:
    """True when torch and diffusers can be imported."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def load_kernel(name: str, settings: Settings) -> NovaKernel:
    """Dummy only when requested. Real GPU startup fails closed if deps are missing."""
    want_dummy = bool(settings.dummy) or name in {"dummy", "dummy.t2i", "dummy.t2i.v1"}
    if want_dummy:
        from nova.kernels.dummy import DummyKernel

        return DummyKernel()

    if not gpu_runtime_available():
        raise RuntimeError(
            "sd.t2i.v1 requires torch and diffusers. "
            "Install GPU extras or start with --dummy / NOVA_DUMMY=1."
        )

    from nova.kernels.sd_t2i import SDText2ImageKernel

    return SDText2ImageKernel(settings)
