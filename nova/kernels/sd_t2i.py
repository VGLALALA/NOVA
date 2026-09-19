"""sd.t2i.v1 — Hugging Face diffusers, local weights only."""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Any

from nova.config import Settings
from nova.hardware import to_torch_device
from nova.kernels.base import NovaKernel
from nova.models import KERNEL_SD_T2I, Device, KernelResult, Task

_MISSING = (
    "torch/diffusers are not installed. Use the dummy kernel "
    "(NOVA_DUMMY=1 or load_kernel('dummy', settings))."
)

_SINGLE_FILE_NAMES = (
    "sd_turbo.safetensors",
    "sd-turbo.safetensors",
    "model.safetensors",
)


def _has_component_weights(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    return any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))


def snapshot_complete(model_dir: Path) -> bool:
    """True when a Hugging Face-style snapshot can load offline."""
    if not (model_dir / "model_index.json").is_file():
        return False
    if not (model_dir / "tokenizer" / "vocab.json").is_file():
        return False
    return (
        _has_component_weights(model_dir / "unet")
        and _has_component_weights(model_dir / "vae")
        and _has_component_weights(model_dir / "text_encoder")
    )


def find_single_file(model_dir: Path) -> Path | None:
    if model_dir.is_file() and model_dir.suffix.lower() == ".safetensors":
        return model_dir
    if not model_dir.is_dir():
        return None
    for name in _SINGLE_FILE_NAMES:
        candidate = model_dir / name
        if candidate.is_file() or candidate.is_symlink():
            try:
                if candidate.stat().st_size > 0:
                    return candidate
            except OSError:
                continue
    safetensors = []
    for path in model_dir.glob("*.safetensors"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                safetensors.append(path)
        except OSError:
            continue
    if len(safetensors) == 1:
        return safetensors[0]
    return None


def resolve_pretrained(settings: Settings) -> str:
    """Pick a local snapshot dir, a single-file checkpoint, or the configured model id."""
    model_dir = Path(settings.model_dir)
    if model_dir.is_file() and model_dir.suffix.lower() == ".safetensors":
        return str(model_dir)
    if model_dir.is_dir():
        if snapshot_complete(model_dir):
            return str(model_dir)
        single = find_single_file(model_dir)
        if single is not None:
            return str(single)
        if (model_dir / "model_index.json").is_file():
            return str(model_dir)
        if any(model_dir.iterdir()):
            return str(model_dir)
    return str(settings.model_id)


def is_single_file_checkpoint(path: str | Path) -> bool:
    return Path(path).is_file() and Path(path).suffix.lower() == ".safetensors"


def uses_turbo_recipe(settings: Settings, pretrained: str) -> bool:
    haystack = f"{settings.model_id} {pretrained}".lower()
    return "turbo" in haystack


class SDText2ImageKernel(NovaKernel):
    kernel_id = KERNEL_SD_T2I

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._pipe: Any = None
        self._torch_device: Any = None
        self._pretrained: str | None = None

    @classmethod
    def compatible(cls, device: Device) -> bool:
        return device.backend in {"cuda", "rocm", "metal", "cpu"}

    @classmethod
    def estimate(cls, task: Task, device: Device) -> float:
        per_step = {"cuda": 0.4, "rocm": 0.5, "metal": 1.0, "cpu": 8.0}
        return per_step.get(device.backend, 2.0) * max(int(task.steps or 1), 1)

    def load(self, device: Device) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(_MISSING) from exc

        torch_dev = torch.device(to_torch_device(device))
        dtype = torch.float32 if device.backend == "cpu" else torch.float16
        if device.backend == "metal":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        pretrained = resolve_pretrained(self.settings)
        self._pretrained = pretrained
        config_dir = Path(self.settings.model_dir)
        local_config = str(config_dir) if (config_dir / "model_index.json").is_file() else None

        try:
            if is_single_file_checkpoint(pretrained):
                kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    "local_files_only": True,
                    "use_safetensors": True,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                }
                if local_config is not None:
                    kwargs["config"] = local_config
                pipe = StableDiffusionPipeline.from_single_file(pretrained, **kwargs)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    pretrained,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {pretrained} with local_files_only=True. "
                "Place a Hugging Face snapshot or sd_turbo.safetensors under "
                f"{self.settings.model_dir}, or use the dummy kernel (NOVA_DUMMY=1)."
            ) from exc

        pipe = pipe.to(torch_dev)
        if device.backend == "metal":
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._torch_device = torch_dev

    def execute_sync(self, task: Task, device: Device) -> KernelResult:
        self.load(device)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(_MISSING) from exc

        t0 = time.perf_counter()
        pretrained = self._pretrained or resolve_pretrained(self.settings)
        guidance = 0.0 if uses_turbo_recipe(self.settings, pretrained) else 7.5
        if self._torch_device is not None and self._torch_device.type == "mps":
            generator = torch.Generator(device="cpu").manual_seed(int(task.seed))
        else:
            generator = torch.Generator(device=self._torch_device).manual_seed(int(task.seed))

        out = self._pipe(
            prompt=task.prompt,
            num_inference_steps=int(task.steps or 4),
            guidance_scale=guidance,
            width=int(task.width or 512),
            height=int(task.height or 512),
            generator=generator,
        )
        image = out.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png = buf.getvalue()
        execution_ms = max(int((time.perf_counter() - t0) * 1000), 0)
        return KernelResult(
            png_bytes=png,
            sha256=hashlib.sha256(png).hexdigest(),
            execution_ms=execution_ms,
            device_id=device.device_id,
            backend=device.backend,
        )
