"""Dummy kernel: valid PNGs with Pillow. No torch required."""

from __future__ import annotations

import hashlib
import io
import random
import textwrap
import time

from PIL import Image, ImageDraw, ImageFont

from nova.kernels.base import NovaKernel
from nova.models import KERNEL_SD_T2I, Device, KernelResult, Task


class DummyKernel(NovaKernel):
    """Stand-in for sd.t2i.v1 used by tests, --simulate, and machines without GPU."""

    kernel_id = KERNEL_SD_T2I

    @classmethod
    def compatible(cls, device: Device) -> bool:
        return True

    @classmethod
    def estimate(cls, task: Task, device: Device) -> float:
        return 0.05

    def execute_sync(self, task: Task, device: Device) -> KernelResult:
        t0 = time.perf_counter()
        width = int(task.width or 512)
        height = int(task.height or 512)
        rng = random.Random(int(task.seed))
        bg = (rng.randint(24, 196), rng.randint(24, 196), rng.randint(24, 196))
        fg = (255, 255, 255)

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        margin = 16
        max_chars = max(12, width // 6)
        lines = textwrap.wrap(task.prompt or "", width=max_chars)
        lines = lines[: max(1, (height - 2 * margin) // 14)]
        lines.extend(
            [
                "",
                f"backend={device.backend} {device.device_id}",
                f"seed={task.seed} steps={task.steps}",
            ]
        )
        y = margin
        for line in lines:
            draw.text((margin, y), line, fill=fg, font=font)
            y += 14
            if y > height - margin:
                break

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        execution_ms = max(int((time.perf_counter() - t0) * 1000), 0)
        return KernelResult(
            png_bytes=png,
            sha256=hashlib.sha256(png).hexdigest(),
            execution_ms=execution_ms,
            device_id=device.device_id,
            backend=device.backend,
        )
