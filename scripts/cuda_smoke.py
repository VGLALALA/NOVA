#!/usr/bin/env python3
"""Generate one local CUDA 512x512 image. Does not hit Hugging Face."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nova.clock import now_utc
from nova.config import Settings
from nova.hardware import preferred_accelerator, probe_devices
from nova.kernels.sd_t2i import SDText2ImageKernel, resolve_pretrained, snapshot_complete
from nova.models import Task
from nova.pngutil import validate_png


def main() -> int:
    model_dir = Path("models/sd-turbo")
    settings = Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo", dummy=False)
    print("snapshot_complete", snapshot_complete(model_dir))
    print("resolve", resolve_pretrained(settings))
    devices = probe_devices()
    print("devices", [(d.backend, d.device_id, d.model, d.memory_total_mb) for d in devices])
    device = preferred_accelerator(devices)
    print("preferred", None if device is None else (device.backend, device.device_id, device.model))
    if device is None or device.backend != "cuda":
        raise SystemExit(f"expected cuda accelerator, got {device}")

    kernel = SDText2ImageKernel(settings)
    t0 = time.perf_counter()
    warmup = kernel.warmup(device)
    print(
        json.dumps(
            {
                "warmup_backend": warmup.backend,
                "warmup_device": warmup.device_id,
                "warmup_execution_ms": warmup.execution_ms,
                "warmup_wall_s": round(time.perf_counter() - t0, 2),
                "warmup_png": len(warmup.png_bytes),
            },
            indent=2,
        )
    )
    Path("/tmp/nova-cuda-warmup.png").write_bytes(warmup.png_bytes)
    validate_png(warmup.png_bytes, 512, 512)

    task = Task(
        task_id="cuda-1",
        job_id="cuda",
        shard_index=0,
        prompt="a neon street at night, cinematic",
        seed=1000,
        steps=4,
        width=512,
        height=512,
        created_at=now_utc(),
    )
    t1 = time.perf_counter()
    tile = kernel.execute_sync(task, device)
    print(
        json.dumps(
            {
                "tile_backend": tile.backend,
                "tile_device": tile.device_id,
                "tile_execution_ms": tile.execution_ms,
                "tile_wall_s": round(time.perf_counter() - t1, 2),
                "tile_png": len(tile.png_bytes),
            },
            indent=2,
        )
    )
    Path("/tmp/nova-cuda-tile.png").write_bytes(tile.png_bytes)
    validate_png(tile.png_bytes, 512, 512)
    print("OK cuda images written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
