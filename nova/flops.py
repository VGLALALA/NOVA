"""FP16 TFLOPS.

`gemm_fp16_tflops` is the dashboard number: a square FP16 matmul
(2 N^3 FLOPs). This is the same family of probe used by NVIDIA's
cuda-samples `cudaTensorCoreGemm` / `cublasLt` peak checks, expressed
as a short torch GEMM that runs on CUDA, ROCm, and Metal.

`unet_fp16_tflops` is optional UNet-GMAC math from SD-Turbo latency.
Do not display that as device peak — a 1-step 512 image is memory-bound.
"""

from __future__ import annotations

from typing import Any

GMAC_PER_STEP_512 = 679.0
GEMM_N = 4096
GEMM_ITERS = 8


def fp16_tflops(
    latency_ms: float,
    *,
    steps: int = 1,
    width: int = 512,
    height: int = 512,
    gmac_per_step_512: float = GMAC_PER_STEP_512,
) -> float:
    """UNet-equivalent TFLOPS from image latency. Not device peak."""
    ms = max(float(latency_ms), 1.0)
    scale = (max(int(width), 1) * max(int(height), 1)) / (512.0 * 512.0)
    flops = max(int(steps), 1) * float(gmac_per_step_512) * 1e9 * 2.0 * scale
    return flops / (ms / 1000.0) / 1e12


def _sync(torch: Any, device: Any) -> None:
    kind = getattr(device, "type", None)
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        fn = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(fn):
            fn()


def gemm_fp16_tflops(
    torch: Any,
    device: Any,
    *,
    n: int = GEMM_N,
    iters: int = GEMM_ITERS,
) -> tuple[float, float]:
    """Time FP16 GEMM. Returns (tflops, elapsed_ms)."""
    dtype = getattr(torch, "float16", None) or torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    _sync(torch, device)
    for _ in range(2):
        torch.mm(a, b)
    _sync(torch, device)
    t0 = torch_timer()
    out = None
    for _ in range(max(iters, 1)):
        out = torch.mm(a, b)
    _sync(torch, device)
    elapsed_s = max(torch_timer() - t0, 1e-6)
    flops = 2.0 * (n**3) * max(iters, 1)
    tflops = flops / elapsed_s / 1e12
    del a, b, out
    return tflops, elapsed_s * 1000.0


def torch_timer() -> float:
    import time

    return time.perf_counter()


def measure_device_fp16_tflops(device_id: str) -> float | None:
    """Run GEMM on a live torch device. None if torch/device missing."""
    try:
        import torch
    except ImportError:
        return None
    device = torch.device(device_id)
    for n in (GEMM_N, 2048, 1024):
        try:
            tflops, _ms = gemm_fp16_tflops(torch, device, n=n)
            return float(tflops)
        except Exception:
            continue
    return None
