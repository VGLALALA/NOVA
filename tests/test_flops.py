import pytest

from nova.flops import fp16_tflops


def test_fp16_tflops_from_warmup_ms() -> None:
    # 679 GMAC * 2 FLOPs / 0.312 s ≈ 4.35 TFLOPS
    assert fp16_tflops(312, steps=1) == pytest.approx(4.353, rel=1e-3)


def test_fp16_tflops_scales_with_steps() -> None:
    one = fp16_tflops(400, steps=1)
    four = fp16_tflops(1600, steps=4)
    assert four == pytest.approx(one, rel=1e-6)


def test_gemm_fp16_tflops_formula() -> None:
    n, iters, seconds = 4096, 8, 0.2
    expected = (2.0 * (n**3) * iters) / seconds / 1e12
    assert expected == pytest.approx(5.497, rel=1e-3)


def test_measure_device_fp16_tflops_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from nova.flops import measure_device_fp16_tflops

    real_import = builtins.__import__

    def _block_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torch)
    assert measure_device_fp16_tflops("cpu") is None
