from __future__ import annotations

from pathlib import Path

from nova.config import Settings
from nova.kernels.sd_t2i import (
    is_single_file_checkpoint,
    resolve_pretrained,
    snapshot_complete,
    uses_turbo_recipe,
)


def _complete_snapshot(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text("{}")
    (model_dir / "tokenizer").mkdir()
    (model_dir / "tokenizer" / "vocab.json").write_text("{}")
    for name in ("unet", "vae", "text_encoder"):
        folder = model_dir / name
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"weights")


def test_resolve_pretrained_prefers_complete_snapshot(tmp_path: Path) -> None:
    model_dir = tmp_path / "sd-turbo"
    _complete_snapshot(model_dir)
    (model_dir / "sd_turbo.safetensors").write_bytes(b"not-a-real-checkpoint")
    settings = Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo")
    assert snapshot_complete(model_dir)
    assert resolve_pretrained(settings) == str(model_dir)
    assert not is_single_file_checkpoint(resolve_pretrained(settings))


def test_incomplete_snapshot_falls_back_to_single_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "sd-turbo"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")
    weights = model_dir / "sd_turbo.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    settings = Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo")
    assert not snapshot_complete(model_dir)
    assert resolve_pretrained(settings) == str(weights)


def test_resolve_pretrained_uses_named_single_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "sd-turbo"
    model_dir.mkdir()
    weights = model_dir / "sd_turbo.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    settings = Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo")
    assert resolve_pretrained(settings) == str(weights)
    assert is_single_file_checkpoint(weights)


def test_resolve_pretrained_accepts_direct_file(tmp_path: Path) -> None:
    weights = tmp_path / "sd_turbo.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    settings = Settings(model_dir=weights, model_id="stabilityai/sd-turbo")
    assert resolve_pretrained(settings) == str(weights)


def test_resolve_pretrained_falls_back_to_lone_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    weights = model_dir / "custom.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    settings = Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo")
    assert resolve_pretrained(settings) == str(weights)


def test_resolve_pretrained_falls_back_to_model_id(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    settings = Settings(model_dir=missing, model_id="stabilityai/sd-turbo")
    assert resolve_pretrained(settings) == "stabilityai/sd-turbo"


def test_turbo_recipe_from_filename(tmp_path: Path) -> None:
    weights = tmp_path / "sd_turbo.safetensors"
    settings = Settings(model_dir=weights, model_id="local")
    assert uses_turbo_recipe(settings, str(weights))
    assert not uses_turbo_recipe(
        Settings(model_dir=tmp_path / "sd15", model_id="runwayml/stable-diffusion-v1-5"),
        "runwayml/stable-diffusion-v1-5",
    )


class _FakeModule:
    def __init__(self) -> None:
        self.memory_format = None

    def to(self, *args, **kwargs):
        self.memory_format = kwargs.get("memory_format")
        return self


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None
        self.sliced = False
        self.vae_sliced = False
        self.vae_tiled = False
        self.xformers = False
        self.safety_checker = object()
        self.requires_safety_checker = True
        self.feature_extractor = object()
        self.unet = _FakeModule()
        self.vae = _FakeModule()
        self.calls: list[dict[str, object]] = []

    def to(self, device):
        self.moved_to = device
        return self

    def enable_attention_slicing(self) -> None:
        self.sliced = True

    def enable_vae_slicing(self) -> None:
        self.vae_sliced = True

    def enable_vae_tiling(self) -> None:
        self.vae_tiled = True

    def enable_xformers_memory_efficient_attention(self) -> None:
        self.xformers = True

    def set_progress_bar_config(self, **kwargs) -> None:
        return None

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        image = type("Img", (), {"save": lambda _self, buf, format="PNG": buf.write(b"\x89PNG")})()
        return type("Out", (), {"images": [image]})()


def test_kernel_load_dispatches_single_file(monkeypatch, tmp_path: Path) -> None:
    from nova.kernels.sd_t2i import SDText2ImageKernel
    from nova.models import Device

    weights = tmp_path / "sd_turbo.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    captured: dict[str, object] = {}
    fake = _FakePipe()

    class FakeSingle:
        @staticmethod
        def from_single_file(path, **kwargs):
            captured["single"] = {"path": path, **kwargs}
            return fake

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("snapshot loader should not run for a .safetensors file")

    import sys
    import types

    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, name: str) -> None:
            self.type = str(name).split(":")[0]
            self.name = str(name)

        def __repr__(self) -> str:
            return self.name

    fake_torch.device = FakeDevice
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoPipelineForText2Image = FakeAuto
    fake_diffusers.StableDiffusionPipeline = FakeSingle
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    kernel = SDText2ImageKernel(Settings(model_dir=weights, model_id="stabilityai/sd-turbo"))
    device = Device(
        device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16
    )
    kernel.load(device)
    assert captured["single"]["path"] == str(weights)
    assert captured["single"]["local_files_only"] is True
    assert captured["single"]["safety_checker"] is None
    assert fake.sliced is True
    assert fake.vae_sliced is True
    assert fake.vae_tiled is True
    assert fake.feature_extractor is None
    assert kernel._pretrained == str(weights)


def test_kernel_load_single_file_uses_local_config(monkeypatch, tmp_path: Path) -> None:
    from nova.kernels.sd_t2i import SDText2ImageKernel
    from nova.models import Device

    model_dir = tmp_path / "sd-turbo"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")
    weights = model_dir / "sd_turbo.safetensors"
    weights.write_bytes(b"not-a-real-checkpoint")
    captured: dict[str, object] = {}
    fake = _FakePipe()

    class FakeSingle:
        @staticmethod
        def from_single_file(path, **kwargs):
            captured["single"] = {"path": path, **kwargs}
            return fake

    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("snapshot loader should not run for incomplete snapshot")

    import sys
    import types

    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, name: str) -> None:
            self.type = str(name).split(":")[0]
            self.name = str(name)

    fake_torch.device = FakeDevice
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoPipelineForText2Image = FakeAuto
    fake_diffusers.StableDiffusionPipeline = FakeSingle
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    kernel = SDText2ImageKernel(Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo"))
    device = Device(
        device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16
    )
    kernel.load(device)
    assert captured["single"]["path"] == str(weights)
    assert captured["single"]["config"] == str(model_dir)


def test_kernel_load_dispatches_snapshot_dir(monkeypatch, tmp_path: Path) -> None:
    from nova.kernels.sd_t2i import SDText2ImageKernel
    from nova.models import Device

    model_dir = tmp_path / "sd-turbo"
    _complete_snapshot(model_dir)
    captured: dict[str, object] = {}
    fake = _FakePipe()

    class FakeSingle:
        @staticmethod
        def from_single_file(*args, **kwargs):
            raise AssertionError("single-file loader should not run for a snapshot")

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["auto"] = {"path": path, **kwargs}
            return fake

    import sys
    import types

    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, name: str) -> None:
            self.type = str(name).split(":")[0]
            self.name = str(name)

    class FakeMatmul:
        allow_tf32 = False

    class FakeCuda:
        matmul = FakeMatmul()

    class FakeCudnn:
        benchmark = False
        allow_tf32 = False

    class FakeBackends:
        cuda = FakeCuda()
        cudnn = FakeCudnn()

    fake_torch.device = FakeDevice
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.channels_last = "channels_last"
    fake_torch.backends = FakeBackends()
    fake_torch.set_float32_matmul_precision = lambda *_args, **_kwargs: None
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoPipelineForText2Image = FakeAuto
    fake_diffusers.StableDiffusionPipeline = FakeSingle
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    kernel = SDText2ImageKernel(Settings(model_dir=model_dir, model_id="stabilityai/sd-turbo"))
    device = Device(
        device_id="cuda:0", backend="cuda", vendor="nvidia", model="N", memory_total_mb=24
    )
    kernel.load(device)
    assert captured["auto"]["path"] == str(model_dir)
    assert captured["auto"]["local_files_only"] is True
    assert captured["auto"].get("variant") == "fp16"
    assert fake.sliced is False
    assert fake.vae_sliced is True
    assert fake.vae_tiled is False
    assert fake.safety_checker is None
    assert fake.xformers is True
    assert fake.unet.memory_format == "channels_last"
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.benchmark is True


def test_adopt_local_single_file(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
    spec = importlib.util.spec_from_file_location("nova_download_model", script)
    assert spec is not None and spec.loader is not None
    download_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_model)

    src = tmp_path / "Downloads" / "sd_turbo.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"checkpoint")
    monkeypatch.setattr(download_model, "_MIN_SINGLE_FILE_BYTES", 1)
    dest_dir = tmp_path / "models" / "sd-turbo"
    monkeypatch.setattr(download_model, "SINGLE_FILE_CANDIDATES", (src,))
    adopted = download_model.adopt_local_single_file(dest_dir)
    assert adopted is not None
    assert adopted.is_file()
    assert adopted.resolve() == src.resolve()


def _install_fake_torch(monkeypatch, *, kind: str = "mps"):
    import sys
    import types

    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, name: str) -> None:
            self.type = str(name).split(":")[0]
            self.name = str(name)

        def __repr__(self) -> str:
            return self.name

    class FakeGen:
        def __init__(self, device=None) -> None:
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    class _Infer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _Cache:
        def __init__(self) -> None:
            self.empty_cache_calls = 0
            self.synchronize_calls = 0

        def empty_cache(self) -> None:
            self.empty_cache_calls += 1

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    fake_torch.device = FakeDevice
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.Generator = FakeGen
    fake_torch.inference_mode = lambda: _Infer()
    fake_torch.cuda = _Cache()
    fake_torch.mps = _Cache()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_execute_releases_mps_cache(monkeypatch, tmp_path: Path) -> None:
    from nova.clock import now_utc
    from nova.kernels.sd_t2i import SDText2ImageKernel
    from nova.models import Device, Task

    fake_torch = _install_fake_torch(monkeypatch, kind="mps")
    fake = _FakePipe()
    kernel = SDText2ImageKernel(Settings(model_dir=tmp_path, model_id="stabilityai/sd-turbo"))
    kernel._pipe = fake
    kernel._torch_device = fake_torch.device("mps")
    kernel._pretrained = "stabilityai/sd-turbo"
    device = Device(
        device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16
    )
    result = kernel.execute_sync(
        Task(
            task_id="t",
            job_id="j",
            shard_index=0,
            prompt="neon",
            seed=7,
            steps=4,
            width=512,
            height=512,
            created_at=now_utc(),
        ),
        device,
    )
    assert result.backend == "metal"
    assert result.png_bytes.startswith(b"\x89PNG")
    assert fake.calls[0]["output_type"] == "pil"
    assert fake.calls[0]["guidance_scale"] == 0.0
    assert fake_torch.mps.empty_cache_calls == 1
    assert fake_torch.cuda.empty_cache_calls == 0
    assert kernel._pipe is fake


def test_execute_oom_rewrapped_and_cache_released(monkeypatch, tmp_path: Path) -> None:
    import pytest

    from nova.clock import now_utc
    from nova.kernels.sd_t2i import SDText2ImageKernel
    from nova.models import Device, Task

    fake_torch = _install_fake_torch(monkeypatch, kind="mps")

    class BoomPipe(_FakePipe):
        def __call__(self, **kwargs):
            raise RuntimeError("MPS backend out of memory (MPS allocated 18 GB)")

    kernel = SDText2ImageKernel(Settings(model_dir=tmp_path, model_id="stabilityai/sd-turbo"))
    kernel._pipe = BoomPipe()
    kernel._torch_device = fake_torch.device("mps")
    kernel._pretrained = "stabilityai/sd-turbo"
    device = Device(
        device_id="mps:0", backend="metal", vendor="apple", model="M", memory_total_mb=16
    )
    with pytest.raises(RuntimeError, match="tile will requeue"):
        kernel.execute_sync(
            Task(
                task_id="t",
                job_id="j",
                shard_index=0,
                prompt="neon",
                seed=7,
                created_at=now_utc(),
            ),
            device,
        )
    assert fake_torch.mps.empty_cache_calls == 1
    assert kernel._pipe is not None


def test_is_oom_detects_cuda_and_mps() -> None:
    from nova.kernels.sd_t2i import _is_oom

    class OutOfMemoryError(RuntimeError):
        pass

    assert _is_oom(OutOfMemoryError("CUDA out of memory"))
    assert _is_oom(RuntimeError("MPS backend out of memory"))
    assert not _is_oom(RuntimeError("prompt too long"))
