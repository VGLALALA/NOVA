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


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None
        self.sliced = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_attention_slicing(self) -> None:
        self.sliced = True

    def set_progress_bar_config(self, **kwargs) -> None:
        return None


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
        device_id="cuda:0", backend="cuda", vendor="nvidia", model="N", memory_total_mb=24
    )
    kernel.load(device)
    assert captured["auto"]["path"] == str(model_dir)
    assert captured["auto"]["local_files_only"] is True
    assert fake.sliced is False


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
