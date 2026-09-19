#!/usr/bin/env python3
"""Download stabilityai/sd-turbo into models/sd-turbo.

Run this once per machine, with internet. At demo time the worker MUST load
with local_files_only=True — do not hit Hugging Face during judging.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "stabilityai/sd-turbo"
SINGLE_FILE_CANDIDATES = (
    Path.home() / "Downloads" / "sd_turbo.safetensors",
    Path.home() / "Downloads" / "sd-turbo.safetensors",
)
_MIN_SINGLE_FILE_BYTES = 1_000_000_000


def _ensure_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("huggingface_hub missing — installing…", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src)
    except OSError:
        import shutil

        shutil.copy2(src, dest)


def adopt_local_single_file(dest_dir: Path) -> Path | None:
    """Use an already-downloaded sd_turbo.safetensors if present."""
    dest_file = dest_dir / "sd_turbo.safetensors"
    if dest_file.is_file() and dest_file.stat().st_size > _MIN_SINGLE_FILE_BYTES:
        return dest_file
    for candidate in SINGLE_FILE_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > _MIN_SINGLE_FILE_BYTES:
            _link_or_copy(candidate, dest_file)
            return dest_file
    return None


def snapshot_ready(dest_dir: Path) -> bool:
    from nova.kernels.sd_t2i import snapshot_complete

    return snapshot_complete(dest_dir)


def convert_single_file(src: Path, dest_dir: Path) -> None:
    """Materialize a local HF snapshot so demo loads stay offline.

    from_single_file still needs tokenizer/config files. Pass the local
    stabilityai/sd-turbo configs when present so conversion does not hit
    gated SD 2.1 repos.
    """
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"Converting {src} → snapshot {dest_dir}")
    kwargs: dict = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if (dest_dir / "model_index.json").is_file():
        kwargs["config"] = str(dest_dir)
        kwargs["local_files_only"] = True
    pipe = StableDiffusionPipeline.from_single_file(str(src), **kwargs)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(dest_dir), safe_serialization=True)
    del pipe


def main() -> int:
    dest = Path(__file__).resolve().parent.parent / "models" / "sd-turbo"
    dest.mkdir(parents=True, exist_ok=True)

    if snapshot_ready(dest):
        print(f"Local snapshot already present → {dest}")
        print("Do not hit Hugging Face during judging.")
        return 0

    adopted = adopt_local_single_file(dest)
    if adopted is not None:
        print(f"Using local single-file checkpoint → {adopted}")
        try:
            convert_single_file(adopted, dest)
            print()
            print(f"Snapshot written → {dest}")
            print("Demo loads this with local_files_only=True.")
            return 0
        except Exception as exc:
            print(f"Snapshot conversion failed ({exc}).", file=sys.stderr)
            print("NOVA can still try from_single_file at load time.", file=sys.stderr)
            print("That path may fetch tokenizer configs unless they are cached.")
            return 0

    _ensure_hub()
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO} → {dest}")
    snapshot_download(repo_id=REPO, local_dir=str(dest))
    print()
    print("Done. Weights are on disk.")
    print()
    print("Reminder: load with local_files_only=True at demo time.")
    print("  pipe = AutoPipelineForText2Image.from_pretrained(")
    print(f"      {str(dest)!r},")
    print("      local_files_only=True,")
    print("      torch_dtype=torch.float16,")
    print("  )")
    print("Do not hit Hugging Face during judging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
