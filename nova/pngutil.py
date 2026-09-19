"""Small PNG helpers for the HTTP data plane and dummy kernel."""

from __future__ import annotations

import hashlib
import io

from PIL import Image

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_png(data: bytes) -> Image.Image:
    if not data or not data.startswith(PNG_MAGIC):
        raise ValueError("not a PNG")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ValueError("not a PNG") from exc
    return img


def validate_png(
    data: bytes,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int]:
    if not data:
        raise ValueError("empty PNG")
    img = decode_png(data)
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("empty PNG")
    if width is not None and w != width:
        raise ValueError(f"expected width {width}, got {w}")
    if height is not None and h != height:
        raise ValueError(f"expected height {height}, got {h}")
    return w, h
