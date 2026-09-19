"""Persistent node identity. Random ids on restart are a demo bug."""

from __future__ import annotations

import secrets
from pathlib import Path

from nova.clock import now_utc
from nova.models import NodeIdentity


def identity_path(data_dir: Path) -> Path:
    return Path(data_dir) / "node.json"


def load_or_create(data_dir: Path) -> NodeIdentity:
    path = identity_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return NodeIdentity.model_validate_json(path.read_text(encoding="utf-8"))
    ident = NodeIdentity(node_id=f"nova-{secrets.token_hex(4)}", created_at=now_utc())
    path.write_text(ident.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return ident
