from __future__ import annotations

from pathlib import Path

from nova.identity import identity_path, load_or_create


def test_load_or_create_writes_node_json(data_dir: Path) -> None:
    ident = load_or_create(data_dir)
    path = identity_path(data_dir)
    assert path == data_dir / "node.json"
    assert path.is_file()
    assert ident.node_id.startswith("nova-")
    assert len(ident.node_id) == len("nova-") + 8
    raw = path.read_text(encoding="utf-8")
    assert ident.node_id in raw
    assert "created_at" in raw


def test_second_call_same_node_id(data_dir: Path) -> None:
    first = load_or_create(data_dir)
    second = load_or_create(data_dir)
    third = load_or_create(data_dir)
    assert first.node_id == second.node_id == third.node_id
    assert first.created_at == second.created_at == third.created_at


def test_load_or_create_never_changes_file(data_dir: Path) -> None:
    first = load_or_create(data_dir)
    path = identity_path(data_dir)
    before = path.read_text(encoding="utf-8")
    again = load_or_create(data_dir)
    after = path.read_text(encoding="utf-8")
    assert again.node_id == first.node_id
    assert after == before
