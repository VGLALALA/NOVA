from pathlib import Path

import yaml

GALLERY = Path(__file__).resolve().parents[1] / "demo" / "gallery.yaml"


def test_gallery_yaml_exists() -> None:
    assert GALLERY.is_file()


def test_gallery_yaml_contract() -> None:
    data = yaml.safe_load(GALLERY.read_text())
    prompts = data["prompts"]
    assert len(prompts) == 24

    seeds = [p["seed"] for p in prompts]
    assert seeds == list(range(1000, 1024))
    assert len(set(seeds)) == 24

    assert data["model"]["steps"] == 4
    assert data["model"]["width"] == 512
    assert data["model"]["height"] == 512
    assert data["job"]["kernel"] == "sd.t2i.v1"
