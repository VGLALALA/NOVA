"""Environment + CLI config. LAN-only. No cloud defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOVA_", env_file=".env", extra="ignore")

    role: str = "coordinator"  # coordinator | worker
    swarm_topic: str = "nova-htn-2026"
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    control_host: str = "0.0.0.0"
    control_port: int = 7946
    coordinator_url: str | None = None
    peers: str | None = None  # comma-separated host:port
    data_dir: Path = Path(".nova")
    model_id: str = "stabilityai/sd-turbo"
    model_dir: Path = Path("models/sd-turbo")
    kernel: str = "sd.t2i.v1"
    dummy: bool = False
    lease_min_s: int = 20
    lease_max_s: int = 90
    heartbeat_s: float = 5.0
    suspect_s: float = 15.0
    offline_s: float = 30.0
    progress_s: float = 2.0
    default_estimate_s: float = 15.0
    max_attempts: int = 3
    advertise_host: str | None = None  # LAN IP printed / announced

    def peer_list(self) -> list[tuple[str, int]]:
        if not self.peers:
            return []
        out: list[tuple[str, int]] = []
        for item in self.peers.split(","):
            item = item.strip()
            if not item:
                continue
            host, _, port = item.partition(":")
            out.append((host, int(port or self.control_port)))
        return out

    def public_http_url(self) -> str:
        host = self.advertise_host or "127.0.0.1"
        return f"http://{host}:{self.http_port}"


def load_settings(**overrides: object) -> Settings:
    s = Settings()
    return s.model_copy(update={k: v for k, v in overrides.items() if v is not None})
