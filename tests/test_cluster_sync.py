from __future__ import annotations

import pytest

from nova.clock import Clock
from nova.config import Settings
from nova.coordinator import Coordinator
from nova.events import EventBus
from nova.models import Device, NodeIdentity, NodeManifest
from nova.clock import now_utc
from nova.protocol import CLUSTER_SYNC, HELLO, msg
from tests.test_coordinator_messages import FakeScheduler, FakeStore, FakeTransport


def _coord(tmp_settings: Settings | None = None) -> tuple[Coordinator, FakeStore, FakeTransport]:
    store = FakeStore()
    bus = EventBus()
    transport = FakeTransport()
    clock = Clock()
    settings = tmp_settings or Settings(advertise_host="10.0.0.1", control_port=7946)
    ident = NodeIdentity(node_id="mac", created_at=now_utc())
    sched = FakeScheduler(store, clock)
    coord = Coordinator(store, sched, transport, bus, settings, clock, ident)
    store.put_node(
        NodeManifest(
            node_id="mac",
            hostname="mac",
            status="online",
            http_url="http://10.0.0.1:8080",
            control_host="10.0.0.1",
            control_port=7946,
            devices=[
                Device(
                    device_id="mps:0",
                    backend="metal",
                    vendor="apple",
                    model="M3 Max",
                    memory_total_mb=18432,
                )
            ],
        )
    )
    return coord, store, transport


@pytest.mark.asyncio
async def test_hello_sends_cluster_roster() -> None:
    coord, store, transport = _coord()
    await transport.inject("peer-cuda", msg(HELLO, "cuda", node_id="cuda", protocol_version=1))
    assert any(env.type == CLUSTER_SYNC for _, env in transport.sent)
    sync = next(env for _, env in transport.sent if env.type == CLUSTER_SYNC)
    ids = [n["node_id"] for n in sync.payload["nodes"]]
    assert "mac" in ids
    got = store.get_node("cuda")
    assert got is not None
    assert got.status == "online"
    assert store.get_last_seen("cuda") is not None


@pytest.mark.asyncio
async def test_cluster_sync_ingests_new_node() -> None:
    coord, store, transport = _coord()
    incoming = NodeManifest(
        node_id="cuda",
        hostname="runpod",
        status="online",
        http_url="http://10.128.1.2:8080",
        control_host="0.tcp.ngrok.io",
        control_port=12345,
        devices=[
            Device(
                device_id="cuda:0",
                backend="cuda",
                vendor="nvidia",
                model="RTX 4000 Ada Generation",
                memory_total_mb=20019,
            )
        ],
    )
    await transport.inject(
        "peer-cuda",
        msg(CLUSTER_SYNC, "cuda", nodes=[incoming.model_dump(mode="json")]),
    )
    got = store.get_node("cuda")
    assert got is not None
    assert got.devices[0].backend == "cuda"
    assert got.control_host == "0.tcp.ngrok.io"


def test_is_self_control_skips_loopback() -> None:
    coord, _, transport = _coord()
    transport.bound_port = 7946
    transport.bound_host = "0.0.0.0"
    assert coord._is_self_control("127.0.0.1", 7946)
    assert coord._is_self_control("10.0.0.1", 7946)
    assert not coord._is_self_control("0.tcp.ngrok.io", 12345)
    assert not coord._should_dial_control("10.128.1.2", 7946)
    assert coord._should_dial_control("0.tcp.ngrok.io", 12345)
