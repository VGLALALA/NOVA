from __future__ import annotations

import asyncio

from nova.network.inprocess import InProcessBroker, InProcessTransport
from nova.network.tcp import TcpTransport
from nova.protocol import HEARTBEAT, HELLO, msg


async def test_tcp_disconnect_fires_under_2s() -> None:
    connected = asyncio.Event()
    disconnected = asyncio.Event()
    seen_connect: list[str] = []
    seen_disconnect: list[str] = []

    async def on_connect(peer_id: str) -> None:
        seen_connect.append(peer_id)
        connected.set()

    async def on_disconnect(peer_id: str) -> None:
        seen_disconnect.append(peer_id)
        disconnected.set()

    server = TcpTransport(
        listen_host="127.0.0.1",
        listen_port=0,
        connect_addrs=None,
        node_id="coord",
    )
    server.on_peer_connect(on_connect)
    server.on_peer_disconnect(on_disconnect)
    client = TcpTransport(
        listen_host=None,
        listen_port=None,
        connect_addrs=[],
        node_id="worker-1",
    )
    try:
        await server.start()
        assert server.bound_port is not None
        client.connect_addrs = [("127.0.0.1", server.bound_port)]
        await client.start()
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        assert "worker-1" in seen_connect

        client._stopping = True
        peers = list(client._peers.values())
        for peer in peers:
            transport = peer.writer.transport
            if transport is not None:
                transport.abort()

        await asyncio.wait_for(disconnected.wait(), timeout=2.0)
        assert seen_disconnect
        assert seen_disconnect[0] in {"worker-1", seen_connect[0]}
    finally:
        await client.stop()
        await server.stop()


async def test_tcp_hello_and_send() -> None:
    got = asyncio.Event()
    messages: list[tuple[str, str]] = []

    async def on_message(peer_id: str, env) -> None:
        messages.append((peer_id, env.type))
        if env.type == HEARTBEAT:
            got.set()

    server = TcpTransport(
        listen_host="127.0.0.1",
        listen_port=0,
        connect_addrs=None,
        node_id="coord",
    )
    server.on_message(on_message)
    connected = asyncio.Event()
    client_ready = asyncio.Event()
    server.on_peer_connect(lambda _pid: connected.set())
    client = TcpTransport(
        listen_host=None,
        listen_port=None,
        connect_addrs=[],
        node_id="worker-1",
    )
    client.on_peer_connect(lambda _pid: client_ready.set())
    try:
        await server.start()
        assert server.bound_port is not None
        client.connect_addrs = [("127.0.0.1", server.bound_port)]
        await client.start()
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        await asyncio.wait_for(client_ready.wait(), timeout=2.0)
        await client.send("coord", msg(HEARTBEAT, "worker-1", node_id="worker-1"))
        await asyncio.wait_for(got.wait(), timeout=2.0)
        types = [t for _, t in messages]
        assert HELLO in types
        assert HEARTBEAT in types
        assert any(pid == "worker-1" for pid, _ in messages)
    finally:
        await client.stop()
        await server.stop()


async def test_inprocess_stop_fires_disconnect() -> None:
    broker = InProcessBroker()
    coord = InProcessTransport(broker, "coord")
    worker = InProcessTransport(broker, "worker-1")
    disconnected = asyncio.Event()
    seen: list[str] = []

    async def on_disconnect(peer_id: str) -> None:
        seen.append(peer_id)
        disconnected.set()

    coord.on_peer_disconnect(on_disconnect)
    await coord.start()
    await worker.start()
    await worker.send("coord", msg(HEARTBEAT, "worker-1"))
    await worker.stop()
    await asyncio.wait_for(disconnected.wait(), timeout=1.0)
    assert seen == ["worker-1"]
    await coord.stop()
