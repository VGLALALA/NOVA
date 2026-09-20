from nova.config import Settings
from nova.main import _make_control, _make_pear, _make_tcp
from nova.network.hub import Hub
from nova.network.pear import PearTransport
from nova.network.tcp import TcpTransport


def test_make_tcp_listen_and_connect() -> None:
    settings = Settings(control_host="127.0.0.1", control_port=7946)
    listen = _make_tcp(node_id="c", listen=True, connect_addrs=[], settings=settings)
    assert isinstance(listen, TcpTransport)
    assert listen.listen_host == "127.0.0.1"
    worker = _make_tcp(
        node_id="w",
        listen=False,
        connect_addrs=[("127.0.0.1", 7946)],
        settings=settings,
    )
    assert isinstance(worker, TcpTransport)
    assert worker.listen_host is None
    assert ("127.0.0.1", 7946) in worker.connect_addrs


def test_make_control_tcp_only_when_pear_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(PearTransport, "available", classmethod(lambda cls: False))
    transport, kind = _make_control(
        node_id="c",
        listen=True,
        connect_addrs=[],
        settings=Settings(),
        include_pear=True,
    )
    assert kind == "tcp"
    assert isinstance(transport, TcpTransport)


def test_make_control_hub_when_pear_available(monkeypatch) -> None:
    monkeypatch.setattr(PearTransport, "available", classmethod(lambda cls: True))
    transport, kind = _make_control(
        node_id="c",
        listen=True,
        connect_addrs=[],
        settings=Settings(swarm_topic="demo"),
        include_pear=True,
    )
    assert kind == "pear+tcp"
    assert isinstance(transport, Hub)
    assert any(isinstance(t, TcpTransport) for t in transport._transports)
    assert any(isinstance(t, PearTransport) for t in transport._transports)
    pear = next(t for t in transport._transports if isinstance(t, PearTransport))
    assert pear.swarm_topic == "demo"


def test_local_worker_skips_pear(monkeypatch) -> None:
    monkeypatch.setattr(PearTransport, "available", classmethod(lambda cls: True))
    transport, kind = _make_control(
        node_id="local",
        listen=False,
        connect_addrs=[("127.0.0.1", 7946)],
        settings=Settings(),
        include_pear=False,
    )
    assert kind == "tcp"
    assert isinstance(transport, TcpTransport)


def test_worker_pear_without_typed_peers(monkeypatch) -> None:
    monkeypatch.setattr(PearTransport, "available", classmethod(lambda cls: True))
    transport, kind = _make_control(
        node_id="w",
        listen=False,
        connect_addrs=[],
        settings=Settings(),
        include_pear=True,
    )
    assert kind == "pear+tcp"
    assert isinstance(transport, Hub)


def test_make_pear_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(PearTransport, "available", classmethod(lambda cls: False))
    assert _make_pear(node_id="n", settings=Settings()) is None


def test_hub_bound_port_follows_tcp() -> None:
    tcp = TcpTransport(
        listen_host="127.0.0.1",
        listen_port=7946,
        connect_addrs=[],
        node_id="c",
    )
    tcp.bound_port = 7947
    tcp.bound_host = "127.0.0.1"
    hub = Hub([tcp])
    assert hub.bound_port == 7947
    assert hub.bound_host == "127.0.0.1"
