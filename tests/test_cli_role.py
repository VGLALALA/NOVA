from nova.config import Settings
from nova.main import dashboard_only_mode, worker_mode


def test_cli_worker_flag_wins() -> None:
    assert worker_mode(True, Settings(role="coordinator")) is True


def test_env_role_worker_without_flag() -> None:
    assert worker_mode(False, Settings(role="worker")) is True
    assert worker_mode(False, Settings(role="WORKER")) is True


def test_default_role_is_coordinator() -> None:
    assert worker_mode(False, Settings(role="coordinator")) is False
    assert worker_mode(False, Settings()) is False


def test_dashboard_flag_skips_local_worker() -> None:
    assert dashboard_only_mode(True, False, Settings(role="coordinator")) is True
    assert dashboard_only_mode(False, True, Settings()) is True
    assert dashboard_only_mode(False, False, Settings(role="dashboard")) is True


def test_plain_start_still_runs_local_worker() -> None:
    assert dashboard_only_mode(False, False, Settings()) is False
    assert dashboard_only_mode(False, False, Settings(role="coordinator")) is False


def test_public_url_overrides_advertised_http() -> None:
    s = Settings(advertise_host="10.0.0.1", http_port=8080, public_url="https://demo.ngrok-free.app/")
    assert s.public_http_url() == "https://demo.ngrok-free.app"
    assert Settings().public_http_url() == "http://127.0.0.1:8080"


def test_peer_list_parses_tcp_url() -> None:
    s = Settings(peers="tcp://4.tcp.ngrok.io:29805")
    assert s.peer_list() == [("4.tcp.ngrok.io", 29805)]
    s2 = Settings(peers="192.168.1.9:7946")
    assert s2.peer_list() == [("192.168.1.9", 7946)]


def test_coordinator_parse_tcp_url() -> None:
    from nova.clock import Clock, now_utc
    from nova.coordinator import Coordinator
    from nova.events import EventBus
    from nova.models import NodeIdentity
    from tests.test_coordinator_messages import FakeScheduler, FakeStore, FakeTransport

    store = FakeStore()
    clock = Clock()
    coord = Coordinator(
        store,
        FakeScheduler(store, clock),
        FakeTransport(),
        EventBus(),
        Settings(),
        clock,
        NodeIdentity(node_id="n", created_at=now_utc()),
    )
    assert coord._parse_tcp_target("tcp://4.tcp.ngrok.io:29805") == ("4.tcp.ngrok.io", 29805)
    assert coord._parse_tcp_target("4.tcp.ngrok.io", 29805) == ("4.tcp.ngrok.io", 29805)
