from nova.config import Settings
from nova.main import worker_mode


def test_cli_worker_flag_wins() -> None:
    assert worker_mode(True, Settings(role="coordinator")) is True


def test_env_role_worker_without_flag() -> None:
    assert worker_mode(False, Settings(role="worker")) is True
    assert worker_mode(False, Settings(role="WORKER")) is True


def test_default_role_is_coordinator() -> None:
    assert worker_mode(False, Settings(role="coordinator")) is False
    assert worker_mode(False, Settings()) is False
