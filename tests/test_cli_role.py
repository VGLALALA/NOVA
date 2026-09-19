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
