"""Local launcher tests. These do not allocate runtimes or contact Colab."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from colab_safe_cli import invoke, report_error  # noqa: E402


class FakeRuntime:
    instances = []

    def __init__(self, connected=True):
        self._kernel_client = object() if connected else None
        self.closed = False
        self.instances.append(self)

    def stop(self, shutdown_kernel=False):
        assert shutdown_kernel is False
        self.closed = True


class FakeSocket:
    def __init__(self, *args, **kwargs):
        self.on_error = kwargs.get("on_error")


def test_cleanup_on_failure_before_user_code_and_restore_patches():
    original_runtime = FakeRuntime.__init__
    original_socket = FakeSocket.__init__

    def app():
        FakeRuntime()
        raise ConnectionError("secret-runtime-url")

    with pytest.raises(ConnectionError):
        invoke(app, FakeRuntime, FakeSocket)
    assert FakeRuntime.instances[-1].closed
    assert FakeRuntime.__init__ is original_runtime
    assert FakeSocket.__init__ is original_socket


@pytest.mark.parametrize("exit_code", [0, 1, 130])
def test_cleanup_on_cli_exit(exit_code):
    def app():
        FakeRuntime()
        raise SystemExit(exit_code)

    with pytest.raises(SystemExit) as result:
        invoke(app, FakeRuntime, FakeSocket)
    assert result.value.code == exit_code
    assert FakeRuntime.instances[-1].closed


def test_unopened_runtime_is_not_stopped():
    invoke(lambda: FakeRuntime(connected=False), FakeRuntime, FakeSocket)
    assert not FakeRuntime.instances[-1].closed


def test_diagnostics_exclude_exception_text_source_and_locals(capsys):
    secret = "DUMMY_CREDENTIAL_NEVER_OUTPUT"
    try:
        raise AttributeError(secret)
    except AttributeError as error:
        report_error("transport", error)
    captured = capsys.readouterr().err
    assert "AttributeError" in captured
    assert "test_colab_transport.py:" in captured
    assert secret not in captured
    assert "raise AttributeError" not in captured


def test_socket_error_callback_keeps_original_handler(capsys):
    handled = []

    def app():
        ws = FakeSocket(on_error=lambda socket, error: handled.append(type(error)))
        ws.on_error(ws, ValueError("DUMMY_PRIVATE_QUERY_STRING"))

    invoke(app, FakeRuntime, FakeSocket)
    assert handled == [ValueError]
    assert "DUMMY_PRIVATE_QUERY_STRING" not in capsys.readouterr().err


def test_one_cleanup_failure_does_not_skip_other_clients(capsys):
    class UnreliableRuntime(FakeRuntime):
        def stop(self):
            if self is self.instances[-2]:
                raise RuntimeError("DUMMY_PRIVATE_CLEANUP_ERROR")
            super().stop()

    invoke(lambda: (UnreliableRuntime(), UnreliableRuntime()), UnreliableRuntime, FakeSocket)
    assert UnreliableRuntime.instances[-1].closed
    assert "DUMMY_PRIVATE_CLEANUP_ERROR" not in capsys.readouterr().err
