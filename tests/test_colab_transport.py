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


@pytest.mark.parametrize('wrapper', ['builtin', 'httpx', 'requests', 'urllib'])
@pytest.mark.parametrize('link', ['cause', 'context'])
def test_wrapped_certificate_failures_never_become_retryable(wrapper, link):
    import ssl
    from urllib.error import URLError
    from colab_safe_cli import retryable_transport_error, sanitized_controller_error
    certificate = ssl.SSLCertVerificationError('PRIVATE_CERT_URL')
    if wrapper == 'builtin':
        error = ConnectionError('PRIVATE_URL')
    elif wrapper == 'httpx':
        httpx = pytest.importorskip('httpx')
        error = httpx.ConnectError('PRIVATE_URL')
    elif wrapper == 'requests':
        requests = pytest.importorskip('requests')
        error = requests.ConnectionError('PRIVATE_URL')
    else:
        error = URLError(certificate)
    setattr(error, '__' + link + '__', certificate)
    assert not retryable_transport_error(error)
    safe = sanitized_controller_error(error, 'download', 'Download failed')
    assert isinstance(safe, RuntimeError)
    assert not retryable_transport_error(safe)
    assert 'PRIVATE' not in str(safe)


def test_transport_chain_is_cycle_safe_and_budget_fails_closed():
    from colab_safe_cli import retryable_transport_error
    first, second = ConnectionError('first'), ConnectionError('second')
    first.__cause__, second.__context__ = second, first
    assert retryable_transport_error(first)
    root = current = ConnectionError('root')
    for _ in range(17):
        current.__cause__ = ConnectionError('nested')
        current = current.__cause__
    assert not retryable_transport_error(root)


def test_wrapped_permission_beats_outer_transient_http_status():
    from types import SimpleNamespace
    from colab_safe_cli import retryable_transport_error
    error = ConnectionError('PRIVATE_URL')
    error.response = SimpleNamespace(status_code=503)
    error.__cause__ = PermissionError('PRIVATE_DENIAL')
    assert not retryable_transport_error(error)
    from colab_safe_cli import sanitized_controller_error
    safe = sanitized_controller_error(error, 'download', 'Download failed')
    assert safe.http_status == 503
    assert not retryable_transport_error(safe)


@pytest.mark.parametrize('status', [401, 403, 407, 429])
def test_nested_http_denials_never_retry(status):
    from types import SimpleNamespace
    from colab_safe_cli import retryable_transport_error, sanitized_controller_error
    inner = RuntimeError('PRIVATE_RESPONSE')
    inner.response = SimpleNamespace(status_code=status)
    outer = ConnectionError('PRIVATE_URL')
    outer.__cause__ = inner
    assert not retryable_transport_error(outer)
    safe = sanitized_controller_error(outer, 'refresh', 'Refresh failed')
    assert safe.http_status == status
    assert not retryable_transport_error(safe)
    assert 'PRIVATE' not in str(safe)
