#!/usr/bin/env python3
"""Colab 0.7.4 launcher with unconditional local websocket cleanup.

Run with the Python environment containing google-colab-cli. This does not
change auth, session selection, TLS verification, or remote kernel lifetime.
"""
from __future__ import annotations

import os
import sys


def safe_http_status(error):
    """Extract numeric transport status only; never retain response objects."""
    from urllib.error import HTTPError
    status = getattr(error, 'http_status', None)
    if status is None:
        status = (error.code if isinstance(error, HTTPError) else
                  getattr(getattr(error, 'response', None), 'status_code', None))
    return status if type(status) is int and 100 <= status <= 599 else None


def _exception_chain(error):
    """Follow only exception links, bounded and cycle-safe; None means overflow."""
    from urllib.error import URLError
    pending, seen, chain = [error], set(), []
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        if len(seen) >= 16:
            return None
        seen.add(id(current))
        chain.append(current)
        pending.extend((current.__context__, current.__cause__))
        if isinstance(current, URLError) and isinstance(current.reason, BaseException):
            pending.append(current.reason)
    return chain


def _transport_chain_security_blocked(error):
    """Security failures anywhere in the bounded exception chain fail closed."""
    chain = _exception_chain(error)
    if chain is None:
        return True
    for current in chain:
        classes = {base.__name__ for base in type(current).__mro__}
        if (classes & {'SSLError', 'SSLCertVerificationError', 'CertificateError',
                       'PermissionError'} or safe_http_status(current) in {401, 403, 407, 429}):
            return True
    return False


def retryable_transport_error(error):
    """Recognized transport outages only, excluding denials and TLS failures."""
    from urllib.error import URLError
    if _transport_chain_security_blocked(error):
        return False
    if getattr(error, 'transport_retryable', None) is False:
        return False
    status = safe_http_status(error)
    if status is not None:
        return status in {408, 500, 502, 503, 504}
    if isinstance(error, URLError):
        return isinstance(error.reason, (ConnectionError, TimeoutError))
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    classes = {(base.__module__.split('.')[0], base.__name__)
               for base in type(error).__mro__}
    if any(name in {'SSLError', 'SSLCertVerificationError'} for _, name in classes):
        return False
    return bool(classes & {('requests', 'ConnectionError'), ('requests', 'Timeout'),
                           ('httpx', 'TransportError'),
                           ('websocket', 'WebSocketConnectionClosedException'),
                           ('websocket', 'WebSocketTimeoutException')})


def sanitized_controller_error(error, phase, message):
    """Return safe typed evidence without URLs, credentials or vendor objects.

    Callers supply fixed phase/message literals. A transient type allows the
    checkpoint watcher's reconciliation loop to retry without parsing text.
    """
    if phase not in {'reconnect', 'refresh', 'execute', 'download'}:
        raise ValueError('Unknown controller error phase')
    chain = _exception_chain(error)
    statuses = [safe_http_status(item) for item in chain or [error]]
    status = next((item for item in statuses if item in {401, 403, 407, 429}),
                  next((item for item in statuses if item is not None), None))
    suffix = f' HTTP {status}' if status is not None else ''
    retryable = retryable_transport_error(error)
    error_type = ConnectionError if retryable else RuntimeError
    safe = error_type(f'{message}; phase={phase}{suffix}; private details suppressed')
    safe.phase = phase
    safe.transport_retryable = retryable
    safe.http_status = status
    return safe


def report_error(label: str, error: BaseException) -> None:
    """Report locations, never exception text, source lines, URLs, or frame locals."""
    status = safe_http_status(error)
    # Print only a validated numeric status. Exception strings, response bodies,
    # headers and request URLs can contain runtime credentials.
    phase = getattr(error, "phase", None)
    phase_suffix = f" phase={phase}" if phase in {"reconnect", "refresh", "execute", "download"} else ""
    suffix = f" HTTP {status}" if type(status) is int and 100 <= status <= 599 else ""
    print(f"[picoagent {label}] {type(error).__name__}{phase_suffix}{suffix}", file=sys.stderr)
    tb = error.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        location = f"{os.path.basename(code.co_filename)}:{tb.tb_lineno} {code.co_name}"
        print(f"[picoagent {label}] {location}", file=sys.stderr)
        tb = tb.tb_next


def invoke(app, runtime_class, websocket_class) -> None:
    """Keep cleanup outside the vendor CLI's initial execute/cwd operation."""
    runtimes = []
    original_init = runtime_class.__init__
    original_ws_init = websocket_class.__init__

    def runtime_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        runtimes.append(self)

    def ws_init(self, *args, **kwargs):
        original_error = kwargs.get("on_error")

        def on_error(ws, error):
            report_error("transport", error)
            if original_error:
                original_error(ws, error)

        kwargs["on_error"] = on_error
        original_ws_init(self, *args, **kwargs)

    runtime_class.__init__ = runtime_init
    websocket_class.__init__ = ws_init
    try:
        app()
    finally:
        for runtime in runtimes:
            if runtime._kernel_client is not None:
                try:
                    runtime.stop()
                except Exception as error:
                    report_error("cleanup", error)
        runtime_class.__init__ = original_init
        websocket_class.__init__ = original_ws_init


def main() -> None:
    from colab_cli.cli import app
    from colab_cli.runtime import ColabRuntime
    import websocket

    try:
        invoke(app, ColabRuntime, websocket.WebSocketApp)
    except Exception as error:
        # Typer's rich traceback can expose runtime URLs and credentials in locals.
        report_error("cli", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
