#!/usr/bin/env python3
"""Colab 0.7.4 launcher with unconditional local websocket cleanup.

Run with the Python environment containing google-colab-cli. This does not
change auth, session selection, TLS verification, or remote kernel lifetime.
"""
from __future__ import annotations

import os
import sys


def report_error(label: str, error: BaseException) -> None:
    """Report locations, never exception text, source lines, URLs, or frame locals."""
    print(f"[picoagent {label}] {type(error).__name__}", file=sys.stderr)
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
