#!/usr/bin/env python3
"""Colab 0.7.4 launcher with unconditional local websocket cleanup.

Run with the Python environment containing google-colab-cli. This does not
change auth, session selection, TLS verification, or remote kernel lifetime.
"""
from __future__ import annotations

import sys
import re


def main() -> None:
    from colab_cli.runtime import ColabRuntime
    from colab_cli.cli import app
    import websocket

    runtimes = []
    original_init = ColabRuntime.__init__
    original_ws_init = websocket.WebSocketApp.__init__

    def runtime_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        runtimes.append(self)

    def ws_init(self, *args, **kwargs):
        original_error = kwargs.get("on_error")

        def on_error(ws, error):
            # Type only: exception strings can contain private proxy URLs/tokens.
            print(f"[picoagent transport] {type(error).__name__}", file=sys.stderr)
            if isinstance(error, AttributeError):
                match = re.search(r"'([A-Za-z_][A-Za-z_0-9.]*)' object has no attribute '([A-Za-z_][A-Za-z_0-9]*)'", str(error))
                if match:
                    print(f"[picoagent transport] {match.group(1)} missing {match.group(2)}", file=sys.stderr)
            if original_error:
                original_error(ws, error)

        kwargs["on_error"] = on_error
        original_ws_init(self, *args, **kwargs)

    ColabRuntime.__init__ = runtime_init
    websocket.WebSocketApp.__init__ = ws_init
    try:
        app()
    finally:
        for runtime in runtimes:
            if runtime._kernel_client is not None:
                try:
                    runtime.stop()
                except Exception as error:
                    print(f"[picoagent cleanup] {type(error).__name__}", file=sys.stderr)


if __name__ == "__main__":
    main()
